import asyncio
import json
import logging
import random
from typing import Dict, List, Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from mindsearch.agent import init_agent
import os

log = logging.getLogger("mindsearch.app")


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(description="MindSearch API")
    parser.add_argument("--host", default="0.0.0.0", type=str, help="Service host")
    parser.add_argument("--port", default=8002, type=int, help="Service port")
    parser.add_argument("--llm_url", default=None, type=str,
                        help="LLM base URL (default: LLM_URL env or http://localhost:8080/v1)")
    parser.add_argument("--search_engine", default="DuckDuckGoSearch", type=str,
                        help="Search engine",
                        choices=["DuckDuckGoSearch", "GoogleSearch", "BraveSearch"])
    return parser.parse_args()


args = parse_arguments()
app = FastAPI(docs_url="/")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerationParams(BaseModel):
    inputs: Union[str, List[Dict]]
    session_id: int = Field(default_factory=lambda: random.randint(0, 999999))
    agent_cfg: Dict = dict()


def _build_adjacency_list(nodes):
    """Build the adjacency_list structure the React frontend expects.

    Format: {
        "root": [{"name": "n1", "id": 1, "state": N}, ...],
        "n1":   [{"name": "response", "id": 100, "state": 0}],  # once done
        ...
    }
    state: 0 = not started, 1 = in progress, 3 = complete
    """
    adj = {}
    root_children = []
    for i, node in enumerate(nodes):
        nid = node["id"]
        state = 3 if node["status"] == "done" else 1
        root_children.append({
            "name": nid,
            "id": i + 1,
            "state": state,
            "depends_on": node.get("depends_on", []),
        })
        if node["status"] == "done":
            # Add a "response" child to signal completion to the frontend
            adj[nid] = [{"name": "response", "id": 100 + i, "state": 0}]
    adj["root"] = root_children
    return adj


def _dedup_nodes(nodes):
    """Last-write-wins dedup by node id."""
    seen = {}
    for n in nodes:
        seen[n["id"]] = n
    return list(seen.values())


def _make_legacy_event(current_node=None, thought=None, adj_list=None,
                       node_stream_state=1, chat_is_over=False,
                       node_content=None, ref2url=None,
                       action=None, search_content=None):
    """Build an SSE event in the format the React frontend expects.

    IMPORTANT: The top-level response.stream_state controls the entire
    conversation lifecycle in the frontend:
      stream_state=1 → event is processed by formatData()
      stream_state=0 → setChatIsOver(true), formatData is SKIPPED

    So top-level stream_state must be 1 for ALL events except the very
    last "conversation done" event. Individual node completion is signaled
    via the adjacency_list structure (adding a "response" child).
    """
    # Top-level stream_state: 0 only for the final "done" signal
    top_stream_state = 0 if chat_is_over else 1

    formatted = {}
    if thought is not None:
        formatted["thought"] = thought
    if adj_list is not None:
        formatted["adjacency_list"] = adj_list
    if action is not None:
        formatted["action"] = action
    if ref2url is not None:
        formatted["ref2url"] = ref2url

    if current_node and current_node not in ("root", "response"):
        # Wrap as a node update
        node_data = {}
        if node_content is not None:
            node_data["content"] = node_content
        node_response = {
            "content": json.dumps(search_content) if search_content else None,
            "formatted": formatted,
            "stream_state": node_stream_state,
        }
        node_data["response"] = node_response
        formatted_outer = {
            "node": {current_node: node_data},
        }
        if adj_list is not None:
            formatted_outer["adjacency_list"] = adj_list
        return {
            "current_node": current_node,
            "response": {
                "content": {"current_node": current_node},
                "formatted": formatted_outer,
                "stream_state": top_stream_state,
            },
        }
    else:
        # Root-level event (planner thought or final response)
        return {
            "current_node": None,
            "response": {
                "content": None,
                "formatted": formatted,
                "stream_state": top_stream_state,
            },
        }


async def run(request: GenerationParams, _request: Request):
    async def generate():
        try:
            llm_url = args.llm_url or os.getenv("LLM_URL", "http://localhost:8080/v1")
            search_engine = os.getenv("SEARCH_ENGINE", args.search_engine)

            agent_graph = init_agent(
                llm_url=llm_url,
                search_engine=search_engine,
            )

            initial_state = {
                "question": inputs,
                "nodes": [],
                "final_answer": "",
                "turns": 0,
            }

            # Track all nodes across events for adjacency list building
            all_nodes = []

            async for event in agent_graph.astream(initial_state, stream_mode="updates"):
                for node_name, update in event.items():
                    log.info("SSE event: node=%s keys=%s", node_name, list(update.keys()) if isinstance(update, dict) else type(update))

                    if node_name == "planner":
                        new_nodes = update.get("nodes", [])
                        if new_nodes:
                            all_nodes.extend(new_nodes)
                            deduped = _dedup_nodes(all_nodes)
                            adj = _build_adjacency_list(deduped)

                            # Emit adjacency list update
                            evt = _make_legacy_event(
                                adj_list=adj,
                                thought="Searching...",
                            )
                            yield {"data": json.dumps(evt, ensure_ascii=False)}

                            # Emit individual node creation events
                            for n in new_nodes:
                                node_evt = _make_legacy_event(
                                    current_node=n["id"],
                                    node_content=n["query"],
                                    thought="Searching...",
                                    action={"parameters": {"query": [n["query"]]}},
                                    adj_list=adj,
                                )
                                yield {"data": json.dumps(node_evt, ensure_ascii=False)}

                    elif node_name == "searcher":
                        done_nodes = update.get("nodes", [])
                        for n in done_nodes:
                            # Update tracked nodes
                            all_nodes.append(n)
                            deduped = _dedup_nodes(all_nodes)
                            adj = _build_adjacency_list(deduped)

                            # Emit node completion with summary as conclusion
                            node_evt = _make_legacy_event(
                                current_node=n["id"],
                                node_content=n["query"],
                                thought=n.get("summary", ""),
                                adj_list=adj,
                                node_stream_state=0,
                            )
                            yield {"data": json.dumps(node_evt, ensure_ascii=False)}

                    elif node_name == "finalize":
                        answer = update.get("final_answer", "")
                        deduped = _dedup_nodes(all_nodes)
                        adj = _build_adjacency_list(deduped)

                        # Emit final response
                        evt = _make_legacy_event(
                            thought=answer,
                            adj_list=adj,
                        )
                        yield {"data": json.dumps(evt, ensure_ascii=False)}

                        # Emit completion signal
                        done_evt = _make_legacy_event(
                            thought=answer,
                            adj_list=adj,
                            chat_is_over=True,
                        )
                        yield {"data": json.dumps(done_evt, ensure_ascii=False)}

                    if await _request.is_disconnected():
                        return

        except Exception as exc:
            msg = "An error occurred while generating the response."
            logging.exception(msg)
            response_json = json.dumps(
                dict(error=dict(msg=msg, details=str(exc))), ensure_ascii=False
            )
            yield {"data": response_json}

    inputs = request.inputs
    return EventSourceResponse(generate(), ping=300)


app.add_api_route("/solve", run, methods=["POST"])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
