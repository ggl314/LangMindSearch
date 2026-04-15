import asyncio
import json
import logging
import random
from typing import Any, Dict, List, Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from mindsearch.agent import init_agent
from mindsearch import history_db
import os

log = logging.getLogger("mindsearch.app")

# Registry of active /solve runs: session_id → asyncio.Event (set to stop).
_stop_events: dict[int, asyncio.Event] = {}


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
    # Multi-turn graph extension: prior completed SearchNode dicts + root question.
    # When prior_nodes is non-empty, the graph resumes from the existing research.
    prior_nodes: List[Dict] = Field(default_factory=list)
    original_question: str = ""  # empty = treat inputs as the root question


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
    session_id = request.session_id
    stop_event = asyncio.Event()
    _stop_events[session_id] = stop_event

    async def generate():
        astream = None
        try:
            llm_url = args.llm_url or os.getenv("LLM_URL", "http://localhost:8080/v1")
            search_engine = os.getenv("SEARCH_ENGINE", args.search_engine)

            agent_graph, tracer = init_agent(
                llm_url=llm_url,
                search_engine=search_engine,
                max_turns=int(os.getenv("MAX_TURNS", "8")),
                max_nodes=int(os.getenv("MAX_NODES", "15")),
                max_concurrent_searchers=int(os.getenv("MAX_CONCURRENT_SEARCHERS", "3")),
                enable_seed_search=os.getenv("ENABLE_SEED_SEARCH", "true").lower() == "true",
                enable_reflection=os.getenv("ENABLE_REFLECTION", "true").lower() == "true",
                enable_compression=os.getenv("ENABLE_COMPRESSION", "false").lower() == "true",
                enable_thinking=os.getenv("ENABLE_THINKING", "true").lower() == "true",
                debug=os.getenv("DEBUG", "false").lower() == "true",
                trace_dir=os.getenv("TRACE_DIR", "./traces"),
            )

            # Multi-turn support: if prior context is provided, extend the graph
            log.info("SOLVE inputs=%r original_question=%r prior_nodes=%d",
                     inputs[:120], request.original_question[:120] if request.original_question else "",
                     len(request.prior_nodes))
            if request.prior_nodes:
                log.info("SOLVE prior_node_ids: %s",
                         ", ".join(f"{n.get('id')}({n.get('status')})" for n in request.prior_nodes))

            if request.original_question:
                combined_question = f"{request.original_question}\n\nFollow-up question: {inputs}"
                log.info("SOLVE mode=followup combined_question=%r", combined_question[:200])
            else:
                combined_question = inputs
                log.info("SOLVE mode=first_question")

            # Seed dispatched_ids from prior_nodes so the dispatcher never
            # re-dispatches nodes already completed in a previous turn.
            prior_ids = [n["id"] for n in request.prior_nodes if n.get("status") == "done"]
            initial_state = {
                "question": combined_question,
                "nodes": list(request.prior_nodes),  # seed with prior completed nodes
                "dispatched_ids": prior_ids,
                "to_dispatch_ids": [],
                "final_answer": "",
                "turns": 0,
                "seed_context": "",
                "reflection_notes": "",
            }
            log.info("SOLVE initial_state.nodes count=%d", len(initial_state["nodes"]))

            # Track all nodes across events for adjacency list building;
            # start with prior nodes so the emitted adjacency list is cumulative.
            all_nodes = list(request.prior_nodes)

            astream = agent_graph.astream(
                initial_state,
                config={"recursion_limit": 200},
                stream_mode="updates",
            )

            # Active heartbeat: yield a real data event every HEARTBEAT_INTERVAL
            # seconds of silence so intermediate proxies / browser idle-tab
            # throttling don't close the stream during long LLM calls.
            # This supplements sse-starlette's ping=15 (comment frames), which
            # was empirically insufficient (connection dropped at ~15s silence).
            HEARTBEAT_INTERVAL = 8

            # Phase tracking — drives the status line shown under the graph.
            # Each event handler updates status_line before yielding; the
            # heartbeat re-emits it so the user always sees what's happening.
            active_ids: set[str] = set()
            status_line: str = "Preparing research plan..."

            def _search_status() -> str:
                if active_ids:
                    ids = ", ".join(sorted(active_ids))
                    return f"Searching: {ids} ({len(active_ids)} active)"
                return "All searches complete — replanning..."

            # Persistent anext task — survives heartbeat timeouts so we don't
            # lose the in-progress graph event.
            anext_task = asyncio.ensure_future(astream.__anext__())

            while True:
                stop_task = asyncio.ensure_future(stop_event.wait())
                done, _ = await asyncio.wait(
                    [anext_task, stop_task],
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=HEARTBEAT_INTERVAL,
                )
                # Drop the stop_task loser; leave anext_task running on timeout.
                if stop_task not in done:
                    stop_task.cancel()
                    try:
                        await stop_task
                    except (asyncio.CancelledError, Exception):
                        pass

                if stop_event.is_set():
                    log.info("SOLVE session %d stopped by user", session_id)
                    anext_task.cancel()
                    try:
                        await anext_task
                    except (asyncio.CancelledError, Exception):
                        pass
                    return

                if not done:
                    # Silence window — re-emit the current phase status line.
                    # Doubles as SSE keepalive and visible progress indicator.
                    hb_evt = _make_legacy_event(thought=status_line)
                    yield {"data": json.dumps(hb_evt, ensure_ascii=False)}
                    continue

                try:
                    event = anext_task.result()
                except StopAsyncIteration:
                    break

                for node_name, update in event.items():
                    log.info("SSE event: node=%s keys=%s", node_name, list(update.keys()) if isinstance(update, dict) else type(update))

                    # LangGraph emits None for routing-only nodes (e.g. reflect
                    # when it skips without writing state). Skip silently.
                    if not isinstance(update, dict):
                        continue

                    if node_name == "seed_searcher":
                        status_line = "Surveying the topic landscape..."
                        seed_ctx = update.get("seed_context", "") or ""
                        evt = _make_legacy_event(
                            thought=(
                                f"{status_line}\n\n"
                                f"{seed_ctx[:500]}"
                            ),
                        )
                        yield {"data": json.dumps(evt, ensure_ascii=False)}

                    elif node_name == "reflect":
                        notes = update.get("reflection_notes", "") or ""
                        status_line = "Reflecting on findings..."
                        if notes:
                            evt = _make_legacy_event(
                                thought=f"Reflection: {notes[:500]}",
                            )
                            yield {"data": json.dumps(evt, ensure_ascii=False)}

                    elif node_name == "planner":
                        new_nodes = update.get("nodes", [])
                        if new_nodes:
                            all_nodes.extend(new_nodes)
                            deduped = _dedup_nodes(all_nodes)
                            adj = _build_adjacency_list(deduped)

                            new_ids = [n["id"] for n in new_nodes]
                            status_line = (
                                f"Planner emitted {len(new_ids)} new node(s): "
                                f"{', '.join(new_ids)} — awaiting dispatch"
                            )

                            # Emit adjacency list update
                            evt = _make_legacy_event(
                                adj_list=adj,
                                thought=status_line,
                            )
                            yield {"data": json.dumps(evt, ensure_ascii=False)}

                            # Emit individual node creation events
                            for n in new_nodes:
                                node_evt = _make_legacy_event(
                                    current_node=n["id"],
                                    node_content=n["query"],
                                    thought=status_line,
                                    action={"parameters": {"query": [n["query"]]}},
                                    adj_list=adj,
                                )
                                yield {"data": json.dumps(node_evt, ensure_ascii=False)}

                    elif node_name == "dispatcher":
                        to_dispatch_ids = update.get("to_dispatch_ids") or []
                        if to_dispatch_ids:
                            active_ids.update(to_dispatch_ids)
                            status_line = (
                                f"Dispatching {len(to_dispatch_ids)} searcher(s): "
                                f"{', '.join(to_dispatch_ids)} ({len(active_ids)} active)"
                            )
                            deduped = _dedup_nodes(all_nodes)
                            adj = _build_adjacency_list(deduped)
                            id_to_query = {n["id"]: n.get("query", "") for n in deduped}
                            for nid in to_dispatch_ids:
                                started_evt = _make_legacy_event(
                                    current_node=nid,
                                    node_content=id_to_query.get(nid, ""),
                                    thought=status_line,
                                    adj_list=adj,
                                )
                                yield {"data": json.dumps(started_evt, ensure_ascii=False)}

                    elif node_name == "searcher":
                        done_nodes = update.get("nodes", [])
                        for n in done_nodes:
                            all_nodes.append(n)
                            active_ids.discard(n["id"])
                            deduped = _dedup_nodes(all_nodes)
                            adj = _build_adjacency_list(deduped)
                            node_evt = _make_legacy_event(
                                current_node=n["id"],
                                node_content=n["query"],
                                thought=n.get("summary", ""),
                                adj_list=adj,
                                node_stream_state=0,
                            )
                            yield {"data": json.dumps(node_evt, ensure_ascii=False)}
                        # After processing all completed searchers in this event,
                        # update the status line so heartbeats show current state.
                        status_line = _search_status()

                    elif node_name == "finalize":
                        status_line = "Synthesizing final answer..."
                        answer = update.get("final_answer", "")
                        deduped = _dedup_nodes(all_nodes)
                        adj = _build_adjacency_list(deduped)

                        evt = _make_legacy_event(thought=answer, adj_list=adj)
                        yield {"data": json.dumps(evt, ensure_ascii=False)}

                        snapshot = {"nodes_snapshot": _dedup_nodes(all_nodes)}
                        yield {"data": json.dumps(snapshot, ensure_ascii=False)}

                        done_evt = _make_legacy_event(
                            thought=answer, adj_list=adj, chat_is_over=True,
                        )
                        yield {"data": json.dumps(done_evt, ensure_ascii=False)}

                        try:
                            trace_path = tracer.save(question=combined_question)
                            if trace_path:
                                yield {"data": json.dumps({"trace_path": trace_path}, ensure_ascii=False)}
                        except Exception:
                            log.exception("Failed to save trace after finalize")

                if await _request.is_disconnected():
                    return

                # Schedule the next graph event for the next iteration.
                anext_task = asyncio.ensure_future(astream.__anext__())

        except asyncio.CancelledError:
            log.info("SOLVE session %d generator cancelled", session_id)
        except Exception as exc:
            exc_type = type(exc).__name__
            exc_detail = str(exc).split("\n")[0][:200]
            msg = f"{exc_type}: {exc_detail}" if exc_detail else exc_type
            logging.exception("Research pipeline error")
            response_json = json.dumps(
                dict(error=dict(msg=msg, details=str(exc))), ensure_ascii=False
            )
            yield {"data": response_json}
            try:
                tracer.save(question=inputs if isinstance(inputs, str) else "")
            except Exception:
                log.exception("Failed to save trace after error")
        finally:
            _stop_events.pop(session_id, None)
            if astream is not None:
                try:
                    await astream.aclose()
                except Exception:
                    pass

    inputs = request.inputs
    # ping=15 sends an SSE comment (": ping\n\n") every 15 seconds to keep
    # the connection alive through proxies and prevent the frontend's
    # fetchEventSource from treating long silent stretches (e.g. during
    # batched searcher ReAct loops) as a disconnection.
    return EventSourceResponse(generate(), ping=15)


app.add_api_route("/solve", run, methods=["POST"])


# ── Stop endpoint ──────────────────────────────────────────────────────────────

class StopRequest(BaseModel):
    session_id: int


@app.post("/stop")
async def stop_run(req: StopRequest):
    event = _stop_events.get(req.session_id)
    if event:
        event.set()
        log.info("STOP session %d signalled", req.session_id)
        return {"stopped": True}
    return {"stopped": False, "reason": "no active run with that session_id"}


# ── History endpoints ──────────────────────────────────────────────────────────

class SaveRequest(BaseModel):
    title: str
    data: Any  # list (legacy) or dict {qaList, allNodes, originalQuestion}


class MakeTitleRequest(BaseModel):
    data: list


@app.get("/history")
async def get_history():
    return history_db.list_researches()


@app.post("/history")
async def post_history(req: SaveRequest):
    rid = history_db.save_research(req.title, req.data)
    return {"id": rid}


@app.get("/history/{rid}")
async def get_research(rid: str):
    rec = history_db.load_research(rid)
    if rec is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Not found")
    return rec


@app.delete("/history/{rid}")
async def delete_research(rid: str):
    history_db.delete_research(rid)
    return {"ok": True}


@app.post("/history/make-title")
async def make_title(req: MakeTitleRequest):
    """Ask the LLM to generate an ≤80-char title from qaList data."""
    try:
        import httpx

        llm_url = args.llm_url or os.getenv("LLM_URL", "http://localhost:8080/v1")

        # Build a compact summary of the research for the prompt
        parts = []
        for item in req.data:
            if item.get("question"):
                parts.append(f"Q: {item['question']}")
            if item.get("response"):
                # Take first 300 chars of response
                parts.append(f"A: {str(item['response'])[:300]}")

        content = "\n".join(parts[:6])  # limit to first 3 Q/A pairs

        payload = {
            "model": "local",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Summarize the following research session into a concise title "
                        f"of at most 80 characters. Reply with only the title, no quotes, "
                        f"no punctuation at the end.\n\n{content}"
                    ),
                }
            ],
            "max_tokens": 60,
            "temperature": 0.3,
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{llm_url}/chat/completions",
                json=payload,
                headers={"Authorization": "Bearer none"},
            )
            resp.raise_for_status()
            result = resp.json()
            title = result["choices"][0]["message"]["content"].strip()
            # Truncate hard at 80 chars
            title = title[:80]

        return {"title": title}
    except Exception as exc:
        log.exception("make-title failed")
        return {"title": "", "error": str(exc)}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
