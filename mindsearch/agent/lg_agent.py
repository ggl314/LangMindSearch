# mindsearch/agent/lg_agent.py

import json
import logging
import operator
import os
from typing import Annotated, TypedDict

log = logging.getLogger("mindsearch.agent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.types import Send

from .lg_prompts import (
    FINAL_SYSTEM,
    FINAL_USER_TEMPLATE,
    PLANNER_SYSTEM,
    PLANNER_USER_TEMPLATE,
    SEARCHER_SYSTEM,
)


# ── Search tool factory ──────────────────────────────────────────────────────

def make_search_tool(engine: str, api_key: str = None, topk: int = 6):
    """Return a LangChain @tool for the chosen search backend."""

    if engine == "DuckDuckGoSearch":
        from duckduckgo_search import DDGS

        @tool
        def search(queries: list[str]) -> str:
            """Search the web. Pass a list of 1-3 query strings."""
            results = []
            with DDGS() as ddgs:
                for q in queries[:3]:
                    try:
                        hits = list(ddgs.text(q, max_results=topk))
                        for h in hits:
                            results.append(
                                f"[{h['title']}]({h['href']})\n{h['body']}"
                            )
                    except Exception as e:
                        results.append(f"Search failed for '{q}': {e}")
            return "\n\n".join(results) if results else "No results found."

        return search

    elif engine == "GoogleSearch":
        import requests

        @tool
        def search(queries: list[str]) -> str:
            """Search the web using Google (via Serper). Pass a list of query strings."""
            results = []
            for q in queries[:3]:
                try:
                    resp = requests.post(
                        "https://google.serper.dev/search",
                        headers={
                            "X-API-KEY": api_key,
                            "Content-Type": "application/json",
                        },
                        json={"q": q, "num": topk},
                        timeout=10,
                    )
                    for item in resp.json().get("organic", []):
                        results.append(
                            f"[{item['title']}]({item['link']})\n"
                            f"{item.get('snippet', '')}"
                        )
                except Exception as e:
                    results.append(f"Search failed for '{q}': {e}")
            return "\n\n".join(results) if results else "No results found."

        return search

    elif engine == "BraveSearch":
        import requests

        @tool
        def search(queries: list[str]) -> str:
            """Search the web using Brave. Pass a list of query strings."""
            results = []
            for q in queries[:3]:
                try:
                    resp = requests.get(
                        "https://api.search.brave.com/res/v1/web/search",
                        headers={
                            "Accept": "application/json",
                            "Accept-Encoding": "gzip",
                            "X-Subscription-Token": api_key,
                        },
                        params={"q": q, "count": topk},
                        timeout=10,
                    )
                    for item in resp.json().get("web", {}).get("results", []):
                        results.append(
                            f"[{item['title']}]({item['url']})\n"
                            f"{item.get('description', '')}"
                        )
                except Exception as e:
                    results.append(f"Search failed for '{q}': {e}")
            return "\n\n".join(results) if results else "No results found."

        return search

    else:
        raise ValueError(f"Unknown search engine: {engine}. "
                         f"Choose: DuckDuckGoSearch, GoogleSearch, BraveSearch")


# ── State definitions ────────────────────────────────────────────────────────

class SearchNode(TypedDict):
    id: str
    query: str
    depends_on: list[str]
    status: str    # "pending" | "done"
    summary: str


class MindSearchState(TypedDict):
    question: str
    # operator.add means updates are appended, not replaced.
    # _dedup_nodes() reconciles duplicates (last-write-wins by id).
    nodes: Annotated[list[SearchNode], operator.add]
    final_answer: str
    turns: int     # planner turn counter, hard stop at MAX_TURNS


class SearcherState(TypedDict):
    node_id: str
    query: str
    question: str
    depends_on: list[str]


MAX_TURNS = 6   # maximum planner iterations before forcing finalize


# ── State helpers ────────────────────────────────────────────────────────────

def _dedup(nodes: list[SearchNode]) -> list[SearchNode]:
    """Last-write-wins deduplication by node id."""
    seen: dict[str, SearchNode] = {}
    for n in nodes:
        seen[n["id"]] = n
    return list(seen.values())


def _completed_ids(nodes: list[SearchNode]) -> set[str]:
    return {n["id"] for n in _dedup(nodes) if n["status"] == "done"}


def _graph_summary(nodes: list[SearchNode]) -> str:
    deduped = _dedup(nodes)
    if not deduped:
        return "No searches conducted yet."
    lines = []
    for n in deduped:
        if n["status"] == "done":
            preview = n["summary"][:200].replace("\n", " ")
            lines.append(f"[{n['id']}] DONE: {n['query']}\n    -> {preview}...")
        else:
            dep_str = f" (waiting for: {n['depends_on']})" if n["depends_on"] else ""
            lines.append(f"[{n['id']}] PENDING: {n['query']}{dep_str}")
    return "\n".join(lines)


# ── Node: planner ────────────────────────────────────────────────────────────

def make_planner_node(llm: ChatOpenAI):
    def planner_node(state: MindSearchState) -> dict:
        turns = state.get("turns", 0)
        existing = state.get("nodes", [])
        log.info("PLANNER turn=%d existing_nodes=%d", turns, len(existing))
        if existing:
            deduped_existing = _dedup(existing)
            log.info("PLANNER existing node ids: %s",
                     ", ".join(f"{n['id']}({n['status']})" for n in deduped_existing))

        # Hard stop — force finalize after MAX_TURNS
        if turns >= MAX_TURNS:
            log.info("PLANNER hit MAX_TURNS=%d, forcing finalize", MAX_TURNS)
            return {"nodes": [], "turns": turns + 1}

        graph_summary = _graph_summary(state["nodes"])
        log.info("PLANNER graph_summary for LLM:\n%s", graph_summary)

        user_msg = PLANNER_USER_TEMPLATE.format(
            question=state["question"],
            graph_summary=graph_summary,
        )

        log.info("PLANNER calling LLM...")
        try:
            response = llm.invoke([
                SystemMessage(content=PLANNER_SYSTEM),
                HumanMessage(content=user_msg),
            ])
        except Exception:
            log.exception("PLANNER LLM call failed")
            raise

        raw = response.content.strip()
        log.info("PLANNER raw LLM response (%d chars): %s", len(raw), raw[:500])

        # Strip markdown fences if model adds them despite instructions
        if "```" in raw:
            parts = raw.split("```")
            # Take the content inside the first fence pair
            raw = parts[1].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("PLANNER JSON parse failed, forcing finalize. Raw: %s", raw[:300])
            # Unparseable — force finalize rather than crash
            return {"nodes": [], "turns": turns + 1}

        log.info("PLANNER parsed action=%s, nodes=%d", parsed.get("action"), len(parsed.get("nodes", [])))

        if parsed.get("action") == "finalize":
            log.info("PLANNER decided to finalize")
            return {"nodes": [], "turns": turns + 1}

        # Build new SearchNode objects
        existing_ids = {n["id"] for n in _dedup(state["nodes"])}
        new_nodes = []
        for raw_node in parsed.get("nodes", []):
            nid = raw_node.get("id", f"n{len(existing_ids)+1}")
            # Guarantee uniqueness
            base = nid
            suffix = 0
            while nid in existing_ids:
                suffix += 1
                nid = f"{base}_{suffix}"
            existing_ids.add(nid)
            new_nodes.append(SearchNode(
                id=nid,
                query=raw_node["query"],
                depends_on=raw_node.get("depends_on", []),
                status="pending",
                summary="",
            ))

        log.info("PLANNER emitting %d new nodes: %s", len(new_nodes), [n["id"] for n in new_nodes])
        return {"nodes": new_nodes, "turns": turns + 1}

    return planner_node


# ── Node: dispatcher ─────────────────────────────────────────────────────────

def _dispatch_sends(state: MindSearchState):
    """Conditional edge: fan out to all unblocked pending nodes via Send()."""
    completed = _completed_ids(state["nodes"])
    to_dispatch = [
        n for n in _dedup(state["nodes"])
        if n["status"] == "pending"
        and all(dep in completed for dep in n["depends_on"])
    ]
    log.info("DISPATCHER dispatching %d nodes (completed=%s): %s",
             len(to_dispatch), completed, [n["id"] for n in to_dispatch])

    return [
        Send("searcher", SearcherState(
            node_id=n["id"],
            query=n["query"],
            question=state["question"],
            depends_on=n["depends_on"],
        ))
        for n in to_dispatch
    ]


# ── Node: searcher ───────────────────────────────────────────────────────────

def make_searcher_node(llm: ChatOpenAI, search_tool):
    llm_with_tools = llm.bind_tools([search_tool])

    def searcher_node(state: SearcherState) -> dict:
        log.info("SEARCHER [%s] starting query: %s", state["node_id"], state["query"])
        messages = [
            SystemMessage(content=SEARCHER_SYSTEM),
            HumanMessage(
                content=f"Search task: {state['query']}\n\n"
                        f"Parent question for context: {state['question']}"
            ),
        ]

        # ReAct loop — max 3 tool call rounds
        final_response = None
        for round_i in range(3):
            log.info("SEARCHER [%s] LLM call round %d...", state["node_id"], round_i + 1)
            try:
                response = llm_with_tools.invoke(messages)
            except Exception:
                log.exception("SEARCHER [%s] LLM call failed round %d", state["node_id"], round_i + 1)
                raise
            messages.append(response)
            final_response = response

            log.info("SEARCHER [%s] round %d: tool_calls=%d, content_len=%d",
                     state["node_id"], round_i + 1,
                     len(response.tool_calls) if response.tool_calls else 0,
                     len(response.content) if response.content else 0)

            if not response.tool_calls:
                break

            for tc in response.tool_calls:
                log.info("SEARCHER [%s] invoking tool %s with args: %s",
                         state["node_id"], tc["name"], str(tc["args"])[:200])
                try:
                    result = search_tool.invoke(tc["args"])
                except Exception:
                    log.exception("SEARCHER [%s] tool invocation failed", state["node_id"])
                    raise
                log.info("SEARCHER [%s] tool result length: %d", state["node_id"], len(str(result)))
                messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tc["id"],
                ))

        summary = (final_response.content or "No results found.") if final_response \
                  else "Search failed."
        log.info("SEARCHER [%s] done, summary length: %d", state["node_id"], len(summary))

        # Return completed node — operator.add appends this to state["nodes"].
        # _dedup() in other nodes will resolve the pending->done transition.
        return {
            "nodes": [SearchNode(
                id=state["node_id"],
                query=state["query"],
                depends_on=state.get("depends_on", []),
                status="done",
                summary=summary,
            )]
        }

    return searcher_node


# ── Node: reflect ────────────────────────────────────────────────────────────

def reflect_node(state: MindSearchState) -> str:
    """Route after searcher batch: dispatcher if more pending, else planner."""
    deduped = _dedup(state["nodes"])
    pending = [n["id"] for n in deduped if n["status"] == "pending"]
    done = [n["id"] for n in deduped if n["status"] == "done"]
    route = "dispatcher" if pending else "planner"
    log.info("REFLECT pending=%s, done=%s -> routing to %s", pending, done, route)
    return route


# ── Routing after planner ────────────────────────────────────────────────────

def route_after_planner(state: MindSearchState) -> str:
    """Route based on whether new pending nodes were added."""
    deduped = _dedup(state["nodes"])
    has_pending = any(n["status"] == "pending" for n in deduped)
    turns = state.get("turns", 0)
    if turns >= MAX_TURNS and not has_pending:
        route = "finalize"
    elif has_pending:
        route = "dispatcher"
    else:
        route = "finalize"
    log.info("ROUTE_AFTER_PLANNER turns=%d, has_pending=%s -> %s", turns, has_pending, route)
    return route


# ── Node: finalize ───────────────────────────────────────────────────────────

def make_finalize_node(llm: ChatOpenAI):
    def finalize_node(state: MindSearchState) -> dict:
        done_nodes = [n for n in _dedup(state["nodes"]) if n["status"] == "done"]
        log.info("FINALIZE synthesizing from %d completed nodes", len(done_nodes))
        findings = "\n\n".join(
            f"Query: {n['query']}\nFindings:\n{n['summary']}"
            for n in done_nodes
        )
        if not findings:
            findings = "No research was completed."

        log.info("FINALIZE calling LLM with %d chars of findings...", len(findings))
        try:
            response = llm.invoke([
                SystemMessage(content=FINAL_SYSTEM),
                HumanMessage(content=FINAL_USER_TEMPLATE.format(
                    question=state["question"],
                    findings=findings,
                )),
            ])
        except Exception:
            log.exception("FINALIZE LLM call failed")
            raise
        log.info("FINALIZE done, answer length: %d", len(response.content) if response.content else 0)
        return {"final_answer": response.content}

    return finalize_node


# ── Graph assembly ───────────────────────────────────────────────────────────

def build_graph(
    llm_url: str,
    search_engine: str,
    api_key: str = None,
    temperature: float = 0.0,
    max_tokens: int = 8192,
):
    llm = ChatOpenAI(
        base_url=llm_url,
        api_key="sk-no-key-required",
        model="local",
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=False,
    )

    search_tool = make_search_tool(engine=search_engine, api_key=api_key)

    graph = StateGraph(MindSearchState)

    graph.add_node("planner",    make_planner_node(llm))
    graph.add_node("dispatcher", lambda s: {})  # no-op; routing is in _dispatch_sends
    graph.add_node("searcher",   make_searcher_node(llm, search_tool))
    graph.add_node("reflect",    lambda s: {})   # pure router, no state change
    graph.add_node("finalize",   make_finalize_node(llm))

    graph.set_entry_point("planner")

    graph.add_conditional_edges("planner", route_after_planner, {
        "dispatcher": "dispatcher",
        "finalize":   "finalize",
    })

    # dispatcher fans out to N parallel searcher nodes via Send()
    graph.add_conditional_edges("dispatcher", _dispatch_sends, ["searcher"])

    graph.add_edge("searcher", "reflect")

    graph.add_conditional_edges("reflect", reflect_node, {
        "dispatcher": "dispatcher",
        "planner":    "planner",
    })

    graph.add_edge("finalize", END)

    return graph.compile()
