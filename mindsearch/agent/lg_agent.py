# mindsearch/agent/lg_agent.py

import json
import logging
import operator
import os
import re
import time
from typing import Annotated, TypedDict

log = logging.getLogger("mindsearch.agent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.types import Send

from mindsearch.tracer import ResearchTracer

from .lg_prompts import (
    COMPRESS_SYSTEM,
    COMPRESS_USER_TEMPLATE,
    FINAL_SYSTEM,
    FINAL_USER_TEMPLATE,
    PLANNER_USER_TEMPLATE,
    REFLECT_SYSTEM,
    REFLECT_USER_TEMPLATE,
    SEARCHER_SYSTEM,
    SEED_SEARCH_SYSTEM,
    make_planner_system,
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
    status: str       # "pending" | "done"
    summary: str      # FINDINGS section only
    leads: str        # LEADS section, stored separately to survive truncation
    category: str     # "core" | "context" | "adjacent" | "emerging" | "critical"
    rationale: str    # one-sentence relevance justification from planner


class MindSearchState(TypedDict):
    question: str
    # operator.add means updates are appended, not replaced.
    # _dedup() reconciles duplicates (last-write-wins by id).
    nodes: Annotated[list[SearchNode], operator.add]
    # Accumulates IDs of all nodes ever dispatched to searchers.
    # Prevents reflect→dispatcher from re-dispatching already-running nodes.
    dispatched_ids: Annotated[list[str], operator.add]
    # Set by dispatcher_node each cycle; read by _dispatch_sends to fan out.
    to_dispatch_ids: list[str]
    final_answer: str
    turns: int                 # planner turn counter, hard stop at MAX_TURNS
    seed_context: str          # landscape survey from seed_searcher
    reflection_notes: str      # latest reflection fed back to planner


class SearcherState(TypedDict):
    node_id: str
    query: str
    question: str
    depends_on: list[str]
    category: str              # forwarded from SearchNode through dispatcher
    rationale: str             # forwarded from SearchNode through dispatcher


# Module-level configuration set by build_graph() — used by planner for
# budget enforcement and _graph_summary() for reporting.
MAX_TURNS = 8
MAX_NODES = 15
# Cap on how many searchers the dispatcher will fan out per cycle. Limits
# concurrent load on a single llama-server; remaining pending nodes are
# picked up by the next reflect→dispatcher cycle.
MAX_CONCURRENT_SEARCHERS = 3


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

    # Budget and category summary
    total = len(deduped)
    categories: dict[str, int] = {}
    for n in deduped:
        cat = n.get("category", "core")
        categories[cat] = categories.get(cat, 0) + 1
    lines.append(f"Total nodes: {total}/{MAX_NODES} (budget)")
    lines.append(f"Category coverage: {dict(categories)}")
    lines.append("")

    for n in deduped:
        cat = n.get("category", "core")
        if n["status"] == "done":
            # Findings get truncated for space; leads preserved separately
            preview = (n.get("summary", "") or "")[:300].replace("\n", " ")
            lines.append(f"[{n['id']}] ({cat}) DONE: {n['query']}")
            lines.append(f"    Findings: {preview}...")
            if n.get("leads"):
                lines.append(f"    Leads: {n['leads'][:400]}")
        else:
            dep_str = f" (waiting for: {n['depends_on']})" if n["depends_on"] else ""
            lines.append(f"[{n['id']}] ({cat}) PENDING: {n['query']}{dep_str}")
    return "\n".join(lines)


def _parse_searcher_output(raw: str) -> tuple[str, str]:
    """Split searcher output into (findings, leads).

    Expects the format:
        FINDINGS:
        ...text...

        LEADS:
        ...text...

    If the format isn't followed, treats entire output as findings.
    """
    if not raw:
        return "", ""

    raw_upper = raw.upper()
    leads_idx = raw_upper.rfind("LEADS")
    if leads_idx == -1:
        return raw.strip(), ""

    # Walk back to line start
    line_start = raw.rfind("\n", 0, leads_idx)
    line_start = 0 if line_start == -1 else line_start + 1

    findings = raw[:line_start].strip()
    leads = raw[leads_idx:].strip()

    # Strip section headers
    for prefix in ("FINDINGS:", "FINDINGS"):
        if findings.upper().startswith(prefix):
            findings = findings[len(prefix):].strip()
            break
    for prefix in ("LEADS:", "LEADS"):
        if leads.upper().startswith(prefix):
            leads = leads[len(prefix):].strip()
            break

    return findings, leads


def _category_counts(nodes: list[SearchNode]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for n in _dedup(nodes):
        cat = n.get("category", "core")
        counts[cat] = counts.get(cat, 0) + 1
    return counts


def _strip_thinking(text: str) -> str:
    """Remove thinking blocks from model output.

    Qwen3.5 wraps reasoning in <think>...</think>.
    Gemma 4 uses <start_of_thought>...<end_of_thought>.
    """
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'<start_of_thought>.*?<end_of_thought>', '', text, flags=re.DOTALL).strip()
    return text


# ── Node: seed searcher ──────────────────────────────────────────────────────

def make_seed_searcher_node(llm: ChatOpenAI, search_tool, tracer: ResearchTracer):
    """Run 2-3 broad searches before planning to ground the planner in reality."""
    llm_with_tools = llm.bind_tools([search_tool])

    def seed_searcher_node(state: MindSearchState) -> dict:
        ctx = tracer.node_start("seed_searcher", tracer.snapshot_state(state))
        question = state["question"]
        log.info("SEED_SEARCHER starting landscape survey for: %s", question[:120])
        messages = [
            SystemMessage(content=SEED_SEARCH_SYSTEM),
            HumanMessage(content=f"Topic to survey: {question}"),
        ]

        seed_context = ""
        try:
            t0 = time.time()
            response = llm_with_tools.invoke(messages)
            tracer.llm_call(
                caller="seed_searcher",
                system_prompt_len=len(SEED_SEARCH_SYSTEM),
                user_prompt=f"Topic to survey: {question}",
                raw_response=response.content or "",
                duration_s=time.time() - t0,
            )
            messages.append(response)

            if response.tool_calls:
                for tc in response.tool_calls:
                    t1 = time.time()
                    try:
                        result = search_tool.invoke(tc["args"])
                        tracer.tool_call(
                            caller="seed_searcher",
                            tool_name=tc.get("name", "search"),
                            args=tc.get("args", {}),
                            result_len=len(str(result)),
                            duration_s=time.time() - t1,
                        )
                    except Exception as e:
                        tracer.tool_call(
                            caller="seed_searcher",
                            tool_name=tc.get("name", "search"),
                            args=tc.get("args", {}),
                            result_len=0,
                            duration_s=time.time() - t1,
                            error=str(e),
                        )
                        raise
                    messages.append(ToolMessage(
                        content=str(result), tool_call_id=tc["id"]
                    ))
                # Get the synthesis
                t2 = time.time()
                summary_response = llm_with_tools.invoke(messages)
                tracer.llm_call(
                    caller="seed_searcher",
                    system_prompt_len=len(SEED_SEARCH_SYSTEM),
                    user_prompt="<post-tool synthesis>",
                    raw_response=summary_response.content or "",
                    duration_s=time.time() - t2,
                )
                seed_context = summary_response.content or ""
            else:
                seed_context = response.content or ""
        except Exception as e:
            log.exception("SEED_SEARCHER failed, proceeding without seed context")
            tracer.llm_call(
                caller="seed_searcher",
                system_prompt_len=len(SEED_SEARCH_SYSTEM),
                user_prompt=f"Topic to survey: {question}",
                raw_response="",
                duration_s=0,
                error=str(e),
            )
            seed_context = ""

        log.info("SEED_SEARCHER done, context length: %d", len(seed_context))
        output = {"seed_context": seed_context}
        tracer.node_end("seed_searcher", ctx, output,
                        notes=f"seed_context={len(seed_context)}ch")
        return output

    return seed_searcher_node


# ── Node: planner ────────────────────────────────────────────────────────────

def make_planner_node(llm: ChatOpenAI, planner_system: str, tracer: ResearchTracer):
    def planner_node(state: MindSearchState) -> dict:
        ctx = tracer.node_start("planner", tracer.snapshot_state(state))
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
            output = {"nodes": [], "turns": turns + 1}
            tracer.budget_check(
                total_nodes=len(_dedup(existing)),
                max_nodes=MAX_NODES,
                turns=turns,
                max_turns=MAX_TURNS,
                categories=_category_counts(existing),
            )
            tracer.node_end("planner", ctx, output,
                            notes=f"forced finalize (MAX_TURNS={MAX_TURNS})")
            return output

        graph_summary = _graph_summary(state["nodes"])
        log.info("PLANNER graph_summary for LLM:\n%s", graph_summary)

        user_msg = PLANNER_USER_TEMPLATE.format(
            question=state["question"],
            seed_context=state.get("seed_context") or "No landscape survey available.",
            reflection_notes=state.get("reflection_notes") or
                             "No reflection yet (first planning turn).",
            graph_summary=graph_summary,
        )

        log.info("PLANNER calling LLM...")
        try:
            t0 = time.time()
            response = llm.invoke([
                SystemMessage(content=planner_system),
                HumanMessage(content=user_msg),
            ])
            llm_duration = time.time() - t0
        except Exception as e:
            log.exception("PLANNER LLM call failed")
            tracer.llm_call(
                caller="planner",
                system_prompt_len=len(planner_system),
                user_prompt=user_msg,
                raw_response="",
                duration_s=0,
                error=str(e),
            )
            tracer.node_end("planner", ctx, {}, notes="LLM call failed")
            raise

        raw = (response.content or "").strip()
        log.info("PLANNER raw LLM response (%d chars): %s", len(raw), raw[:500])

        # Strip markdown fences if model adds them despite instructions
        if "```" in raw:
            parts = raw.split("```")
            if len(parts) > 1:
                raw = parts[1].strip()
                if raw.startswith("json"):
                    raw = raw[4:].strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("PLANNER JSON parse failed, forcing finalize. Raw: %s", raw[:300])
            tracer.llm_call(
                caller="planner",
                system_prompt_len=len(planner_system),
                user_prompt=user_msg,
                raw_response=response.content or "",
                duration_s=llm_duration,
                error="JSONDecodeError",
            )
            output = {"nodes": [], "turns": turns + 1}
            tracer.node_end("planner", ctx, output,
                            notes="forced finalize (parse error)")
            return output

        tracer.llm_call(
            caller="planner",
            system_prompt_len=len(planner_system),
            user_prompt=user_msg,
            raw_response=response.content or "",
            parsed=parsed,
            duration_s=llm_duration,
        )

        log.info("PLANNER parsed action=%s, nodes=%d",
                 parsed.get("action"), len(parsed.get("nodes", [])))

        if parsed.get("action") == "finalize":
            log.info("PLANNER decided to finalize")
            output = {"nodes": [], "turns": turns + 1}
            tracer.budget_check(
                total_nodes=len(_dedup(existing)),
                max_nodes=MAX_NODES,
                turns=turns,
                max_turns=MAX_TURNS,
                categories=_category_counts(existing),
            )
            tracer.node_end("planner", ctx, output, notes="planner chose finalize")
            return output

        # Build new SearchNode objects
        existing_ids = {n["id"] for n in _dedup(state["nodes"])}
        new_nodes: list[SearchNode] = []
        for raw_node in parsed.get("nodes", []):
            nid = raw_node.get("id", f"n{len(existing_ids) + len(new_nodes) + 1}")
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
                leads="",
                category=raw_node.get("category", "core"),
                rationale=raw_node.get("rationale", ""),
            ))

        # Enforce exploration budget
        existing_count = len(_dedup(state["nodes"]))
        remaining_budget = MAX_NODES - existing_count
        if remaining_budget <= 0:
            log.info("PLANNER hit node budget (%d), forcing finalize", MAX_NODES)
            output = {"nodes": [], "turns": turns + 1}
            tracer.budget_check(
                total_nodes=existing_count,
                max_nodes=MAX_NODES,
                turns=turns,
                max_turns=MAX_TURNS,
                categories=_category_counts(state["nodes"]),
            )
            tracer.node_end("planner", ctx, output,
                            notes=f"forced finalize (budget={MAX_NODES})")
            return output

        trimmed = False
        if len(new_nodes) > remaining_budget:
            new_nodes = new_nodes[:remaining_budget]
            trimmed = True

        log.info("PLANNER emitting %d new nodes: %s",
                 len(new_nodes), [n["id"] for n in new_nodes])

        projected_total = existing_count + len(new_nodes)
        projected_categories = dict(_category_counts(state["nodes"]))
        for n in new_nodes:
            c = n.get("category", "core")
            projected_categories[c] = projected_categories.get(c, 0) + 1
        tracer.budget_check(
            total_nodes=projected_total,
            max_nodes=MAX_NODES,
            turns=turns + 1,
            max_turns=MAX_TURNS,
            categories=projected_categories,
        )

        output = {"nodes": new_nodes, "turns": turns + 1}
        notes = f"emitted {len(new_nodes)} nodes"
        if trimmed:
            notes += " (trimmed to budget)"
        tracer.node_end("planner", ctx, output, notes=notes)
        return output

    return planner_node


# ── Node: dispatcher ─────────────────────────────────────────────────────────

def make_dispatcher_node(tracer: ResearchTracer):
    def dispatcher_node(state: MindSearchState) -> dict:
        """Select unblocked pending nodes that haven't been dispatched yet.

        Caps the batch at MAX_CONCURRENT_SEARCHERS so a single llama-server
        isn't hammered by 6+ parallel ReAct loops at once. Remaining pending
        nodes are picked up by the next reflect→dispatcher cycle after the
        current batch completes.
        """
        ctx = tracer.node_start("dispatcher", tracer.snapshot_state(state))
        completed = _completed_ids(state["nodes"])
        already_dispatched = set(state.get("dispatched_ids", []))
        eligible = [
            n for n in _dedup(state["nodes"])
            if n["status"] == "pending"
            and all(dep in completed for dep in n["depends_on"])
            and n["id"] not in already_dispatched
        ]
        to_dispatch = eligible[:MAX_CONCURRENT_SEARCHERS]
        deferred = eligible[MAX_CONCURRENT_SEARCHERS:]
        new_ids = [n["id"] for n in to_dispatch]
        if deferred:
            log.info("DISPATCHER dispatching %d/%d nodes (cap=%d, deferred=%s): %s",
                     len(to_dispatch), len(eligible), MAX_CONCURRENT_SEARCHERS,
                     [n["id"] for n in deferred], new_ids)
        else:
            log.info("DISPATCHER dispatching %d nodes (completed=%s, already_dispatched=%s): %s",
                     len(to_dispatch), completed, already_dispatched, new_ids)
        output = {"dispatched_ids": new_ids, "to_dispatch_ids": new_ids}
        tracer.node_end("dispatcher", ctx, output,
                        notes=f"dispatching={new_ids} cap={MAX_CONCURRENT_SEARCHERS}")
        return output

    return dispatcher_node


# ── Node: searcher ───────────────────────────────────────────────────────────

def make_searcher_node(fast_llm: ChatOpenAI, reason_llm: ChatOpenAI,
                       search_tool, tracer: ResearchTracer):
    fast_with_tools = fast_llm.bind_tools([search_tool])
    reason_with_tools = reason_llm.bind_tools([search_tool])

    def searcher_node(state: SearcherState) -> dict:
        ctx = tracer.node_start("searcher", {
            "node_id": state["node_id"],
            "query": (state["query"] or "")[:150],
            "category": state.get("category", "core"),
        })
        log.info("SEARCHER [%s] starting query: %s", state["node_id"], state["query"])
        messages = [
            SystemMessage(content=SEARCHER_SYSTEM),
            HumanMessage(
                content=f"Search task: {state['query']}\n\n"
                        f"Parent question for context: {state['question']}"
            ),
        ]

        # ReAct loop — max 3 tool call rounds.
        # Use fast LLM for tool-calling rounds, reasoning LLM for the final
        # synthesis so it can think through the evidence.
        final_response = None
        for round_i in range(3):
            is_likely_final = (round_i == 2)  # last allowed round
            llm = reason_with_tools if is_likely_final else fast_with_tools
            log.info("SEARCHER [%s] LLM call round %d (%s)...",
                     state["node_id"], round_i + 1,
                     "reason" if is_likely_final else "fast")
            try:
                t0 = time.time()
                response = llm.invoke(messages)
                tracer.llm_call(
                    caller=f"searcher[{state['node_id']}]r{round_i+1}",
                    system_prompt_len=len(SEARCHER_SYSTEM),
                    user_prompt=f"Search task: {state['query']}",
                    raw_response=response.content or "",
                    duration_s=time.time() - t0,
                )
            except Exception as e:
                log.exception("SEARCHER [%s] LLM call failed round %d",
                              state["node_id"], round_i + 1)
                tracer.llm_call(
                    caller=f"searcher[{state['node_id']}]r{round_i+1}",
                    system_prompt_len=len(SEARCHER_SYSTEM),
                    user_prompt=f"Search task: {state['query']}",
                    raw_response="",
                    duration_s=0,
                    error=str(e),
                )
                tracer.node_end("searcher", ctx, {},
                                notes=f"[{state['node_id']}] LLM failed")
                raise
            messages.append(response)
            final_response = response

            log.info("SEARCHER [%s] round %d: tool_calls=%d, content_len=%d",
                     state["node_id"], round_i + 1,
                     len(response.tool_calls) if response.tool_calls else 0,
                     len(response.content) if response.content else 0)

            if not response.tool_calls:
                if round_i == 0:
                    log.warning(
                        "SEARCHER [%s] round 1: model made NO tool calls — skipped search. "
                        "content preview: %s",
                        state["node_id"],
                        (response.content or "")[:500],
                    )
                break

            for tc in response.tool_calls:
                log.info("SEARCHER [%s] invoking tool %s with args: %s",
                         state["node_id"], tc["name"], str(tc["args"])[:200])
                try:
                    t1 = time.time()
                    result = search_tool.invoke(tc["args"])
                    tracer.tool_call(
                        caller=f"searcher[{state['node_id']}]r{round_i+1}",
                        tool_name=tc.get("name", "search"),
                        args=tc.get("args", {}),
                        result_len=len(str(result)),
                        duration_s=time.time() - t1,
                    )
                except Exception as e:
                    log.exception("SEARCHER [%s] tool invocation failed",
                                  state["node_id"])
                    tracer.tool_call(
                        caller=f"searcher[{state['node_id']}]r{round_i+1}",
                        tool_name=tc.get("name", "search"),
                        args=tc.get("args", {}),
                        result_len=0,
                        duration_s=0,
                        error=str(e),
                    )
                    raise
                result_str = str(result)
                log.info("SEARCHER [%s] tool result length: %d  preview: %s",
                         state["node_id"], len(result_str), result_str[:400])
                messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tc["id"],
                ))

        # If the last round still had tool calls, do one more reasoning call
        # for final synthesis after those tool results were appended.
        if final_response and final_response.tool_calls:
            log.info("SEARCHER [%s] extra reasoning call for synthesis...",
                     state["node_id"])
            t0 = time.time()
            final_response = reason_with_tools.invoke(messages)
            tracer.llm_call(
                caller=f"searcher[{state['node_id']}]synth",
                system_prompt_len=len(SEARCHER_SYSTEM),
                user_prompt=f"Search task: {state['query']}",
                raw_response=final_response.content or "",
                duration_s=time.time() - t0,
            )
            messages.append(final_response)

        raw_output = (final_response.content or "No results found.") if final_response \
                     else "Search failed."
        log.info("SEARCHER [%s] raw_output before strip (%d chars): %s",
                 state["node_id"], len(raw_output), raw_output[:600])
        raw_output = _strip_thinking(raw_output)
        if raw_output != (final_response.content or "No results found." if final_response else "Search failed."):
            log.info("SEARCHER [%s] after strip (%d chars): %s",
                     state["node_id"], len(raw_output), raw_output[:400])
        findings, leads = _parse_searcher_output(raw_output)
        log.info("SEARCHER [%s] parsed: findings=%d chars, leads=%d chars",
                 state["node_id"], len(findings), len(leads))

        output = {
            "nodes": [SearchNode(
                id=state["node_id"],
                query=state["query"],
                depends_on=state.get("depends_on", []),
                status="done",
                summary=findings,
                leads=leads,
                category=state.get("category", "core"),
                rationale=state.get("rationale", ""),
            )]
        }
        tracer.node_end(
            "searcher", ctx, output,
            notes=f"[{state['node_id']}] findings={len(findings)}ch leads={len(leads)}ch",
        )
        return output

    return searcher_node


# ── Node: reflect ────────────────────────────────────────────────────────────

def make_reflect_node(llm: ChatOpenAI, tracer: ResearchTracer):
    """LLM-powered reflection after each searcher batch.

    Writes reflection_notes into state for the planner to read next turn.
    """
    def reflect_node(state: MindSearchState) -> dict:
        ctx = tracer.node_start("reflect", tracer.snapshot_state(state))
        deduped = _dedup(state["nodes"])
        pending = [n for n in deduped if n["status"] == "pending"]
        done = [n for n in deduped if n["status"] == "done"]

        # If there are unblocked pending nodes, skip reflection — just route
        if pending:
            completed_ids = {n["id"] for n in done}
            unblocked = [n for n in pending
                         if all(d in completed_ids for d in n["depends_on"])]
            if unblocked:
                tracer.node_end("reflect", ctx, {},
                                notes=f"skipped (unblocked pending={len(unblocked)})")
                return {}  # dispatcher will pick them up

        # Only reflect if we have enough completed results to assess
        if len(done) < 3:
            tracer.node_end("reflect", ctx, {},
                            notes=f"skipped (done={len(done)} < 3)")
            return {}

        findings_summary = "\n".join(
            f"- [{n['id']}] ({n.get('category','core')}) {n['query']}: "
            f"{(n.get('summary','') or '')[:150]}"
            for n in done
        )

        categories_hit = set(n.get("category", "core") for n in done)
        all_categories = {"core", "context", "adjacent", "emerging", "critical"}
        total = len(deduped)

        user_msg = REFLECT_USER_TEMPLATE.format(
            question=state["question"],
            findings=findings_summary,
            categories_covered=", ".join(sorted(categories_hit)),
            categories_missing=", ".join(sorted(all_categories - categories_hit)),
            total_nodes=total,
            budget_remaining=MAX_NODES - total,
        )

        notes_text = ""
        try:
            t0 = time.time()
            response = llm.invoke([
                SystemMessage(content=REFLECT_SYSTEM),
                HumanMessage(content=user_msg),
            ])
            tracer.llm_call(
                caller="reflect",
                system_prompt_len=len(REFLECT_SYSTEM),
                user_prompt=user_msg,
                raw_response=response.content or "",
                duration_s=time.time() - t0,
            )
            notes_text = _strip_thinking(response.content or "")
        except Exception as e:
            log.exception("REFLECT LLM call failed, proceeding without notes")
            tracer.llm_call(
                caller="reflect",
                system_prompt_len=len(REFLECT_SYSTEM),
                user_prompt=user_msg,
                raw_response="",
                duration_s=0,
                error=str(e),
            )
            notes_text = ""

        log.info("REFLECT notes: %s", notes_text[:300])
        output = {"reflection_notes": notes_text}
        tracer.node_end("reflect", ctx, output,
                        notes=f"notes={len(notes_text)}ch: {notes_text[:100]}")
        return output

    return reflect_node


# ── Node: finalize ───────────────────────────────────────────────────────────

def make_finalize_node(llm: ChatOpenAI, enable_compression: bool,
                       tracer: ResearchTracer):
    COMPRESS_THRESHOLD = 6000  # chars, only checked if compression enabled

    def finalize_node(state: MindSearchState) -> dict:
        ctx = tracer.node_start("finalize", tracer.snapshot_state(state))
        done_nodes = [n for n in _dedup(state["nodes"]) if n["status"] == "done"]
        log.info("FINALIZE synthesizing from %d completed nodes", len(done_nodes))

        # Group by category
        by_category: dict[str, list[str]] = {
            "core": [], "context": [], "adjacent": [],
            "emerging": [], "critical": []
        }
        for n in done_nodes:
            cat = n.get("category", "core")
            if cat not in by_category:
                cat = "core"
            by_category[cat].append(
                f"Query: {n['query']}\nFindings:\n{n.get('summary','')}"
            )

        def _join(items: list[str]) -> str:
            return "\n\n---\n\n".join(items) if items else "(no findings)"

        all_text = {cat: _join(items) for cat, items in by_category.items()}

        total_chars = sum(len(v) for v in all_text.values())
        compressed = False

        # Optional compression for small-context models
        if enable_compression:
            if total_chars > COMPRESS_THRESHOLD:
                log.info("FINALIZE compressing: %d chars > %d threshold",
                         total_chars, COMPRESS_THRESHOLD)
                compressed = True
                for cat, text in list(all_text.items()):
                    if text == "(no findings)" or len(text) < 800:
                        continue
                    try:
                        t0 = time.time()
                        result = llm.invoke([
                            SystemMessage(content=COMPRESS_SYSTEM),
                            HumanMessage(content=COMPRESS_USER_TEMPLATE.format(
                                category=cat,
                                question=state["question"],
                                findings=text,
                                max_words=300,
                            )),
                        ])
                        tracer.llm_call(
                            caller=f"finalize.compress[{cat}]",
                            system_prompt_len=len(COMPRESS_SYSTEM),
                            user_prompt=f"[compress category={cat}, {len(text)}ch]",
                            raw_response=result.content or "",
                            duration_s=time.time() - t0,
                        )
                        all_text[cat] = result.content or text[:2000]
                    except Exception as e:
                        log.exception("FINALIZE compression failed for '%s'", cat)
                        tracer.llm_call(
                            caller=f"finalize.compress[{cat}]",
                            system_prompt_len=len(COMPRESS_SYSTEM),
                            user_prompt=f"[compress category={cat}, {len(text)}ch]",
                            raw_response="",
                            duration_s=0,
                            error=str(e),
                        )
                        all_text[cat] = text[:2000]

        user_msg = FINAL_USER_TEMPLATE.format(
            question=state["question"],
            core_findings=all_text["core"],
            context_findings=all_text["context"],
            adjacent_findings=all_text["adjacent"],
            emerging_findings=all_text["emerging"],
            critical_findings=all_text["critical"],
        )

        log.info("FINALIZE calling synthesis LLM with %d chars...", len(user_msg))
        try:
            t0 = time.time()
            response = llm.invoke([
                SystemMessage(content=FINAL_SYSTEM),
                HumanMessage(content=user_msg),
            ])
            tracer.llm_call(
                caller="finalize",
                system_prompt_len=len(FINAL_SYSTEM),
                user_prompt=user_msg,
                raw_response=response.content or "",
                duration_s=time.time() - t0,
            )
        except Exception as e:
            log.exception("FINALIZE LLM call failed")
            tracer.llm_call(
                caller="finalize",
                system_prompt_len=len(FINAL_SYSTEM),
                user_prompt=user_msg,
                raw_response="",
                duration_s=0,
                error=str(e),
            )
            tracer.node_end("finalize", ctx, {}, notes="LLM failed")
            raise

        answer = _strip_thinking(response.content or "")
        answer_len = len(answer)
        log.info("FINALIZE done, answer length: %d", answer_len)
        output = {"final_answer": answer}
        notes = f"total_findings={total_chars}ch, answer={answer_len}ch"
        if compressed:
            notes += ", compressed"
        tracer.node_end("finalize", ctx, output, notes=notes)
        return output

    return finalize_node


# ── Graph assembly ───────────────────────────────────────────────────────────

def build_graph(
    llm_url: str,
    search_engine: str,
    api_key: str = None,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    max_turns: int = 8,
    max_nodes: int = 15,
    max_concurrent_searchers: int = 3,
    enable_seed_search: bool = True,
    enable_reflection: bool = True,
    enable_compression: bool = False,
    enable_thinking: bool = True,
    debug: bool = False,
    trace_dir: str = "./traces",
):
    global MAX_TURNS, MAX_NODES, MAX_CONCURRENT_SEARCHERS
    MAX_TURNS = max_turns
    MAX_NODES = max_nodes
    MAX_CONCURRENT_SEARCHERS = max_concurrent_searchers

    tracer = ResearchTracer(enabled=debug, output_dir=trace_dir)

    # Fast LLM — thinking disabled, used for planner, seed searcher, searcher
    # tool-call rounds. Structured output where speed matters.
    fast_llm = ChatOpenAI(
        base_url=llm_url,
        api_key="sk-no-key-required",
        model="local",
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=False,
        max_retries=6,
        extra_body={
            "cache_prompt": True,
            **({"chat_template_kwargs": {"enable_thinking": False}} if enable_thinking else {}),
        },
    )

    # Reasoning LLM — thinking enabled, used for reflect, finalize, searcher
    # final synthesis. Quality matters more than speed.
    reason_llm = ChatOpenAI(
        base_url=llm_url,
        api_key="sk-no-key-required",
        model="local",
        temperature=temperature,
        max_tokens=max_tokens * 2,  # thinking tokens need more headroom
        streaming=False,
        max_retries=6,
        extra_body={
            "cache_prompt": True,
            **({"chat_template_kwargs": {"enable_thinking": True}} if enable_thinking else {}),
        },
    )

    search_tool = make_search_tool(engine=search_engine, api_key=api_key)
    planner_system = make_planner_system(max_nodes=max_nodes)

    graph = StateGraph(MindSearchState)

    graph.add_node("planner",    make_planner_node(fast_llm, planner_system, tracer))
    graph.add_node("dispatcher", make_dispatcher_node(tracer))
    graph.add_node("searcher",   make_searcher_node(fast_llm, reason_llm, search_tool, tracer))
    graph.add_node("finalize",   make_finalize_node(reason_llm, enable_compression, tracer))

    if enable_seed_search:
        graph.add_node("seed_searcher",
                       make_seed_searcher_node(fast_llm, search_tool, tracer))
        graph.set_entry_point("seed_searcher")
        graph.add_edge("seed_searcher", "planner")
    else:
        graph.set_entry_point("planner")

    if enable_reflection:
        graph.add_node("reflect", make_reflect_node(reason_llm, tracer))
    else:
        graph.add_node("reflect", lambda s: {})

    # ── Routing closures capture tracer from enclosing scope ────────────

    def _route_after_planner(state: MindSearchState) -> str:
        deduped = _dedup(state["nodes"])
        has_pending = any(n["status"] == "pending" for n in deduped)
        turns = state.get("turns", 0)
        if turns >= MAX_TURNS and not has_pending:
            route = "finalize"
        elif has_pending:
            route = "dispatcher"
        else:
            route = "finalize"
        log.info("ROUTE_AFTER_PLANNER turns=%d, has_pending=%s -> %s",
                 turns, has_pending, route)
        tracer.route_decision(
            "route_after_planner", route,
            reason=f"turns={turns}, has_pending={has_pending}",
        )
        return route

    def _reflect_router(state: MindSearchState) -> str:
        """Route after reflection: dispatcher if unblocked pending, else planner."""
        deduped = _dedup(state["nodes"])
        completed_ids = {n["id"] for n in deduped if n["status"] == "done"}
        pending_unblocked = [
            n for n in deduped
            if n["status"] == "pending"
            and all(d in completed_ids for d in n["depends_on"])
        ]
        route = "dispatcher" if pending_unblocked else "planner"
        log.info("REFLECT_ROUTER pending_unblocked=%d -> %s",
                 len(pending_unblocked), route)
        tracer.route_decision(
            "reflect_router", route,
            reason=f"pending_unblocked={len(pending_unblocked)}",
        )
        return route

    def _dispatch_sends_wrapped(state: MindSearchState):
        """Conditional edge: fan out to nodes selected by dispatcher via Send().

        Falls back to 'planner' if dispatcher found nothing new to dispatch.
        """
        to_dispatch_ids = set(state.get("to_dispatch_ids", []))
        if not to_dispatch_ids:
            log.info("DISPATCHER: nothing new to dispatch, routing to planner")
            tracer.route_decision(
                "dispatch_sends", "planner",
                reason="nothing to dispatch",
            )
            return "planner"
        to_dispatch = [
            n for n in _dedup(state["nodes"])
            if n["id"] in to_dispatch_ids
        ]
        sends = [
            Send("searcher", SearcherState(
                node_id=n["id"],
                query=n["query"],
                question=state["question"],
                depends_on=n["depends_on"],
                category=n.get("category", "core"),
                rationale=n.get("rationale", ""),
            ))
            for n in to_dispatch
        ]
        ids = [n["id"] for n in to_dispatch]
        tracer.route_decision(
            "dispatch_sends", f"fan_out({ids})",
            reason=f"count={len(ids)}",
        )
        return sends

    graph.add_conditional_edges("planner", _route_after_planner, {
        "dispatcher": "dispatcher",
        "finalize":   "finalize",
    })

    # dispatcher -> _dispatch_sends either returns a list of Send() (fan out to
    # searcher) or the string "planner" for fallback.
    graph.add_conditional_edges("dispatcher", _dispatch_sends_wrapped,
                                ["searcher", "planner"])

    graph.add_edge("searcher", "reflect")

    graph.add_conditional_edges("reflect", _reflect_router, {
        "dispatcher": "dispatcher",
        "planner":    "planner",
    })

    graph.add_edge("finalize", END)

    return graph.compile(), tracer
