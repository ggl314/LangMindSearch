# mindsearch/agent/lg_plan.py
#
# Outer (plan-level) graph for MindSearch explicit-planning mode.
#
# Flow:
#   seed_searcher  → plan_generator  → (user confirms / amends plan)
#                                      ↓
#   stage_s1 ──┐
#   stage_s2 ──┼──►  narrative_synthesis → END
#   stage_s3 ──┘    (may depend on earlier stages via depends_on)
#
# The inner per-stage research reuses the existing MindSearch graph
# (planner → dispatcher → searcher → reflect → finalize).

import copy
import json
import logging
import re
import time
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from mindsearch.tracer import ResearchTracer
from .lg_prompts import (
    NARRATIVE_SYNTHESIS_SYSTEM,
    NARRATIVE_SYNTHESIS_USER_TEMPLATE,
    NEEDS_PLAN_CHECK,
    PLAN_AMENDER_SYSTEM,
    PLAN_AMENDER_USER_TEMPLATE,
    PLAN_CONFIRM_CHECK,
    PLAN_GENERATOR_SYSTEM,
    PLAN_GENERATOR_USER_TEMPLATE,
    STAGE_FINALIZE_SYSTEM,
    STAGE_QUERY_TEMPLATE,
)

log = logging.getLogger("mindsearch.plan")


# ── Types ────────────────────────────────────────────────────────────────────

PlanStage = dict   # {id, title, description, search_hints, depends_on,
                   #  status, summary}

# Public shape of a plan as stored / serialised:
# {
#   "stages": [
#     {"id": "s1", "title": "...", "description": "...",
#      "search_hints": [...], "depends_on": [...],
#      "status": "pending" | "active" | "done",
#      "summary": ""},
#     ...
#   ]
# }


# ── JSON extraction helper ───────────────────────────────────────────────────

def _extract_json(raw: str) -> Optional[dict]:
    """Parse the first JSON object from raw LLM output.

    Tolerant of markdown fences and leading/trailing prose.
    Returns None on failure.
    """
    if not raw:
        return None

    text = raw.strip()

    # Strip <think>...</think> blocks if model emits them
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'<start_of_thought>.*?<end_of_thought>', '', text,
                  flags=re.DOTALL).strip()

    # Strip markdown fences
    if "```" in text:
        parts = text.split("```")
        # pick the fenced block that looks like JSON
        for p in parts:
            s = p.strip()
            if s.startswith("json"):
                s = s[4:].strip()
            if s.startswith("{") and s.endswith("}"):
                text = s
                break

    # Find the first balanced {...} if prose leaks in
    start = text.find("{")
    if start == -1:
        return None
    # naive brace balance — good enough for well-formed planner output
    depth = 0
    end = -1
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        return None

    candidate = text[start:end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        log.warning("_extract_json: failed to parse: %s", candidate[:300])
        return None


# ── Plan validation / normalisation ──────────────────────────────────────────

def _normalise_plan(raw_plan: dict) -> dict:
    """Coerce a raw plan dict into the canonical shape.

    - Ensures every stage has id/title/description/search_hints/depends_on
    - Strips dangling depends_on references
    - Assigns sequential ids if the planner omitted any
    """
    stages_in = raw_plan.get("stages") or []
    out: list[PlanStage] = []
    seen_ids: set[str] = set()

    for i, s in enumerate(stages_in):
        if not isinstance(s, dict):
            continue
        sid = str(s.get("id") or f"s{i + 1}").strip()
        if sid in seen_ids:
            sid = f"{sid}_{i}"
        seen_ids.add(sid)

        title = str(s.get("title") or "").strip() or f"Stage {i + 1}"
        description = str(s.get("description") or "").strip()
        hints = s.get("search_hints") or []
        if isinstance(hints, str):
            hints = [hints]
        hints = [str(h).strip() for h in hints if str(h).strip()]
        deps = s.get("depends_on") or []
        if isinstance(deps, str):
            deps = [deps]
        deps = [str(d).strip() for d in deps if str(d).strip()]

        out.append({
            "id": sid,
            "title": title,
            "description": description,
            "search_hints": hints,
            "depends_on": deps,
            "status": s.get("status") or "pending",
            "summary": s.get("summary") or "",
        })

    # Clean dangling depends_on references
    valid_ids = {s["id"] for s in out}
    for s in out:
        s["depends_on"] = [d for d in s["depends_on"] if d in valid_ids]

    return {"stages": out}


def _public_plan(plan: dict) -> dict:
    """Trim internal fields for wire/display; keeps status/summary."""
    return {
        "stages": [
            {
                "id": s["id"],
                "title": s["title"],
                "description": s["description"],
                "search_hints": s.get("search_hints", []),
                "depends_on": s.get("depends_on", []),
                "status": s.get("status", "pending"),
                "summary": s.get("summary", ""),
            }
            for s in plan.get("stages", [])
        ]
    }


# ── Dependency-scoped context forwarding ─────────────────────────────────────

def gather_dependency_summaries(stage: PlanStage, all_stages: list[PlanStage]) -> str:
    """Collect summaries ONLY from stages this stage depends on."""
    dep_ids = set(stage.get("depends_on") or [])
    if not dep_ids:
        return "This is an independent stage. No prior context."

    by_id = {s["id"]: s for s in all_stages}
    parts: list[str] = []
    for did in dep_ids:
        s = by_id.get(did)
        if not s:
            continue
        if s.get("status") == "done" and s.get("summary"):
            parts.append(f"## {s['title']}\n{s['summary']}")
    return "\n\n".join(parts) if parts else \
        "Dependencies completed but produced no findings."


# ── Plan generation / amendment / confirmation ───────────────────────────────

def needs_plan(fast_llm: ChatOpenAI, query: str) -> bool:
    """Ask the fast LLM whether this query warrants a multi-stage plan."""
    try:
        resp = fast_llm.invoke([
            HumanMessage(content=NEEDS_PLAN_CHECK.format(query=query))
        ])
        raw = (resp.content or "").strip().upper()
        # Strip thinking, whitespace; first non-space char is the answer
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        return raw.startswith("PLAN")
    except Exception:
        log.exception("needs_plan classifier failed — defaulting to DIRECT")
        return False


def generate_plan(fast_llm: ChatOpenAI, query: str, seed_context: str,
                  tracer: Optional[ResearchTracer] = None) -> Optional[dict]:
    """Call the plan generator; return a normalised plan dict or None."""
    user_msg = PLAN_GENERATOR_USER_TEMPLATE.format(
        query=query,
        seed_context=seed_context or "No landscape survey available.",
    )
    log.info("PLAN_GENERATOR calling LLM with query=%r seed_len=%d",
             query[:120], len(seed_context or ""))
    try:
        t0 = time.time()
        resp = fast_llm.invoke([
            SystemMessage(content=PLAN_GENERATOR_SYSTEM),
            HumanMessage(content=user_msg),
        ])
        dur = time.time() - t0
        raw = resp.content or ""
        log.info("PLAN_GENERATOR raw (%d chars): %s", len(raw), raw[:400])
        if tracer:
            tracer.llm_call(
                caller="plan_generator",
                system_prompt_len=len(PLAN_GENERATOR_SYSTEM),
                user_prompt=user_msg,
                raw_response=raw,
                duration_s=dur,
            )
    except Exception as e:
        log.exception("PLAN_GENERATOR LLM call failed")
        if tracer:
            tracer.llm_call(
                caller="plan_generator",
                system_prompt_len=len(PLAN_GENERATOR_SYSTEM),
                user_prompt=user_msg,
                raw_response="",
                duration_s=0,
                error=str(e),
            )
        return None

    parsed = _extract_json(raw)
    if not parsed:
        log.warning("PLAN_GENERATOR: could not parse JSON from response")
        return None
    plan = _normalise_plan(parsed)
    if not plan["stages"]:
        log.warning("PLAN_GENERATOR: parsed plan has no stages")
        return None
    return plan


def amend_plan(fast_llm: ChatOpenAI, current_plan: dict, amendment_text: str,
               tracer: Optional[ResearchTracer] = None) -> Optional[dict]:
    """Apply a user's amendment request to the current plan."""
    current_public = _public_plan(current_plan)
    user_msg = PLAN_AMENDER_USER_TEMPLATE.format(
        current_plan_json=json.dumps(current_public, indent=2),
        amendment_text=amendment_text,
    )
    log.info("PLAN_AMENDER amending plan (stages=%d) with: %s",
             len(current_public["stages"]), amendment_text[:200])
    try:
        t0 = time.time()
        resp = fast_llm.invoke([
            SystemMessage(content=PLAN_AMENDER_SYSTEM),
            HumanMessage(content=user_msg),
        ])
        dur = time.time() - t0
        raw = resp.content or ""
        log.info("PLAN_AMENDER raw (%d chars): %s", len(raw), raw[:400])
        if tracer:
            tracer.llm_call(
                caller="plan_amender",
                system_prompt_len=len(PLAN_AMENDER_SYSTEM),
                user_prompt=user_msg,
                raw_response=raw,
                duration_s=dur,
            )
    except Exception as e:
        log.exception("PLAN_AMENDER LLM call failed")
        if tracer:
            tracer.llm_call(
                caller="plan_amender",
                system_prompt_len=len(PLAN_AMENDER_SYSTEM),
                user_prompt=user_msg,
                raw_response="",
                duration_s=0,
                error=str(e),
            )
        return None

    parsed = _extract_json(raw)
    if not parsed:
        log.warning("PLAN_AMENDER: could not parse JSON; keeping current plan")
        return None
    amended = _normalise_plan(parsed)
    if not amended["stages"]:
        log.warning("PLAN_AMENDER: amended plan has no stages; keeping current")
        return None
    return amended


def classify_plan_response(fast_llm: ChatOpenAI, user_message: str) -> str:
    """Return 'confirm' or 'amend' based on user's free-text reply to the plan."""
    txt = (user_message or "").strip()
    if not txt:
        return "amend"  # empty message can't be a confirmation
    # Cheap short-circuits for obvious confirmations
    low = txt.lower()
    if low in {"yes", "ok", "okay", "go", "start", "confirm", "confirmed",
               "proceed", "looks good", "lgtm", "run", "execute",
               "start research", "go ahead"}:
        return "confirm"
    try:
        resp = fast_llm.invoke([
            HumanMessage(content=PLAN_CONFIRM_CHECK.format(user_message=txt))
        ])
        raw = (resp.content or "").strip().upper()
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        # Grab first A/B token
        for ch in raw:
            if ch == "A":
                return "confirm"
            if ch == "B":
                return "amend"
        return "amend"
    except Exception:
        log.exception("classify_plan_response failed — defaulting to amend")
        return "amend"


# ── Outer DAG execution (sequential with topological order) ──────────────────

def topo_order(stages: list[PlanStage]) -> list[list[PlanStage]]:
    """Group stages into dependency-respecting waves.

    Returns a list of waves; each wave is a list of stages whose dependencies
    are satisfied by prior waves. Stages within a wave are order-independent
    (would be parallelisable).
    """
    remaining = {s["id"]: s for s in stages}
    done: set[str] = set()
    waves: list[list[PlanStage]] = []
    while remaining:
        wave = [
            s for s in remaining.values()
            if all(d in done for d in s.get("depends_on") or [])
        ]
        if not wave:
            # Cycle or dangling dep — bail gracefully by dumping the rest
            log.warning("topo_order: unresolved dependencies in %s",
                        list(remaining.keys()))
            waves.append(list(remaining.values()))
            break
        waves.append(wave)
        for s in wave:
            done.add(s["id"])
            remaining.pop(s["id"], None)
    return waves


def run_stage(
    stage: PlanStage,
    all_stages: list[PlanStage],
    *,
    user_query: str,
    llm_url: str,
    search_engine: str,
    search_api_key: Optional[str],
    enable_thinking: bool,
    stage_max_turns: int,
    stage_max_nodes: int,
    stage_max_concurrent: int,
    debug: bool,
    trace_dir: str,
    event_cb=None,
) -> dict:
    """Run one full inner MindSearch for a single plan stage.

    Returns a dict with keys: stage_id, summary, inner_nodes (list).
    event_cb(evt_type, payload) is called with inner-graph milestones if set.
    """
    from .lg_agent import build_graph, _dedup as _dedup_nodes  # lazy to avoid cycles

    log.info("STAGE [%s] starting: %s", stage["id"], stage["title"])

    dep_ctx = gather_dependency_summaries(stage, all_stages)
    hints = "\n".join(f"- {h}" for h in (stage.get("search_hints") or [])) \
            or "- (no explicit hints; plan as you see fit)"
    stage_query = STAGE_QUERY_TEMPLATE.format(
        stage_title=stage["title"],
        stage_description=stage["description"],
        search_hints=hints,
        dependency_context=dep_ctx,
    )

    # Each stage compiles its own fresh inner graph — seed already done at
    # the outer level, so enable_seed_search=False here.
    inner_graph, _tracer = build_graph(
        llm_url=llm_url,
        search_engine=search_engine,
        api_key=search_api_key,
        max_turns=stage_max_turns,
        max_nodes=stage_max_nodes,
        max_concurrent_searchers=stage_max_concurrent,
        enable_seed_search=False,
        enable_reflection=True,
        enable_compression=False,
        enable_thinking=enable_thinking,
        debug=debug,
        trace_dir=trace_dir,
    )

    inner_initial = {
        "question": stage_query,
        "nodes": [],
        "dispatched_ids": [],
        "to_dispatch_ids": [],
        "final_answer": "",
        "turns": 0,
        "seed_context": "",
        "reflection_notes": "",
    }

    summary = ""
    inner_nodes: list[dict] = []
    try:
        # Stream inner updates so the frontend can show stage progress.
        for event in inner_graph.stream(
            inner_initial,
            config={"recursion_limit": 200},
            stream_mode="updates",
        ):
            for node_name, update in event.items():
                if not isinstance(update, dict):
                    continue
                if node_name == "planner":
                    new_nodes = update.get("nodes") or []
                    if new_nodes and event_cb:
                        event_cb("stage_inner_planner", {
                            "stage_id": stage["id"],
                            "new_node_ids": [n["id"] for n in new_nodes],
                        })
                elif node_name == "dispatcher":
                    to_dispatch = update.get("to_dispatch_ids") or []
                    if to_dispatch and event_cb:
                        event_cb("stage_inner_dispatch", {
                            "stage_id": stage["id"],
                            "dispatched": list(to_dispatch),
                        })
                elif node_name == "searcher":
                    for n in update.get("nodes") or []:
                        inner_nodes.append(n)
                        if event_cb:
                            event_cb("stage_inner_searcher_done", {
                                "stage_id": stage["id"],
                                "node_id": n.get("id"),
                                "query": n.get("query"),
                                "summary_len": len(n.get("summary") or ""),
                            })
                elif node_name == "finalize":
                    summary = update.get("final_answer") or ""
    except Exception:
        log.exception("STAGE [%s] inner graph failed", stage["id"])

    if not summary:
        # Fallback: if finalize was never reached, synthesise what we have
        summary = _fallback_stage_summary(inner_nodes)

    log.info("STAGE [%s] done, summary length: %d", stage["id"], len(summary))
    return {
        "stage_id": stage["id"],
        "summary": summary,
        "inner_nodes": _dedup_nodes(inner_nodes),
    }


def _fallback_stage_summary(inner_nodes: list[dict]) -> str:
    """Concatenate any completed inner-node findings as a last-resort summary."""
    parts: list[str] = []
    for n in inner_nodes:
        if (n.get("summary") or "").strip():
            parts.append(f"### {n.get('query','(unknown query)')}\n{n['summary']}")
    return "\n\n".join(parts) if parts else "No findings."


# ── Narrative synthesis ──────────────────────────────────────────────────────

def narrative_synthesise(
    reason_llm: ChatOpenAI,
    *,
    query: str,
    stages: list[PlanStage],
    tracer: Optional[ResearchTracer] = None,
) -> str:
    """Weave completed stage summaries into the final report."""
    completed = [s for s in stages if s.get("status") == "done"]
    if not completed:
        return "No research stages completed."

    plan_outline = "\n".join(
        f"- [{s['id']}] {s['title']}"
        + (f" (depends on {', '.join(s['depends_on'])})" if s.get("depends_on") else "")
        for s in stages
    )

    all_summaries = "\n\n---\n\n".join(
        f"## Stage {s['id']}: {s['title']}\n{s.get('summary', '')}"
        for s in completed
    )

    user_msg = NARRATIVE_SYNTHESIS_USER_TEMPLATE.format(
        query=query,
        plan_outline=plan_outline,
        all_stage_summaries=all_summaries,
    )
    log.info("NARRATIVE_SYNTH calling LLM (user_msg=%d chars, stages=%d)...",
             len(user_msg), len(completed))
    try:
        t0 = time.time()
        resp = reason_llm.invoke([
            SystemMessage(content=NARRATIVE_SYNTHESIS_SYSTEM),
            HumanMessage(content=user_msg),
        ])
        dur = time.time() - t0
        raw = resp.content or ""
        if tracer:
            tracer.llm_call(
                caller="narrative_synthesis",
                system_prompt_len=len(NARRATIVE_SYNTHESIS_SYSTEM),
                user_prompt=user_msg,
                raw_response=raw,
                duration_s=dur,
            )
    except Exception as e:
        log.exception("NARRATIVE_SYNTH LLM call failed")
        if tracer:
            tracer.llm_call(
                caller="narrative_synthesis",
                system_prompt_len=len(NARRATIVE_SYNTHESIS_SYSTEM),
                user_prompt=user_msg,
                raw_response="",
                duration_s=0,
                error=str(e),
            )
        # Fallback: just concatenate stage summaries with headings
        return all_summaries

    # Strip <think> blocks if model leaks them
    text = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    text = re.sub(r'<start_of_thought>.*?<end_of_thought>', '', text,
                  flags=re.DOTALL).strip()
    return text or all_summaries


# ── LLM factories for plan-level calls ───────────────────────────────────────

def make_plan_llms(llm_url: str, enable_thinking: bool,
                   max_tokens: int = 8192) -> tuple[ChatOpenAI, ChatOpenAI]:
    """Return (fast_llm, reason_llm) for plan generation and synthesis.

    fast_llm — thinking off: plan generator, amender, classifier
    reason_llm — thinking on: narrative synthesis
    """
    fast_llm = ChatOpenAI(
        base_url=llm_url,
        api_key="sk-no-key-required",
        model="local",
        temperature=0.0,
        max_tokens=max_tokens,
        streaming=False,
        max_retries=6,
        extra_body={
            "cache_prompt": True,
            **({"chat_template_kwargs": {"enable_thinking": False}}
               if enable_thinking else {}),
        },
    )
    reason_llm = ChatOpenAI(
        base_url=llm_url,
        api_key="sk-no-key-required",
        model="local",
        temperature=0.0,
        max_tokens=max_tokens * 2,
        streaming=False,
        max_retries=6,
        extra_body={
            "cache_prompt": True,
            **({"chat_template_kwargs": {"enable_thinking": True}}
               if enable_thinking else {}),
        },
    )
    return fast_llm, reason_llm


# ── Seed search (reused) ─────────────────────────────────────────────────────

def run_seed_search(
    *,
    query: str,
    llm_url: str,
    search_engine: str,
    search_api_key: Optional[str],
    enable_thinking: bool,
    debug: bool,
    trace_dir: str,
) -> str:
    """Run only the seed searcher from the inner graph to produce a landscape
    survey that the plan generator can use as grounding."""
    # Reuse the inner graph assembled with seed enabled, but stop after seed.
    # The simplest approach is to directly build the seed node and invoke it.
    from .lg_agent import (
        build_graph,  # noqa: F401  (kept for parity / future extension)
        make_seed_searcher_node,
        make_search_tool,
    )
    search_tool = make_search_tool(engine=search_engine, api_key=search_api_key)
    fast_llm, _ = make_plan_llms(llm_url=llm_url,
                                 enable_thinking=enable_thinking)
    tracer = ResearchTracer(enabled=debug, output_dir=trace_dir)
    seed_node = make_seed_searcher_node(fast_llm, search_tool, tracer)
    state = {
        "question": query,
        "nodes": [],
        "dispatched_ids": [],
        "to_dispatch_ids": [],
        "final_answer": "",
        "turns": 0,
        "seed_context": "",
        "reflection_notes": "",
    }
    try:
        out = seed_node(state)
    except Exception:
        log.exception("run_seed_search failed")
        return ""
    return (out or {}).get("seed_context", "") or ""


# ── Convenience: full plan-flow execution (used by app.py) ──────────────────

def set_stage_status(plan: dict, stage_id: str, status: str,
                     summary: Optional[str] = None) -> None:
    """In-place update to a stage's status (and optionally summary)."""
    for s in plan.get("stages", []):
        if s["id"] == stage_id:
            s["status"] = status
            if summary is not None:
                s["summary"] = summary
            return


def clone_plan(plan: dict) -> dict:
    """Deep-copy a plan dict safely."""
    return copy.deepcopy(plan)
