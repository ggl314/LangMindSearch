# mindsearch/tracer.py

"""Structured execution tracer for LangMindSearch.

Captures every node execution, LLM call, state transition, and routing
decision. Writes a JSON trace file for post-hoc debugging.

Usage:
    tracer = ResearchTracer(enabled=True, output_dir="./traces")
    # ... pass tracer to nodes ...
    tracer.save()           # writes trace JSON
    tracer.print_summary()  # prints human-readable summary to stdout
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Optional

log = logging.getLogger("mindsearch.tracer")


class ResearchTracer:
    """Collects structured trace events during a research run."""

    def __init__(self, enabled: bool = False, output_dir: str = "./traces"):
        self.enabled = enabled
        self.output_dir = output_dir
        self.events: list[dict] = []
        self.start_time: float = time.time()
        self.run_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._event_counter: int = 0

    def _elapsed(self) -> float:
        return round(time.time() - self.start_time, 3)

    def _next_seq(self) -> int:
        self._event_counter += 1
        return self._event_counter

    # ── Core event recording ──────────────────────────────────────────

    def node_start(self, node_name: str, state_snapshot: dict) -> dict:
        """Call at the top of every graph node. Returns a context dict
        to pass to node_end."""
        if not self.enabled:
            return {"t0": time.time()}

        evt = {
            "seq": self._next_seq(),
            "type": "node_start",
            "node": node_name,
            "elapsed": self._elapsed(),
            "state_snapshot": state_snapshot,
        }
        self.events.append(evt)
        log.debug("TRACE node_start: %s (seq=%d)", node_name, evt["seq"])
        return {"t0": time.time(), "seq": evt["seq"]}

    def node_end(self, node_name: str, ctx: dict, output_delta: dict,
                 route: str = None, notes: str = None):
        """Call at the end of every graph node."""
        duration = round(time.time() - ctx.get("t0", time.time()), 3)
        if not self.enabled:
            log.info("TRACE %s completed in %.3fs", node_name, duration)
            return

        evt = {
            "seq": self._next_seq(),
            "type": "node_end",
            "node": node_name,
            "elapsed": self._elapsed(),
            "duration_s": duration,
            "output_delta": _safe_serialize(output_delta),
        }
        if route:
            evt["route"] = route
        if notes:
            evt["notes"] = notes
        self.events.append(evt)
        log.info("TRACE %s completed in %.3fs (seq=%d)%s",
                 node_name, duration, evt["seq"],
                 f" -> {route}" if route else "")

    def llm_call(self, caller: str, system_prompt_len: int,
                 user_prompt: str, raw_response: str,
                 parsed: Any = None, duration_s: float = 0,
                 error: str = None):
        """Record an LLM call with full prompts and response."""
        if not self.enabled:
            return

        evt = {
            "seq": self._next_seq(),
            "type": "llm_call",
            "caller": caller,
            "elapsed": self._elapsed(),
            "duration_s": round(duration_s, 3),
            "system_prompt_chars": system_prompt_len,
            "user_prompt_chars": len(user_prompt) if user_prompt else 0,
            "user_prompt": (user_prompt[:2000]) if user_prompt else None,  # cap for trace size
            "raw_response_chars": len(raw_response) if raw_response else 0,
            "raw_response": (raw_response[:3000]) if raw_response else None,
        }
        if parsed is not None:
            evt["parsed"] = _safe_serialize(parsed)
        if error:
            evt["error"] = error
        self.events.append(evt)

    def tool_call(self, caller: str, tool_name: str,
                  args: dict, result_len: int, duration_s: float = 0,
                  error: str = None):
        """Record a tool invocation."""
        if not self.enabled:
            return

        evt = {
            "seq": self._next_seq(),
            "type": "tool_call",
            "caller": caller,
            "elapsed": self._elapsed(),
            "duration_s": round(duration_s, 3),
            "tool_name": tool_name,
            "args": _safe_serialize(args),
            "result_chars": result_len,
        }
        if error:
            evt["error"] = error
        self.events.append(evt)

    def route_decision(self, router_name: str, decision: str,
                       reason: str = None):
        """Record a conditional routing decision."""
        if not self.enabled:
            return

        evt = {
            "seq": self._next_seq(),
            "type": "route",
            "router": router_name,
            "elapsed": self._elapsed(),
            "decision": decision,
        }
        if reason:
            evt["reason"] = reason
        self.events.append(evt)

    def budget_check(self, total_nodes: int, max_nodes: int,
                     turns: int, max_turns: int,
                     categories: dict):
        """Record a budget/constraint check."""
        if not self.enabled:
            return

        self.events.append({
            "seq": self._next_seq(),
            "type": "budget_check",
            "elapsed": self._elapsed(),
            "total_nodes": total_nodes,
            "max_nodes": max_nodes,
            "turns": turns,
            "max_turns": max_turns,
            "categories": categories,
        })

    # ── State snapshot helpers ────────────────────────────────────────

    def snapshot_state(self, state: dict) -> dict:
        """Create a serializable snapshot of MindSearchState.

        Captures structure without massive content. Full content is in
        the node-specific events.
        """
        if not self.enabled:
            return {}

        nodes = state.get("nodes", []) or []
        # Dedup for snapshot
        seen = {}
        for n in nodes:
            seen[n["id"]] = n
        deduped = list(seen.values())

        return {
            "question": (state.get("question", "") or "")[:200],
            "turns": state.get("turns", 0),
            "seed_context_len": len(state.get("seed_context", "") or ""),
            "reflection_notes_len": len(state.get("reflection_notes", "") or ""),
            "nodes": [
                {
                    "id": n["id"],
                    "status": n["status"],
                    "category": n.get("category", "core"),
                    "query": (n.get("query", "") or "")[:100],
                    "summary_len": len(n.get("summary", "") or ""),
                    "leads_len": len(n.get("leads", "") or ""),
                }
                for n in deduped
            ],
            "dispatched_ids": state.get("dispatched_ids", []),
            "to_dispatch_ids": state.get("to_dispatch_ids", []),
        }

    # ── Output ────────────────────────────────────────────────────────

    def save(self, question: str = "") -> Optional[str]:
        """Write trace to JSON file. Returns the file path."""
        if not self.enabled or not self.events:
            return None

        os.makedirs(self.output_dir, exist_ok=True)
        filename = f"trace_{self.run_id}.json"
        filepath = os.path.join(self.output_dir, filename)

        trace = {
            "run_id": self.run_id,
            "question": question,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "total_duration_s": self._elapsed(),
            "total_events": len(self.events),
            "summary": self._build_summary(),
            "events": self.events,
        }

        with open(filepath, "w") as f:
            json.dump(trace, f, indent=2, default=str)

        log.info("TRACE saved to %s (%d events)", filepath, len(self.events))
        return filepath

    def print_summary(self):
        """Print a human-readable execution summary to stdout."""
        summary = self._build_summary()
        if not summary:
            return

        print("\n" + "=" * 70)
        print("RESEARCH EXECUTION TRACE")
        print("=" * 70)
        print(f"Run ID:     {self.run_id}")
        print(f"Duration:   {summary['total_duration_s']}s")
        print(f"LLM calls:  {summary['llm_calls']} "
              f"(total {summary['llm_total_duration_s']}s)")
        print(f"Tool calls: {summary['tool_calls']}")
        print(f"Nodes:      {summary['nodes_completed']}/{summary['nodes_total']}")
        print(f"Categories: {summary['categories']}")
        print()

        # Print execution flow
        print("EXECUTION FLOW:")
        print("-" * 70)
        for evt in self.events:
            t = f"[{evt['elapsed']:7.1f}s]"
            if evt["type"] == "node_start":
                print(f"  {t} > {evt['node']}")
            elif evt["type"] == "node_end":
                dur = evt.get("duration_s", 0)
                route = evt.get("route", "")
                notes = evt.get("notes", "")
                suffix = f" -> {route}" if route else ""
                if notes:
                    suffix += f" ({notes})"
                print(f"  {t} # {evt['node']} ({dur:.1f}s){suffix}")
            elif evt["type"] == "llm_call":
                dur = evt.get("duration_s", 0)
                err = " X" if evt.get("error") else ""
                print(f"  {t}   LLM {evt['caller']} "
                      f"({dur:.1f}s, {evt.get('raw_response_chars',0)}ch){err}")
            elif evt["type"] == "tool_call":
                dur = evt.get("duration_s", 0)
                err = " X" if evt.get("error") else ""
                print(f"  {t}   TOOL {evt['caller']} -> {evt['tool_name']} "
                      f"({dur:.1f}s, {evt.get('result_chars',0)}ch){err}")
            elif evt["type"] == "route":
                print(f"  {t}   ROUTE {evt['router']} -> {evt['decision']}"
                      f"{' (' + evt['reason'] + ')' if evt.get('reason') else ''}")
            elif evt["type"] == "budget_check":
                print(f"  {t}   BUDGET nodes={evt['total_nodes']}/{evt['max_nodes']} "
                      f"turns={evt['turns']}/{evt['max_turns']} "
                      f"cats={evt['categories']}")

        # Print node timeline
        print()
        print("NODE TIMELINE:")
        print("-" * 70)
        node_events = {}
        for evt in self.events:
            if evt["type"] in ("node_start", "node_end") and evt["node"] == "searcher":
                # Searcher events have node_id in notes or state
                continue
            if evt["type"] == "node_start":
                node_events.setdefault(evt["node"], []).append(("start", evt["elapsed"]))
            elif evt["type"] == "node_end":
                node_events.setdefault(evt["node"], []).append(
                    ("end", evt["elapsed"], evt.get("duration_s", 0))
                )

        for node, events in node_events.items():
            parts = []
            for e in events:
                if e[0] == "start":
                    parts.append(f"start@{e[1]:.1f}s")
                else:
                    parts.append(f"end@{e[1]:.1f}s ({e[2]:.1f}s)")
            print(f"  {node:20s} {' -> '.join(parts)}")

        print("=" * 70)

    def _build_summary(self) -> dict:
        """Build a summary dict from events."""
        if not self.events:
            return {}

        llm_events = [e for e in self.events if e["type"] == "llm_call"]
        tool_events = [e for e in self.events if e["type"] == "tool_call"]
        node_end_events = [e for e in self.events if e["type"] == "node_end"]

        # Count unique search nodes (from budget_check or node events)
        budget_events = [e for e in self.events if e["type"] == "budget_check"]
        if budget_events:
            last_budget = budget_events[-1]
            nodes_total = last_budget["total_nodes"]
            categories = last_budget["categories"]
        else:
            nodes_total = 0
            categories = {}

        searcher_ends = [e for e in node_end_events if e["node"] == "searcher"]
        nodes_completed = len(searcher_ends)

        return {
            "total_duration_s": self._elapsed(),
            "llm_calls": len(llm_events),
            "llm_total_duration_s": round(
                sum(e.get("duration_s", 0) for e in llm_events), 1
            ),
            "llm_errors": len([e for e in llm_events if e.get("error")]),
            "tool_calls": len(tool_events),
            "tool_errors": len([e for e in tool_events if e.get("error")]),
            "nodes_total": nodes_total,
            "nodes_completed": nodes_completed,
            "categories": categories,
            "route_decisions": [
                {"router": e["router"], "decision": e["decision"]}
                for e in self.events if e["type"] == "route"
            ],
        }


def _safe_serialize(obj: Any) -> Any:
    """Make an object JSON-serializable, falling back to str()."""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(v) for v in obj]
    return str(obj)
