# LangMindSearch v3: Test Query for Full Transition Coverage

## The Query

```
What are the practical trade-offs of running local LLM inference on consumer 
hardware versus renting cloud GPU compute for a small team, and how has this 
changed in the last year?
```

## Why This Query

This query is engineered to trigger every transition in the state machine.
Here's the mapping from query properties to transitions.

---

## Transition Coverage Analysis

### Transitions triggered by query structure

| # | Transition | What triggers it | Expected behaviour |
|---|-----------|-----------------|-------------------|
| 1 | `START → seed_searcher` | Enabled by default | Seed searcher runs broad searches like "local LLM inference hardware 2025", "cloud GPU rental vs local inference" |
| 2 | `seed_searcher → planner` | Unconditional edge | Planner receives landscape survey covering the topic space |
| 3 | `planner → dispatcher` (Turn 1) | Planner emits 3-5 nodes | Natural decomposition into parallel sub-questions (see below) |
| 4 | `dispatcher → searcher` (fan-out) | Multiple independent nodes | At least 2-3 nodes have no dependencies and dispatch in parallel |
| 5 | `searcher → reflect` (batch 1) | Auto-join after parallel batch | All parallel searchers complete, reflect fires once |
| 6 | `reflect → dispatcher` | Dependency chaining | At least one node depends on a just-completed node (see below) |
| 7 | `dispatcher → searcher` (dep-chain) | Unblocked dependent node | The dependent node is now dispatchable |
| 8 | `searcher → reflect` (batch 2) | Dependent searcher completes | Reflect fires for the second batch |
| 9 | `reflect → planner` (with LLM) | All nodes done, ≥3 completed | Reflection LLM assesses coverage, writes reflection_notes |
| 10 | `planner → dispatcher` (Turn 2) | Planner responds to reflection gaps | Planner adds more nodes based on reflection feedback |
| 11 | `dispatcher → searcher → reflect → planner` (Turn 3) | Still not comprehensive enough | Third planning cycle |
| 12 | `planner → finalize` | Planner says finalize or budget nears cap | No more pending nodes |
| 13 | `finalize → END` | Unconditional | Final report generated |

### Why dependency chaining will trigger (`reflect → dispatcher`)

The query has a natural two-phase structure:

**Phase 1** (independent, parallel):
- "What consumer hardware can run LLMs locally?" (core)
- "What cloud GPU rental options exist?" (core)
- "History of local inference cost trends" (context)

**Phase 2** (depends on Phase 1):
- "Compare performance per dollar: local vs cloud" (core, depends on the first two)

A competent planner will create the comparison node with `depends_on: ["n1", "n2"]`.
This forces the `reflect → dispatcher` path: after n1 and n2 complete, reflect
sees n4 (the comparison) is pending with satisfied dependencies, routes to
dispatcher, which dispatches n4.

### Why multiple planning rounds will trigger (`reflect → planner`, multiple times)

The query spans several dimensions that a single planning round won't cover:

- **Core**: hardware specs, cloud pricing, performance comparison
- **Context**: how this has changed in the last year (requires temporal search)
- **Adjacent**: power consumption, noise, network latency, data privacy implications
- **Emerging**: new hardware releases (RTX 5090, Mac Studio M4 Ultra), new cloud providers
- **Critical**: hidden costs (electricity, cooling, maintenance), reliability, scaling limits

A good planner will cover 3-4 of these in Turn 1, get reflection feedback
identifying gaps ("no coverage of hidden costs or privacy implications"), and
add more nodes in Turn 2. Turn 3 may add emerging/critical nodes if the
reflection identifies remaining gaps.

### Why category diversity will trigger

The query naturally spans all five categories:

| Category | Natural sub-questions |
|----------|----------------------|
| **core** | What hardware? What cloud options? What's the perf/$ comparison? |
| **context** | How has local inference evolved in the last year? What changed? |
| **adjacent** | Data privacy of cloud vs local. Power costs. Team workflow implications. |
| **emerging** | New consumer GPUs (5090, M4 Ultra). New quantisation methods. New cloud pricing. |
| **critical** | Hidden costs. Scaling limits. Reliability. When does each approach fail? |

### Why the query won't terminate too early

The "how has this changed in the last year" clause forces temporal research
that can't be collapsed into a single search. The "practical trade-offs"
framing encourages the planner to explore multiple dimensions rather than
giving a quick factual answer. The "small team" qualifier adds specificity
that requires dedicated research (not just "individual vs enterprise").

### Why the query won't run forever

The topic is bounded — there are only so many consumer GPUs, cloud providers,
and cost dimensions. The MAX_NODES=15 budget is sufficient but will be
reached in 2-3 planning rounds. The planner should naturally finalize after
covering the main dimensions.

---

## Expected Execution Trace (Happy Path)

```
[  0.0s] ▶ seed_searcher
[  0.0s]   🤖 seed_searcher LLM (tool call)
[  1.5s]   🔧 seed_searcher → search (3 broad queries)
[  3.0s]   🤖 seed_searcher LLM (synthesis)
[  5.0s] ◼ seed_searcher (5.0s)

[  5.0s] ▶ planner (turn 0)
[  5.0s]   📊 nodes=0/15 turns=0/8 cats={}
[  5.1s]   🤖 planner LLM
[  9.0s] ◼ planner (4.0s) → emits n1(core), n2(core), n3(context), n4(core, depends_on=[n1,n2])

[  9.0s]   ↗ route_after_planner → dispatcher (has_pending=True)
[  9.0s] ▶ dispatcher
[  9.0s] ◼ dispatcher → dispatching [n1, n2, n3] (n4 blocked by n1,n2)

[  9.0s]   ↗ dispatch_sends → fan_out([n1, n2, n3])
[  9.1s] ▶ searcher[n1] ┐
[  9.1s] ▶ searcher[n2] ├ parallel
[  9.1s] ▶ searcher[n3] ┘
[  9.1s]   🤖 searcher[n1] LLM round 1
[  9.2s]   🔧 searcher[n1] → search
[  ...parallel execution...]
[ 18.0s] ◼ searcher[n1] (8.9s)
[ 19.0s] ◼ searcher[n2] (9.9s)
[ 17.0s] ◼ searcher[n3] (7.9s)

[ 19.0s] ▶ reflect (auto-join after n1+n2+n3)
            pending=[n4], n4.depends_on=[n1,n2] both done → skip LLM
[ 19.0s]   ↗ reflect_router → dispatcher (pending unblocked)

[ 19.0s] ▶ dispatcher
[ 19.0s] ◼ dispatcher → dispatching [n4]
[  ...n4 searches...]
[ 27.0s] ◼ searcher[n4] (8.0s)

[ 27.0s] ▶ reflect (auto-join after n4)
            no pending, 4 done → run LLM reflection
[ 27.0s]   🤖 reflect LLM
[ 30.0s] ◼ reflect (3.0s) → writes reflection_notes
[ 30.0s]   ↗ reflect_router → planner (no pending)

[ 30.0s] ▶ planner (turn 1)
[ 30.0s]   📊 nodes=4/15 turns=1/8 cats={core:3, context:1}
            reflection_notes: "No coverage of: cloud pricing specifics, hidden costs
            (electricity, cooling), data privacy, emerging hardware (RTX 5090).
            Adjacent and critical categories empty."
[ 30.1s]   🤖 planner LLM
[ 34.0s] ◼ planner (4.0s) → emits n5(adjacent), n6(critical), n7(emerging), n8(core)

[  ...cycle repeats: dispatcher → searchers → reflect...]
[  ...reflect identifies remaining gaps...]

[ 55.0s] ▶ planner (turn 2)
[ 55.0s]   📊 nodes=8/15 turns=2/8 cats={core:4, context:1, adjacent:1, critical:1, emerging:1}
            reflection_notes: "Good coverage across categories. Minor gap: no
            comparison of quantisation methods affecting local viability.
            Consider finalizing — research is reasonably comprehensive."
[ 55.1s]   🤖 planner LLM
[ 58.0s] ◼ planner → emits n9(core), n10(emerging) then...

[ 70.0s] ▶ planner (turn 3)
[ 70.0s]   📊 nodes=10/15 turns=3/8 cats={core:5, context:1, adjacent:1, critical:1, emerging:2}
            reflection_notes: "Comprehensive coverage achieved."
[ 70.1s]   🤖 planner LLM → {"action": "finalize"}
[ 73.0s] ◼ planner → no new nodes

[ 73.0s]   ↗ route_after_planner → finalize (no pending)

[ 73.0s] ▶ finalize
[ 73.0s]   🤖 finalize LLM (synthesis)
[ 82.0s] ◼ finalize (9.0s) → writes final_answer

[ 82.0s] → END
```

**Expected totals**: ~10-12 nodes, 3-4 planner turns, 12-16 LLM calls, 8-12 tool calls, ~80-120s total.

---

## Transition Verification Checklist

After running the test query with `--debug`, verify each transition occurred
by checking the trace file:

```bash
# Extract all transitions from the trace
cat traces/trace_*.json | python3 -c "
import json, sys
trace = json.load(sys.stdin)
for evt in trace['events']:
    if evt['type'] == 'node_start':
        print(f\"  ▶ {evt['node']:20s} @ {evt['elapsed']:.1f}s\")
    elif evt['type'] == 'node_end':
        r = evt.get('route', '')
        print(f\"  ◼ {evt['node']:20s} @ {evt['elapsed']:.1f}s ({evt.get('duration_s',0):.1f}s){' → '+r if r else ''}\")
    elif evt['type'] == 'route':
        print(f\"  ↗ {evt['router']:20s} → {evt['decision']}\")
"
```

Then check off each transition:

- [ ] `seed_searcher` started and completed
- [ ] `planner` Turn 0 emitted nodes with mixed categories
- [ ] `route_after_planner` → `dispatcher` (Turn 0)
- [ ] `dispatcher` fan-out to 2+ parallel searchers
- [ ] All parallel searchers completed
- [ ] `reflect` fired exactly once after parallel batch (not once per searcher)
- [ ] `reflect → dispatcher` (dependency chaining, at least once)
- [ ] `dispatcher` dispatched the dependency-blocked node
- [ ] `reflect` ran LLM and wrote `reflection_notes` (at least once)
- [ ] `reflect → planner` (at least once)
- [ ] `planner` Turn 1+ referenced reflection feedback in its context
- [ ] `planner` Turn 1+ emitted nodes addressing reflection gaps
- [ ] At least 3 of 5 categories represented across all nodes
- [ ] `planner → finalize` (either explicit finalize or budget/turn exhaustion)
- [ ] `finalize` completed and wrote `final_answer`
- [ ] Leads parsed: at least some searcher nodes have non-empty `leads` field
- [ ] Budget stayed within MAX_NODES

### Transitions that may or may not occur (both are valid)

- [ ] `dispatcher → planner` (empty dispatch fallback) — valid but not guaranteed
- [ ] Budget exhaustion forced finalize — depends on how many nodes planner creates
- [ ] Turn exhaustion forced finalize — depends on when planner decides to finalize
- [ ] Compression in finalize — only if `ENABLE_COMPRESSION=true` and findings are large

---

## Running the Test

### Terminal mode (recommended for first test)

```bash
cd LangMindSearch

python -m mindsearch.terminal \
  --debug \
  --query "What are the practical trade-offs of running local LLM inference on consumer hardware versus renting cloud GPU compute for a small team, and how has this changed in the last year?"
```

This will:
1. Run the full pipeline
2. Print the execution flow with emoji markers
3. Print the trace summary with timing
4. Save the full trace to `./traces/trace_YYYYMMDD_HHMMSS.json`

### Via the API

```bash
# Start the server with debug enabled
DEBUG=true python -m mindsearch.app --llm_url http://localhost:8080/v1

# Send the query
curl -X POST http://localhost:8002/solve \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": "What are the practical trade-offs of running local LLM inference on consumer hardware versus renting cloud GPU compute for a small team, and how has this changed in the last year?"
  }' \
  --no-buffer
```

### Inspecting the trace

```bash
# Pretty-print the summary
cat traces/trace_*.json | python3 -c "
import json, sys
t = json.load(sys.stdin)
s = t['summary']
print(f\"Duration: {s['total_duration_s']}s\")
print(f\"LLM calls: {s['llm_calls']} ({s['llm_total_duration_s']}s)\")
print(f\"Tool calls: {s['tool_calls']}\")
print(f\"Nodes: {s['nodes_completed']}/{s['nodes_total']}\")
print(f\"Categories: {s['categories']}\")
print(f\"Errors: LLM={s['llm_errors']}, Tool={s['tool_errors']}\")
print()
print('Route decisions:')
for r in s['route_decisions']:
    print(f\"  {r['router']} → {r['decision']}\")
"
```

```bash
# Find all planner LLM calls and see what they produced
cat traces/trace_*.json | python3 -c "
import json, sys
t = json.load(sys.stdin)
for evt in t['events']:
    if evt['type'] == 'llm_call' and evt['caller'] == 'planner':
        print(f\"--- Planner call @ {evt['elapsed']:.1f}s ({evt['duration_s']:.1f}s) ---\")
        print(f\"Response ({evt['raw_response_chars']} chars):\")
        print(evt.get('raw_response', '')[:500])
        print()
"
```

```bash
# Check if reflection influenced planning
cat traces/trace_*.json | python3 -c "
import json, sys
t = json.load(sys.stdin)
for evt in t['events']:
    if evt['type'] == 'llm_call' and evt['caller'] == 'reflect':
        print(f\"--- Reflection @ {evt['elapsed']:.1f}s ---\")
        print(evt.get('raw_response', '')[:300])
        print()
    if evt['type'] == 'llm_call' and evt['caller'] == 'planner':
        prompt = evt.get('user_prompt', '')
        if 'Reflection feedback' in prompt:
            # Extract just the reflection section
            start = prompt.find('Reflection feedback')
            end = prompt.find('Research graph', start)
            print(f\"--- Planner saw this reflection @ {evt['elapsed']:.1f}s ---\")
            print(prompt[start:end][:300])
            print()
"
```

---

## Fallback Test Queries

If the primary query doesn't trigger dependency chaining (the planner may
choose not to use `depends_on`), try these alternatives that more strongly
encourage it:

### Forces comparison (dependency chaining)

```
Compare the mass transit systems of London, Tokyo, and New York: 
what could each city learn from the other two?
```

A competent planner will research each city independently (parallel), then
create comparison nodes that depend on two or three prior nodes.

### Forces temporal + emerging (multi-round planning)

```
How has the competitive landscape for open-weight LLMs changed since 
Llama 2's release, and what does the next 12 months likely look like?
```

Turn 1 covers historical and current state. Reflection will flag missing
forward-looking analysis. Turn 2 adds emerging/speculative nodes.

### Forces critical category (controversies)

```
What are the arguments for and against allowing autonomous AI agents 
to manage personal financial portfolios without human oversight?
```

Naturally splits into pro/con research with critical-category nodes for
risks, regulatory concerns, and failure cases.
