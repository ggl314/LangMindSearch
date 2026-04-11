# LangMindSearch v3: State Machine & Transition Reference

## Purpose

This document is the authoritative reference for how the LangMindSearch research
graph executes. It defines every node, every possible transition, the exact
conditions that govern each routing decision, and the precise state mutations
each node performs. Use it to:

1. **Understand** what should happen at each step
2. **Debug** by comparing observed behaviour against this spec
3. **Identify** invalid transitions (if you see a transition not listed here,
   something is broken)

---

## 1. Graph Topology (v3 with all features enabled)

```
                    ┌─────────────────────────────────────────────┐
                    │                                             │
                    ▼                                             │
              seed_searcher                                      │
                    │                                             │
                    │ (unconditional)                             │
                    ▼                                             │
    ┌──────── planner ◄──────────────────────────┐               │
    │               │                            │               │
    │     route_after_planner                    │               │
    │         │           │                      │               │
    │    has_pending   no_pending                │               │
    │         │           │                      │               │
    │         ▼           ▼                      │               │
    │    dispatcher    finalize ──► END          │               │
    │         │                                  │               │
    │    _dispatch_sends                         │               │
    │      │          │                          │               │
    │   fan_out    nothing                       │               │
    │      │       to_send ──────────────────────┘               │
    │      │                                                     │
    │      ▼                                                     │
    │  searcher (×N parallel)                                    │
    │      │                                                     │
    │      │ (unconditional, auto-join)                          │
    │      ▼                                                     │
    │   reflect                                                  │
    │      │                                                     │
    │  reflect_router                                            │
    │    │           │                                           │
    │  pending    all_done ──────────────────────────────────────┘
    │    │
    │    ▼
    └─► dispatcher
```

### Complete List of Valid Transitions

| From | To | Condition | Edge Type |
|------|----|-----------|-----------|
| `START` | `seed_searcher` | `enable_seed_search=True` | entry point |
| `START` | `planner` | `enable_seed_search=False` | entry point |
| `seed_searcher` | `planner` | always | unconditional |
| `planner` | `dispatcher` | pending nodes exist in state | conditional: `route_after_planner` |
| `planner` | `finalize` | no pending nodes (finalize or budget/turn exhaustion) | conditional: `route_after_planner` |
| `dispatcher` | `searcher` (×N) | dispatcher found unblocked nodes to send | conditional: `_dispatch_sends` returns `Send()` list |
| `dispatcher` | `planner` | dispatcher found nothing new to send | conditional: `_dispatch_sends` returns `"planner"` |
| `searcher` | `reflect` | always (after each searcher completes) | unconditional |
| `reflect` | `dispatcher` | pending unblocked nodes exist | conditional: `reflect_router` |
| `reflect` | `planner` | no pending nodes remain | conditional: `reflect_router` |
| `finalize` | `END` | always | unconditional |

### Invalid Transitions (if you see these, something is broken)

| From | To | Why Invalid |
|------|----|-------------|
| `planner` | `searcher` | Planner never dispatches directly; it goes through dispatcher |
| `planner` | `reflect` | Planner goes to dispatcher or finalize only |
| `searcher` | `planner` | Searcher always goes to reflect first |
| `searcher` | `dispatcher` | Searcher always goes to reflect first |
| `searcher` | `finalize` | Searcher always goes to reflect first |
| `reflect` | `finalize` | Reflect goes to dispatcher or planner only |
| `reflect` | `searcher` | Reflect never dispatches directly |
| `finalize` | anything | Finalize always goes to END |
| `dispatcher` | `reflect` | Dispatcher goes to searcher(s) or planner only |
| `dispatcher` | `finalize` | Dispatcher goes to searcher(s) or planner only |
| `seed_searcher` | anything except `planner` | Seed searcher has exactly one outgoing edge |

---

## 2. Node Specifications

### 2.1 `seed_searcher`

**Purpose**: Run broad web searches to discover the topic landscape before planning.

**Runs**: Exactly once, at the very start of execution. Never re-entered.

**Input state read**:
- `question` — the user's query

**Output state written**:
- `seed_context: str` — the landscape summary (TOPIC MAP + KEY ENTITIES)

**LLM calls**: 1-2 (initial call with tools, then synthesis if tool calls were made)

**Tool calls**: 1-3 search invocations (broad queries)

**Can fail**: Yes. On failure, writes `seed_context = ""`. Graph continues.

**Next node**: Always `planner`.

**Invariants**:
- After this node, `seed_context` is a non-empty string (success) or `""` (failure)
- No other state fields are modified
- This node does not touch `nodes`, `turns`, `reflection_notes`, or any other field

---

### 2.2 `planner`

**Purpose**: Decide what to search next. Emits a JSON object that either adds new search nodes or signals finalize.

**Runs**: Multiple times. Once per planning round. Capped by `MAX_TURNS`.

**Input state read**:
- `question` — original query
- `seed_context` — landscape survey (from seed_searcher)
- `reflection_notes` — coverage assessment (from reflect, empty on first call)
- `nodes` — all existing nodes with their status, summaries, leads, categories
- `turns` — current turn counter

**Output state written**:
- `nodes: list[SearchNode]` — zero or more NEW pending nodes (appended via `operator.add`)
- `turns: int` — incremented by 1 every call

**The planner never modifies existing nodes.** It only appends new pending ones.

**LLM calls**: Exactly 1 per invocation.

**Decision logic** (what the planner LLM produces):
- `{"action": "search", "nodes": [...]}` → new nodes added, turns incremented
- `{"action": "finalize"}` → no nodes added, turns incremented (triggers route to finalize)
- JSON parse failure → no nodes added, turns incremented (forced finalize)

**Budget enforcement** (in code, after LLM response):
- If `turns >= MAX_TURNS`: return empty nodes, increment turns (force finalize)
- If `total_existing_nodes >= MAX_NODES`: return empty nodes, increment turns (force finalize)
- If new nodes would exceed remaining budget: trim to fit

**Next node**: Determined by `route_after_planner`:

| Condition | Route |
|-----------|-------|
| New pending nodes exist in state after this call | `dispatcher` |
| No pending nodes AND (`turns >= MAX_TURNS` OR planner said finalize OR budget exhausted) | `finalize` |

**Invariants**:
- `turns` always increases by exactly 1
- New nodes always have `status="pending"`, `summary=""`, `leads=""`
- Node IDs are guaranteed unique (dedup suffix logic)
- The planner never deletes or modifies existing nodes
- The planner never writes to `seed_context`, `reflection_notes`, or `final_answer`

---

### 2.3 `dispatcher`

**Purpose**: Select which pending nodes are ready to run (dependencies satisfied, not already dispatched) and mark them for fan-out.

**Runs**: After planner (if pending nodes exist) or after reflect (if pending nodes were unblocked by a just-completed batch).

**Input state read**:
- `nodes` — to find pending nodes and check dependency satisfaction
- `dispatched_ids` — to avoid re-dispatching nodes already in flight

**Output state written**:
- `dispatched_ids: list[str]` — IDs of newly dispatched nodes (appended via `operator.add`)
- `to_dispatch_ids: list[str]` — IDs selected this round (overwritten, not appended)

**LLM calls**: None. This is pure logic.

**Selection logic**:
A node is dispatchable if ALL of:
1. `status == "pending"`
2. Every ID in `depends_on` is in the completed set
3. Its ID is NOT in `dispatched_ids` (never been dispatched before)

**Next node**: Determined by `_dispatch_sends`:

| Condition | Route |
|-----------|-------|
| `to_dispatch_ids` is non-empty | `Send("searcher", ...)` for each dispatchable node (fan-out) |
| `to_dispatch_ids` is empty | `planner` (fallback — nothing to do) |

**Invariants**:
- `dispatched_ids` only grows, never shrinks
- A node ID appears in `dispatched_ids` at most once
- `to_dispatch_ids` is ephemeral — it's the "current batch" and is overwritten each call
- Dispatcher never modifies `nodes`, `turns`, or any other field
- Dispatcher never creates, deletes, or changes node status

**The empty-dispatch fallback to planner**: This happens when reflect routes to dispatcher because it saw pending nodes, but those pending nodes were already dispatched in a prior round (they're in-flight in parallel searchers). Dispatcher finds nothing new to send, so falls back to planner. This is a valid but unusual path.

---

### 2.4 `searcher`

**Purpose**: Execute web searches for a single node and produce findings + leads.

**Runs**: In parallel. LangGraph's `Send()` mechanism creates independent instances. Multiple searchers can run concurrently for independent nodes.

**Input**: `SearcherState` (NOT `MindSearchState` — this is a sub-state):
- `node_id` — which node this searcher is responsible for
- `query` — the search query
- `question` — the original user question (for context)
- `depends_on` — dependency list (for metadata only, already satisfied)
- `category` — forwarded from planner
- `rationale` — forwarded from planner

**Output state written**:
- `nodes: list[SearchNode]` — exactly ONE node with `status="done"`, containing:
  - `summary` — parsed FINDINGS section
  - `leads` — parsed LEADS section
  - Same `id`, `query`, `depends_on`, `category`, `rationale` as input

**How the pending→done transition works**:
The searcher emits a NEW `SearchNode` dict with the same `id` but `status="done"`.
Because `nodes` uses `operator.add`, this is APPENDED to the list (the old pending
entry remains). The `_dedup()` function (last-write-wins) resolves this: when any
other node calls `_dedup(state["nodes"])`, the done version wins because it was
appended later.

**LLM calls**: 1-3 (ReAct loop: call LLM, if it requests tool calls, execute them, call LLM again, up to 3 rounds).

**Tool calls**: 0-3 search invocations per round.

**Can fail**: Yes. LLM call failure or tool failure raises an exception. **This crashes the entire graph.** There is no try/except that catches and produces a partial result. (This is a known gap — consider wrapping in try/except that returns `summary="Search failed"` instead of crashing.)

**Next node**: Always `reflect`. This is an unconditional edge.

**LangGraph parallel execution semantics**:
- Multiple `Send("searcher", ...)` calls from dispatcher create parallel branches
- Each branch runs independently with its own `SearcherState`
- All branches MUST complete before `reflect` runs
- LangGraph auto-joins: `reflect` fires exactly ONCE after ALL searchers in the batch finish
- The joined state contains ALL searcher outputs accumulated via `operator.add`

**Invariants**:
- Each searcher invocation produces exactly one node update
- The node ID in the output matches the `node_id` from input
- The output node always has `status="done"` (never pending, never any other value)
- Searcher never reads or writes `turns`, `seed_context`, `reflection_notes`, `final_answer`
- Searcher never modifies any node other than its own

---

### 2.5 `reflect`

**Purpose**: Assess research coverage after a searcher batch completes. Write actionable guidance for the planner. Route to dispatcher (if pending nodes exist) or planner (for next planning round).

**Runs**: Once per searcher batch. Due to LangGraph auto-join, this fires exactly once after ALL parallel searchers in a batch complete — not once per searcher.

**Input state read**:
- `nodes` — all nodes (to assess coverage and find pending)
- `question` — for the reflection LLM call

**Output state written**:
- `reflection_notes: str` — coverage assessment for the planner (overwritten each time)

**When the LLM call is skipped**:
1. If there are pending nodes with unblocked dependencies → skip LLM, just route to dispatcher. Rationale: those nodes need to run before reflection can be meaningful.
2. If fewer than 3 nodes are done → skip LLM. Rationale: not enough data to assess.

**When the LLM call fires**:
All nodes are done (no pending) AND at least 3 completed → run the reflection LLM, write `reflection_notes`.

**LLM calls**: 0 or 1.

**Next node**: Determined by `reflect_router`:

| Condition | Route |
|-----------|-------|
| Any pending node exists whose dependencies are all satisfied and has not been dispatched | `dispatcher` |
| No pending nodes remain | `planner` |

**Invariants**:
- `reflection_notes` is overwritten (not appended) each time reflect writes it
- Reflect never modifies `nodes`, `turns`, `seed_context`, or `final_answer`
- Reflect never creates or deletes nodes
- The reflect node is always reached from a searcher (unconditional edge), never directly from planner or dispatcher

---

### 2.6 `finalize`

**Purpose**: Synthesise all completed research into a final report.

**Runs**: Exactly once. Terminal node.

**Input state read**:
- `question` — original query
- `nodes` — all completed nodes (grouped by category)

**Output state written**:
- `final_answer: str` — the full research report

**LLM calls**: 1 (synthesis) + optionally up to 5 (one per category, if compression enabled and findings are large).

**Next node**: Always `END`. Execution terminates.

**Invariants**:
- After finalize, `final_answer` is a non-empty string (unless LLM fails, in which case exception)
- Finalize never modifies `nodes`, `turns`, `seed_context`, or `reflection_notes`
- Finalize is reached from `route_after_planner` only — never from reflect, dispatcher, or searcher

---

## 3. State Fields Reference

### Complete `MindSearchState` schema

| Field | Type | Merge Strategy | Written By | Read By |
|-------|------|----------------|------------|---------|
| `question` | `str` | overwrite | initial state | planner, searcher, reflect, finalize |
| `nodes` | `list[SearchNode]` | `operator.add` (append) | planner (pending), searcher (done) | planner, dispatcher, reflect, finalize |
| `dispatched_ids` | `list[str]` | `operator.add` (append) | dispatcher | dispatcher |
| `to_dispatch_ids` | `list[str]` | overwrite | dispatcher | `_dispatch_sends` |
| `final_answer` | `str` | overwrite | finalize | app.py (SSE output) |
| `turns` | `int` | overwrite | planner | planner, `route_after_planner` |
| `seed_context` | `str` | overwrite | seed_searcher | planner |
| `reflection_notes` | `str` | overwrite | reflect | planner |

### `SearchNode` schema

| Field | Type | Set When Created (planner) | Set When Completed (searcher) |
|-------|------|---------------------------|-------------------------------|
| `id` | `str` | ✓ unique, e.g. "n1" | same (carried through) |
| `query` | `str` | ✓ concrete search string | same (carried through) |
| `depends_on` | `list[str]` | ✓ list of prerequisite node IDs | same (carried through) |
| `status` | `str` | `"pending"` | `"done"` |
| `summary` | `str` | `""` | FINDINGS section text |
| `leads` | `str` | `""` | LEADS section text |
| `category` | `str` | one of 5 categories | same (carried through) |
| `rationale` | `str` | one-sentence relevance | same (carried through) |

### Node lifecycle

```
CREATED (planner)          DISPATCHED (dispatcher)       COMPLETED (searcher)
status="pending"     →     (no status change;            status="done"
summary=""                  id added to                   summary=<findings>
leads=""                    dispatched_ids)               leads=<leads>
```

The `_dedup()` function resolves the lifecycle: both the pending and done versions
coexist in the `nodes` list (due to `operator.add`). `_dedup()` takes the last
occurrence of each ID, which is always the done version if the searcher has completed.

---

## 4. Execution Cycles

A "cycle" is one complete planning-and-search round. Here is the exact sequence
of node executions for different scenarios.

### 4.1 First Cycle (with seed search)

```
1. seed_searcher        writes: seed_context
2. planner (turn 0)     reads: question, seed_context, reflection_notes=""
                        writes: nodes=[n1,n2,n3,n4], turns=1
3. route_after_planner  has_pending=True → dispatcher
4. dispatcher           selects: n1,n2,n3 (n4 depends on n1)
                        writes: dispatched_ids=[n1,n2,n3], to_dispatch_ids=[n1,n2,n3]
5. _dispatch_sends      fan_out → Send(searcher, n1), Send(searcher, n2), Send(searcher, n3)
6. searcher[n1]  ┐
   searcher[n2]  ├ (parallel)    each writes: nodes=[done_node]
   searcher[n3]  ┘
7. reflect              (auto-join: fires once after n1+n2+n3 all done)
                        sees: n4 pending, deps satisfied → skip LLM
                        writes: nothing
8. reflect_router       pending=[n4] → dispatcher
9. dispatcher           selects: n4 (deps met, not yet dispatched)
                        writes: dispatched_ids=[n4], to_dispatch_ids=[n4]
10. _dispatch_sends     fan_out → Send(searcher, n4)
11. searcher[n4]        writes: nodes=[done_n4]
12. reflect             (auto-join: n4 done)
                        no pending nodes, 4 done nodes → run LLM reflection
                        writes: reflection_notes="..."
13. reflect_router      pending=[] → planner
```

### 4.2 Subsequent Cycle

```
14. planner (turn 1)    reads: question, seed_context, reflection_notes, graph_summary
                        writes: nodes=[n5,n6], turns=2
15. route_after_planner has_pending=True → dispatcher
16. dispatcher          selects: n5,n6
17. _dispatch_sends     fan_out → Send(searcher, n5), Send(searcher, n6)
18. searcher[n5] ┐
    searcher[n6] ┘     (parallel)
19. reflect            no pending → LLM reflection → writes reflection_notes
20. reflect_router     pending=[] → planner
```

### 4.3 Finalize Cycle

```
21. planner (turn 2)   LLM returns {"action": "finalize"}
                       writes: nodes=[], turns=3
22. route_after_planner has_pending=False → finalize
23. finalize           reads all done nodes, groups by category
                       writes: final_answer="..."
24. END
```

### 4.4 Budget Exhaustion

```
21. planner (turn 2)   existing_nodes=15 >= MAX_NODES
                       code forces: nodes=[], turns=3 (LLM never called)
22. route_after_planner has_pending=False → finalize
23. finalize → END
```

### 4.5 Turn Exhaustion

```
21. planner (turn 8)   turns=8 >= MAX_TURNS=8
                       code forces: nodes=[], turns=9 (LLM never called)
22. route_after_planner turns >= MAX_TURNS, has_pending=False → finalize
23. finalize → END
```

### 4.6 Empty Dispatch Fallback

This occurs when reflect routes to dispatcher but there's nothing new to dispatch:

```
12. reflect            sees: n4 pending, depends_on=[n1], n1 is done
                       → route to dispatcher
13. dispatcher         BUT n4 was already in dispatched_ids (dispatched in step 9)
                       → to_dispatch_ids = []
14. _dispatch_sends    to_dispatch_ids empty → return "planner"
15. planner (next turn)
```

This is valid but indicates a logic inconsistency between reflect's pending
detection (which doesn't check dispatched_ids) and dispatcher's selection
(which does). The fallback to planner is safe but wastes a cycle.

### 4.7 No Seed Search (enable_seed_search=False)

```
1. planner (turn 0)    seed_context="No landscape survey available."
                       (everything else same as 4.1 starting from step 2)
```

### 4.8 No Reflection (enable_reflection=False)

Reflect node is replaced with `lambda s: {}`. It still runs (it's on the
unconditional edge from searcher) but does nothing. Routing still works
because `reflect_router` reads the same state.

---

## 5. Failure Modes and Recovery

### 5.1 Planner LLM produces invalid JSON

**What happens**: `json.loads()` raises `JSONDecodeError`.

**Recovery**: Planner catches it, logs warning, returns `{"nodes": [], "turns": turns + 1}`.

**Effect**: `route_after_planner` sees no pending nodes → routes to `finalize`. Research ends with whatever nodes are already completed.

**How to detect in trace**: `llm_call` event for planner has `raw_response` that isn't valid JSON. Next `node_end` for planner has `notes` containing "forced finalize".

### 5.2 Planner produces valid JSON but missing/wrong fields

**What happens**: `raw_node.get("query")` may raise `KeyError` if `query` is missing.

**Recovery**: Currently NONE — this will crash. **This is a gap.** Should wrap node parsing in try/except per node.

**How to detect**: Exception in planner. `llm_call` event shows the raw response; manual inspection reveals the malformed node.

### 5.3 Searcher LLM call fails (connection error, timeout)

**What happens**: `llm_with_tools.invoke()` raises an exception.

**Recovery**: Currently NONE — exception propagates, crashes the graph. **This is a gap.** The searcher should catch, log, and return `summary="Search failed for: {query}"`.

**How to detect in trace**: `llm_call` event with `error` field. No subsequent `node_end` for that searcher.

### 5.4 Search tool fails (API error, rate limit)

**What happens**: `search_tool.invoke()` raises an exception.

**Recovery**: Currently NONE — exception propagates. Same gap as 5.3.

### 5.5 Reflection LLM call fails

**What happens**: Exception caught in try/except.

**Recovery**: `reflection_notes` set to `""`. Planner proceeds without reflection. Graceful degradation.

### 5.6 Finalize LLM call fails

**What happens**: Exception propagates, crashes the graph.

**Recovery**: Currently NONE. Research results are lost. **This is a gap.** Should catch and return `final_answer` as a raw dump of findings.

### 5.7 Infinite loop (planner keeps adding nodes without finalizing)

**What happens**: Planner generates new nodes every turn but never says finalize.

**Recovery**: `MAX_TURNS` hard stop forces finalize. `MAX_NODES` budget also forces finalize. Both are enforced in code before the LLM is called on each turn.

**How to detect in trace**: `budget_check` events show turns and node counts increasing. Planner `raw_response` shows it keeps emitting search actions.

### 5.8 Semantic dedup failure (planner generates near-duplicate queries)

**What happens**: Planner ignores the dedup rule and generates queries that return the same results as existing nodes. Wastes budget.

**Recovery**: Prompt-level only. Code-level dedup is by exact ID, not semantic. The planner can see all existing queries in `_graph_summary` and is instructed not to duplicate, but may ignore this.

**How to detect in trace**: Look at planner `llm_call` events — compare new node queries against existing ones in the `state_snapshot`. If they're semantically similar, the prompt needs strengthening.

### 5.9 Leads parser fails to split FINDINGS/LEADS

**What happens**: `_parse_searcher_output` doesn't find "LEADS" heading. Returns entire output as findings, empty string for leads.

**Recovery**: Graceful degradation. Summary contains everything, leads is empty. Planner still sees the full summary (truncated to 300ch) but loses structured lead visibility.

**How to detect in trace**: Searcher `node_end` output_delta shows `leads=""` even though the `llm_call` raw_response clearly contains lead-like content.

---

## 6. State Consistency Rules

These rules must hold at ALL times. If any are violated, there is a bug.

### 6.1 Node ID uniqueness
```
RULE: No two entries in _dedup(state["nodes"]) have the same ID.
```
`_dedup` is last-write-wins, so raw `state["nodes"]` CAN have duplicate IDs
(the pending and done versions of the same node). But after dedup, IDs are unique.

### 6.2 Node status transitions
```
RULE: A node's status can only transition: pending → done. Never done → pending.
      Never pending → pending. Never done → done.
```
Because `_dedup` takes the last occurrence, the done version (appended by searcher) always wins over the pending version (appended by planner). A node that has completed can never become pending again.

### 6.3 Turns monotonically increase
```
RULE: state["turns"] increases by exactly 1 each time planner runs.
      No other node modifies turns.
```

### 6.4 dispatched_ids only grows
```
RULE: Once an ID is in dispatched_ids, it stays there forever.
      dispatched_ids is append-only (operator.add).
```

### 6.5 Searcher output matches input
```
RULE: The node returned by a searcher has the same id, query, depends_on,
      category, and rationale as the SearcherState input.
      Only status, summary, and leads are changed.
```

### 6.6 Planner never modifies existing nodes
```
RULE: The planner's output contains only NEW nodes with status="pending".
      It never emits a node whose ID already exists in state.
      (The dedup suffix logic guarantees this.)
```

### 6.7 Final answer written exactly once
```
RULE: final_answer is "" until finalize runs, then it's set once.
      No other node writes final_answer.
```

### 6.8 seed_context written exactly once
```
RULE: seed_context is "" until seed_searcher runs, then it's set once.
      No other node writes seed_context.
      (seed_searcher runs exactly once.)
```

### 6.9 reflection_notes overwritten per cycle
```
RULE: reflection_notes is overwritten (not appended) each time reflect runs.
      The planner always sees the LATEST reflection, not a history.
```

### 6.10 Parallel searchers don't interfere
```
RULE: Parallel searchers each receive independent SearcherState.
      They each write to nodes via operator.add (append).
      They never read from or write to each other's state.
      The auto-join in LangGraph merges their outputs after ALL complete.
```

---

## 7. Concurrency and Ordering Guarantees

### 7.1 Within a single cycle
```
planner → dispatcher → [searchers in parallel] → reflect → planner
```
This ordering is guaranteed by the graph edges. Planner always finishes
before dispatcher starts. All searchers finish before reflect starts.

### 7.2 Parallel searchers
Multiple `Send("searcher", ...)` calls create truly parallel branches.
LangGraph provides these guarantees:
- All branches complete before the next node (reflect) fires
- State updates from all branches are merged (operator.add concatenates node lists)
- No ordering guarantee between parallel searchers (n1 may finish before or after n2)

### 7.3 operator.add merge semantics
When parallel searchers complete, their `nodes` outputs are concatenated in
unspecified order. This is fine because `_dedup()` uses last-write-wins by ID,
and each searcher writes a unique ID.

### 7.4 No cross-cycle state leakage
Each cycle's dispatcher writes fresh `to_dispatch_ids`. The previous cycle's
`to_dispatch_ids` are overwritten. `dispatched_ids` accumulates across cycles
(append-only) to prevent re-dispatch.

---

## 8. Debugging Decision Tree

Use this when something goes wrong.

### "The graph never produces any search results"

```
1. Check: Did planner run?
   → Look for planner node_start/node_end in trace
   → If no: check seed_searcher — did it crash?

2. Check: Did planner emit nodes?
   → Look at planner node_end output_delta
   → If nodes=[] and turns=1: planner said finalize on first turn,
     or JSON parse failed, or budget was 0
   → Check planner llm_call raw_response

3. Check: Did route_after_planner go to dispatcher?
   → Look at route decision event
   → If finalize: planner didn't emit pending nodes

4. Check: Did dispatcher find anything to dispatch?
   → Look at dispatcher node_end
   → If to_dispatch_ids=[]: all pending nodes have unmet deps,
     or all were already dispatched
```

### "Research is too narrow / not exploring"

```
1. Check: Did seed_searcher run and produce context?
   → Look at seed_searcher node_end output_delta.seed_context
   → If empty: seed search failed, planner has no landscape

2. Check: What did the planner see?
   → Find planner llm_call user_prompt
   → Verify seed_context and reflection_notes are present
   → If missing: state wiring bug

3. Check: What categories did the planner emit?
   → Find planner llm_call raw_response
   → Look at category fields
   → If all "core": planner is ignoring exploration instructions

4. Check: Is reflection producing useful feedback?
   → Find reflect llm_call raw_response
   → If empty or generic: reflection prompt needs improvement
   → If specific but planner ignores: planner prompt needs strengthening
```

### "Leads are not being followed up"

```
1. Check: Did the searcher output LEADS?
   → Find searcher llm_call raw_response — does it contain "LEADS:" heading?
   → If no: searcher prompt not being followed

2. Check: Were leads parsed correctly?
   → Find searcher node_end output_delta
   → Is leads field populated, or is it ""?
   → If "": _parse_searcher_output failed to split

3. Check: Can the planner see the leads?
   → Find planner llm_call user_prompt
   → Search for "Leads:" in the graph_summary portion
   → If not visible: _graph_summary truncation or formatting bug

4. Check: Is the planner choosing to act on leads?
   → Find planner llm_call raw_response
   → If it adds nodes matching leads: working
   → If it ignores them: planner prompt needs to emphasise lead follow-up
```

### "The graph is running too long / too many nodes"

```
1. Check budget_check events in trace
   → Are total_nodes approaching MAX_NODES?
   → Are turns approaching MAX_TURNS?

2. Check: Is the planner emitting large batches?
   → Count nodes per planner turn from output_delta

3. Check: Is semantic dedup failing?
   → Compare queries across planner turns — are they similar?

4. Check: Is reflection encouraging more research than needed?
   → Read reflection_notes — is it identifying real gaps or being too demanding?
```

### "Reflect is routing to dispatcher but nothing happens"

```
This is the empty-dispatch fallback (Section 4.6).

1. Reflect sees pending nodes → routes to dispatcher
2. Dispatcher checks dispatched_ids → those nodes were already dispatched
3. Dispatcher writes to_dispatch_ids=[] → _dispatch_sends falls back to planner
4. Net effect: wasted cycle, but no harm

Root cause: reflect doesn't check dispatched_ids, only pending status.
This is by design — the fallback is safe. But if you see it happening
repeatedly, consider adding dispatched_ids awareness to reflect_router.
```

---

## 9. Quick Reference: "Who writes what"

| State Field | seed_searcher | planner | dispatcher | searcher | reflect | finalize |
|-------------|:---:|:---:|:---:|:---:|:---:|:---:|
| `question` | — | — | — | — | — | — |
| `nodes` | — | ✓ (pending) | — | ✓ (done) | — | — |
| `dispatched_ids` | — | — | ✓ | — | — | — |
| `to_dispatch_ids` | — | — | ✓ | — | — | — |
| `turns` | — | ✓ | — | — | — | — |
| `seed_context` | ✓ | — | — | — | — | — |
| `reflection_notes` | — | — | — | — | ✓ | — |
| `final_answer` | — | — | — | — | — | ✓ |

No field is written by more than two nodes. `nodes` is the only field written by
two different nodes (planner and searcher), and they write different statuses
(pending vs done) with non-overlapping IDs.
