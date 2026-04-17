# LangMindSearch: Explicit Research Planning — Architectural Plan

## Overview

Transform MindSearch from an implicit-plan system (planner decides what to
search based on reasoning context) into an explicit-plan system where:

1. A seed search establishes topic background
2. The system generates a structured research plan and shows it to the user
3. The user can amend the plan (add/remove/reorder items) via conversation
4. Once confirmed, plan stages execute as a DAG — independent stages run in
   parallel, dependent stages wait for prerequisites
5. Each stage runs as an independent MindSearch sub-research
6. Stage summaries are synthesised into a final coherent narrative

---

## Two-Level Graph Architecture

The plan itself is a LangGraph — not a JSON document sitting in state.
Plan stages are nodes in an **outer graph** with edges defined by `depends_on`.
Each outer node wraps an **inner graph** (the existing MindSearch
planner→dispatcher→searcher→reflect cycle).

```
OUTER GRAPH (plan-level, built dynamically from confirmed plan):

  seed_searcher
       │
       ▼
  plan_generator ◄──── plan_amender (loop until confirmed)
       │
       ▼
  stage_s1 ──────┐
  stage_s2 ──────┤
  stage_s3 ──────┼──► stage_s7 (depends_on: [s1, s2]) ──┐
  stage_s4 ──────┘                                       │
  stage_s5 ──────────────────────────────────────────────┼──► synthesise → END
  stage_s6 ──────────────────────────────────────────────┘


INNER GRAPH (per-stage, existing MindSearch architecture):

  planner → dispatcher → searcher(×N) → reflect → planner → ... → finalize
```

Independent stages (empty `depends_on`) dispatch in parallel via LangGraph's
`Send()` mechanism — the same pattern the inner graph already uses for parallel
searchers. Dependent stages wait for all prerequisites to complete before
starting. The outer graph's dispatcher handles this scheduling.

A stage only receives summaries from the stages it explicitly depends on,
NOT from all prior stages. This prevents irrelevant context from leaking
into independent research tracks. The final synthesis step receives ALL
stage summaries.

---

## Why the Plan Is a LangGraph, Not a JSON Document

Storing the plan as a JSON blob in state creates two representations that can
diverge: the JSON plan and the code that interprets it. Making the plan a
compiled LangGraph eliminates the translation layer:

- **Scheduling is native**: LangGraph handles parallel dispatch and dependency
  waiting through its existing `Send()` and conditional edge mechanisms.
- **Tracing is unified**: The tracer captures both outer stage transitions and
  inner search node transitions as a nested trace. You see
  "stage s2 started → [inner: planner t0, searcher n1, n2, reflect, finalize]
  → stage s2 done" in a single timeline.
- **Amendments modify the graph**: Adding a stage adds a node. Removing one
  removes a node and updates edges. The graph is recompiled after each
  amendment — no JSON-to-execution translation needed.
- **Resumability**: LangGraph's checkpointing can persist the outer graph
  state, allowing interrupted research to resume at the last completed stage.

---

## Phase 1: Research Plan Data Model

### Types

```python
class PlanStage(TypedDict):
    id: str              # "s1", "s2", ...
    title: str           # "Fundamentals of Speculative Decoding for MoE"
    description: str     # 2-3 sentences describing what to research
    search_hints: list[str]  # suggested search queries (guidance, not mandatory)
    depends_on: list[str]    # stage IDs that must complete first
    status: str          # "pending" | "active" | "done"
    summary: str         # rich output after stage completes


class OuterState(TypedDict):
    query: str
    seed_context: str
    stages: Annotated[list[PlanStage], operator.add]  # append-only, like inner nodes
    completed_stage_ids: Annotated[list[str], operator.add]
    final_report: str
    plan_confirmed: bool
    messages: list[dict]  # conversation history for plan refinement


class StageExecState(TypedDict):
    """Input to each stage node in the outer graph."""
    stage_id: str
    stage_title: str
    stage_description: str
    search_hints: list[str]
    query: str                    # original user question (for context)
    dependency_summaries: str     # summaries from depends_on stages ONLY
```

### Dependency-scoped context forwarding

When a stage node starts, it receives summaries ONLY from stages listed in
its `depends_on`. This is critical for preventing context pollution:

```python
def _gather_dependency_summaries(stage: PlanStage, all_stages: list[PlanStage]) -> str:
    """Collect summaries only from stages this stage depends on."""
    dep_ids = set(stage.get("depends_on", []))
    if not dep_ids:
        return "This is an independent stage. No prior context."

    summaries = []
    for s in all_stages:
        if s["id"] in dep_ids and s["status"] == "done" and s.get("summary"):
            summaries.append(f"## {s['title']}\n{s['summary']}")
    return "\n\n".join(summaries) if summaries else "Dependencies completed but produced no findings."
```

---

## Phase 2: Plan Generation

### Prompt: `PLAN_GENERATOR_SYSTEM`

```python
PLAN_GENERATOR_SYSTEM = f"""{_DATE}

You are a research planning agent. Given a user's research question and a
landscape survey of the topic, generate a structured research plan as a DAG
of stages.

PLAN STRUCTURE:
- 5-10 stages, each focused on a distinct sub-topic
- Each stage must be specific enough to research independently
- Order stages from foundational to specific where there's a real dependency
- Cover the user's explicit questions AND important adjacent territory

DEPENDENCY RULES (critical):
- Most stages should be independent: depends_on = []
- ONLY add a dependency when a stage genuinely cannot be researched without
  findings from another stage (e.g. a "compare approaches" stage that needs
  the individual approach stages to complete first)
- Do NOT create a linear chain where each stage depends on the previous one
- Independent stages will run in parallel for speed

For each stage provide:
- id: sequential string ("s1", "s2", ...)
- title: clear, specific title
- description: 2-3 sentences on what to research and why
- search_hints: 2-3 suggested search queries as starting points
- depends_on: list of stage IDs (empty for independent stages)

Do NOT include stages for:
- Executive summaries or introductions (that's the synthesis step)
- Conclusions or "future work" (unless the user specifically asked)
- Meta-commentary about the research process

Output format (JSON only, no explanation):

{{
  "stages": [
    {{
      "id": "s1",
      "title": "Fundamentals of Speculative Decoding for MoE Models",
      "description": "Establish the theoretical basis for speculative decoding
        in sparse models, including rejection sampling mechanics and why
        memory bandwidth is the primary bottleneck.",
      "search_hints": [
        "speculative decoding MoE models memory bandwidth bottleneck",
        "rejection sampling speculative decoding acceptance rate"
      ],
      "depends_on": []
    }},
    {{
      "id": "s2",
      "title": "DeepSeek Draft Model Architecture and Training",
      "description": "Investigate how DeepSeek created their draft models,
        including V2-Lite architecture choices and distillation approach.",
      "search_hints": [
        "DeepSeek V2 Lite draft model architecture",
        "DeepSeek speculative decoding training data"
      ],
      "depends_on": []
    }},
    {{
      "id": "s3",
      "title": "MoE Replica vs Dense Distillation Trade-offs",
      "description": "Compare the two main approaches: building a smaller MoE
        that mirrors the master architecture vs distilling into a dense model.
        Requires understanding of both the master architecture (s1) and
        existing implementations (s2).",
      "search_hints": [
        "MoE draft model vs dense draft speculative decoding",
        "acceptance rate MoE replica vs dense speculator"
      ],
      "depends_on": ["s1", "s2"]
    }}
  ]
}}
"""
```

Note: the example shows s1 and s2 as independent (both `depends_on: []`) and
s3 depending on both. s1 and s2 will run in parallel; s3 waits for both.

### Prompt: `PLAN_GENERATOR_USER_TEMPLATE`

```python
PLAN_GENERATOR_USER_TEMPLATE = """Research question:
{query}

Landscape survey (from preliminary search):
{seed_context}

Generate a structured research plan of 5-10 stages as a DAG.
Most stages should be independent (depends_on: []).
Only add dependencies where genuinely needed."""
```

---

## Phase 3: Plan Presentation and Amendment

### SSE event for plan display

```python
evt = {
    "type": "research_plan",
    "plan": {
        "stages": [
            {
                "id": "s1",
                "title": "...",
                "description": "...",
                "search_hints": [...],
                "depends_on": []
            },
            ...
        ]
    },
    "awaiting_confirmation": True
}
```

The frontend renders the plan showing:
- Stage titles and descriptions
- Dependency relationships (which stages feed into which)
- A "Start Research" confirmation button
- A text input for amendment requests

User can:
- Confirm ("looks good" / click confirm)
- Add ("also research X")
- Remove ("drop stage 5")
- Reorder / change dependencies ("make stage 3 independent of stage 1")
- Modify ("change stage 4 to focus more on Y")

### Prompt: `PLAN_AMENDER_SYSTEM`

```python
PLAN_AMENDER_SYSTEM = f"""{_DATE}

You are a research plan editor. You will receive an existing research plan
and a user's amendment request. Modify the plan accordingly.

You can:
- Add new stages (assign new sequential IDs)
- Remove stages (remove them and clean up depends_on references)
- Change dependencies (make stages independent or add new dependencies)
- Modify stage titles, descriptions, or search hints

RULES:
- Preserve stages the user didn't mention
- When removing a stage, also remove it from other stages' depends_on lists
- Keep most stages independent (depends_on: []) unless there's a real reason
- Maintain logical ordering where dependencies exist

Output the complete amended plan in the same JSON format.
"""
```

### Confirmation classifier

```python
PLAN_CONFIRM_CHECK = """The user responded to a research plan with:
"{user_message}"

Is the user:
A) Confirming the plan (wants to proceed with research)
B) Requesting changes to the plan

Respond with only "A" or "B"."""
```

Runs on fast LLM. "A" → compile outer graph and execute. "B" → pass to amender.

---

## Phase 4: Outer Graph Compilation

### Build the outer graph dynamically from the confirmed plan

```python
def build_outer_graph(
    plan_stages: list[PlanStage],
    llm_url: str,
    search_engine: str,
    **inner_config,
) -> CompiledGraph:
    """Build the outer DAG from the confirmed research plan.

    Each stage becomes a node. Edges are defined by depends_on.
    Independent stages dispatch in parallel via Send().
    """
    graph = StateGraph(OuterState)

    # Add a node for each stage
    for stage in plan_stages:
        stage_fn = make_stage_node(
            stage=stage,
            llm_url=llm_url,
            search_engine=search_engine,
            **inner_config,
        )
        graph.add_node(f"stage_{stage['id']}", stage_fn)

    # Add the synthesis node
    graph.add_node("synthesise", make_synthesis_node(reason_llm))

    # Add the stage dispatcher (schedules unblocked stages)
    graph.add_node("stage_dispatcher", stage_dispatcher_node)

    # Entry point
    graph.set_entry_point("stage_dispatcher")

    # Dispatcher fans out to ready stages or routes to synthesis
    graph.add_conditional_edges(
        "stage_dispatcher",
        stage_dispatch_router,
        [f"stage_{s['id']}" for s in plan_stages] + ["synthesise"],
    )

    # Every stage routes back to dispatcher after completion
    for stage in plan_stages:
        graph.add_edge(f"stage_{stage['id']}", "stage_dispatcher")

    graph.add_edge("synthesise", END)

    return graph.compile()
```

### Stage dispatcher (outer-level equivalent of inner dispatcher)

```python
def stage_dispatcher_node(state: OuterState) -> dict:
    """Select which plan stages are ready to run.

    A stage is ready if:
    - status == "pending"
    - all depends_on stages are in completed_stage_ids
    - it hasn't been dispatched yet
    """
    completed = set(state.get("completed_stage_ids", []))
    all_stages = _dedup_stages(state["stages"])

    ready = [
        s for s in all_stages
        if s["status"] == "pending"
        and all(dep in completed for dep in s.get("depends_on", []))
    ]

    return {"ready_stage_ids": [s["id"] for s in ready]}


def stage_dispatch_router(state: OuterState):
    """Fan out to ready stages, or route to synthesis if all done."""
    ready_ids = state.get("ready_stage_ids", [])

    if not ready_ids:
        # Check if all stages are done
        all_stages = _dedup_stages(state["stages"])
        if all(s["status"] == "done" for s in all_stages):
            return "synthesise"
        # Some stages pending but blocked — shouldn't happen if plan is valid
        log.warning("OUTER_DISPATCHER: no ready stages but not all done")
        return "synthesise"  # fail-safe

    all_stages = _dedup_stages(state["stages"])
    return [
        Send(f"stage_{sid}", StageExecState(
            stage_id=sid,
            stage_title=stage["title"],
            stage_description=stage["description"],
            search_hints=stage.get("search_hints", []),
            query=state["query"],
            dependency_summaries=_gather_dependency_summaries(stage, all_stages),
        ))
        for sid in ready_ids
        for stage in all_stages if stage["id"] == sid
    ]
```

### Stage node (wraps inner MindSearch)

```python
def make_stage_node(stage: PlanStage, llm_url: str, search_engine: str, **config):
    """Create an outer graph node that runs a full inner MindSearch for one stage."""

    def stage_node(state: StageExecState) -> dict:
        log.info("STAGE [%s] starting: %s", state["stage_id"], state["stage_title"])

        # Build stage query from template
        stage_query = STAGE_QUERY_TEMPLATE.format(
            stage_title=state["stage_title"],
            stage_description=state["stage_description"],
            search_hints="\n".join(f"- {h}" for h in state.get("search_hints", [])),
            dependency_context=state.get("dependency_summaries", ""),
        )

        # Build and run a fresh inner MindSearch graph
        inner_graph, tracer = build_graph(
            llm_url=llm_url,
            search_engine=search_engine,
            max_turns=4,
            max_nodes=8,
            enable_seed_search=False,  # seed already done at plan level
            enable_reflection=True,
            **config,
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

        # Run inner graph to completion (synchronous within the stage node)
        final_state = inner_graph.invoke(
            inner_initial,
            config={"recursion_limit": 100},
        )

        summary = final_state.get("final_answer", "No findings.")
        log.info("STAGE [%s] done, summary length: %d", state["stage_id"], len(summary))

        # Return completed stage — appended to outer state via operator.add
        return {
            "stages": [PlanStage(
                id=state["stage_id"],
                title=state["stage_title"],
                description=state["stage_description"],
                search_hints=state.get("search_hints", []),
                depends_on=[],  # not needed after execution
                status="done",
                summary=summary,
            )],
            "completed_stage_ids": [state["stage_id"]],
        }

    return stage_node
```

### Stage query template

```python
STAGE_QUERY_TEMPLATE = """Research stage: {stage_title}

Objective: {stage_description}

Suggested search directions:
{search_hints}

Context from prerequisite stages (only stages this one depends on):
{dependency_context}

Research this stage thoroughly. Include all specific numbers, benchmarks,
technical details, and source URLs you find. This summary will be used
by a later synthesis step to build a comprehensive report, so preserve
all concrete details — do not generalise."""
```

### Stage-level finalize prompt

The inner MindSearch finalize for each stage produces dense research notes,
not a polished report:

```python
STAGE_FINALIZE_SYSTEM = f"""{_DATE}

You are a research summariser for one stage of a multi-stage research project.
Write a detailed summary of all findings for this stage.

CRITICAL: This summary will be read by another agent who will synthesise
multiple stage summaries into a final report. You must preserve:
- ALL specific numbers, benchmarks, percentages, and metrics
- ALL named entities (people, companies, products, papers)
- ALL source URLs
- Technical details and specific claims

Do NOT write a polished report. Write a dense, fact-packed summary that
preserves maximum information. Think of this as detailed research notes,
not a finished product. Length is not a concern — err on the side of
including too much detail rather than too little.
"""
```

---

## Phase 5: Final Synthesis (Narrative Report)

After all stages complete, the synthesis node receives ALL stage summaries
and produces the coherent narrative.

```python
NARRATIVE_SYNTHESIS_SYSTEM = f"""{_DATE}

You are a research report writer. You will receive detailed summaries from
multiple research stages, each covering a different aspect of the topic.
Your job is to weave these into a single, coherent narrative report.

STRUCTURE:
- Use the stage titles as a guide for section organisation, but feel free
  to merge, split, or reorder sections if it improves the narrative flow
- Build a logical progression: foundational concepts → specific findings
  → practical implications → technical details
- Cross-reference between sections where findings from one stage inform
  another
- Resolve any contradictions between stages by noting both viewpoints

CONTENT:
- Include all specific numbers, benchmarks, and technical details from
  the stage summaries
- Cite sources with URLs where provided
- Do not add information not present in the stage summaries
- Do not add an executive summary, conclusion, or "future work" section
  unless the research plan explicitly included one

STYLE:
- Professional, substantive tone
- Dense with facts — avoid filler sentences
- Use markdown with clear headings
- Use tables for comparative data where appropriate
"""

NARRATIVE_SYNTHESIS_USER = """Original research question:
{query}

Research plan that was executed:
{plan_outline}

Stage summaries (from all completed stages):
{all_stage_summaries}

Write a comprehensive narrative report synthesising all findings."""
```

---

## Phase 6: App Integration

### API flow

```
POST /solve  {inputs: "research question"}
  → SSE: seed_search progress
  → SSE: {type: "research_plan", plan: {...}, awaiting_confirmation: true}

POST /solve  {inputs: "add a stage about X", plan_id: "..."}
  → SSE: {type: "research_plan", plan: {amended...}, awaiting_confirmation: true}

POST /solve  {inputs: "looks good", plan_id: "...", confirm: true}
  → SSE: {type: "stage_started", stage_id: "s1"}
  → SSE: inner MindSearch events for s1 (planner, searcher, reflect...)
  → SSE: {type: "stage_complete", stage_id: "s1", summary: "..."}
  → SSE: {type: "stage_started", stage_id: "s2"}  ← may overlap with s1 if independent
  → ...
  → SSE: {type: "synthesis_started"}
  → SSE: {type: "final_report", content: "..."}
```

Note: independent stages may run concurrently, so stage_started events for
s1 and s2 may interleave. The frontend should handle this — show multiple
stages as "active" simultaneously.

### Frontend changes

1. **Plan display**: Show stages as a DAG/tree, not a linear list. Independent
   stages shown side-by-side; dependent stages shown below their prerequisites.
2. **Confirm button**: "Start Research"
3. **Amendment input**: Text field for modification requests
4. **Stage progress**: During execution, show active/completed/pending states.
   Multiple stages may be active simultaneously.
5. **Expandable stage cards**: Click to see description, search hints, and
   (after completion) the stage summary
6. **Dependency visualization**: Show which stages feed into which (arrows or
   indentation)

### State persistence

The plan and stage summaries are persisted in history_db so that:
- The user can leave and come back
- Research can be resumed if interrupted mid-execution
- Follow-up questions can reference the plan structure

---

## Phase 7: Backward Compatibility

### Quick mode (no plan)

For simple questions, preserve the current direct-execution path:

```python
NEEDS_PLAN_CHECK = """Given this research question:
"{query}"

Does this require a structured multi-stage research plan (5+ distinct
sub-topics to investigate), or can it be answered with a single focused
research run (1-3 related sub-topics)?

Respond with only "PLAN" or "DIRECT"."""
```

- "PLAN" → plan-first flow (this document)
- "DIRECT" → current MindSearch flow (seed → search → synthesise)

A "Quick" button in the frontend forces DIRECT mode regardless.

---

## Implementation Order

### Step 1: Data model and plan generation
- Add `PlanStage`, `OuterState`, `StageExecState` types
- Add `PLAN_GENERATOR_SYSTEM` prompt with DAG dependency rules
- Implement plan generation from seed context + query
- Test: generate plans, verify most stages are independent, dependencies are sparse

### Step 2: Plan presentation and amendment (backend)
- New SSE event types for plan display and confirmation
- Implement `PLAN_CONFIRM_CHECK` classifier
- Implement `PLAN_AMENDER_SYSTEM` for plan modifications
- Test: amend plans, verify depends_on is correctly maintained

### Step 3: Outer graph compiler
- Implement `build_outer_graph` that compiles confirmed plan into LangGraph
- Implement `stage_dispatcher_node` and `stage_dispatch_router`
- Implement `make_stage_node` wrapping inner MindSearch
- Test: compile a 5-stage plan with 2 dependencies, verify parallel dispatch
  of independent stages and waiting on dependent stages

### Step 4: Stage execution
- Implement `STAGE_QUERY_TEMPLATE` and `STAGE_FINALIZE_SYSTEM`
- Implement `_gather_dependency_summaries` for scoped context forwarding
- SSE events for per-stage progress
- Test: run 3 independent stages in parallel, verify no context leakage

### Step 5: Narrative synthesis
- Implement `NARRATIVE_SYNTHESIS_SYSTEM` prompt
- Wire all stage summaries into synthesis
- Test: verify final report has narrative flow and cross-references

### Step 6: Frontend plan UI
- Plan display as DAG (not linear list)
- Confirm/amend UI
- Multi-stage parallel progress indicators
- Expandable stage cards

### Step 7: Quick/Plan mode selector
- Add query classifier
- Wire both paths in the app
- "Quick" button forces direct mode

---

## Budget and Timing Estimates

### With parallelism (independent stages run concurrently)

For a typical plan with 8 stages, 5 independent + 3 dependent:

| Phase | Wall Time | Notes |
|-------|-----------|-------|
| Seed search | ~10s | |
| Plan generation | ~5s | |
| Plan amendments | ~5-15s | 1-3 LLM calls |
| Independent stages (5, parallel) | ~90s | limited by slowest of 5 |
| Dependent stages (3, sequential) | ~180s | each ~60s, sequential |
| Narrative synthesis | ~20-30s | |
| **Total** | **~5-6 min** | vs ~9-14 min if fully sequential |

Parallelism cuts total time significantly. The bottleneck is the dependent
stages, which must wait for their prerequisites.

### Resource usage per stage

| Resource | Per Stage | Total (8 stages) |
|----------|-----------|-------------------|
| Search nodes | 6-8 | ~56 |
| LLM calls | 10-15 | ~100 |
| Search API calls | 6-12 | ~72 |

### Concurrency on llama-server

With 5 independent stages running in parallel, each running its own inner
MindSearch with up to 3 concurrent searchers, the peak concurrent LLM
requests could be 15. This exceeds the typical `-np 3` slot configuration.

Options:
- **Sequential outer, parallel inner** (conservative): Run stages one at a
  time but allow parallel searchers within each. Peak concurrency = 3. Safe
  default. Total time ~9-14 min.
- **Parallel outer, sequential inner** (balanced): Run independent stages in
  parallel but limit each stage to 1 searcher. Peak concurrency = 5.
  Total time ~5-6 min.
- **Full parallel** (aggressive): Both levels parallel. Requires `-np 8+` on
  llama-server. Fastest but needs more VRAM for KV cache slots.

The outer graph should respect a `MAX_CONCURRENT_STAGES` config (default 2-3)
that limits how many stages dispatch simultaneously.

---

## Key Differences from Current Architecture

| Aspect | Current | Proposed |
|--------|---------|----------|
| Plan visibility | Implicit (inside planner reasoning) | Explicit (shown to user as DAG) |
| User control | None (submit query, wait) | Full (amend plan before execution) |
| Plan representation | N/A | LangGraph (compiled from plan JSON) |
| Scope control | Node budget + prompt hints | User-approved stage list |
| Stage isolation | None (single shared context) | Full (each stage is independent graph) |
| Context forwarding | All prior results visible | Only declared dependencies |
| Stage parallelism | N/A | Independent stages run concurrently |
| Report structure | Emergent (LLM decides) | Determined by plan stages |
| Total search depth | 15 nodes | ~56 nodes across 8 stages |
| Filler sections | Prompt must suppress them | Can't appear (not in plan) |
| Tracing | Single-level | Nested (outer stage + inner search) |
