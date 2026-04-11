<div id="top"></div>

<div align="center">

<picture>
  <source srcset="assets/logo.svg" media="(prefers-color-scheme: light)">
  <source srcset="assets/logo-darkmode.svg" media="(prefers-color-scheme: dark)">
  <img src="assets/logo.svg" alt="Logo" width="50%">
</picture>

**LangMindSearch** — Exploratory deep research on local LLMs

[📃 Original Paper](https://arxiv.org/abs/2407.20183)

</div>

## What is LangMindSearch?

LangMindSearch is a fork of [MindSearch](https://github.com/InternLM/MindSearch) rebuilt around two ideas: **exploratory research** and **local-first inference**.

The original MindSearch decomposes a question into sub-queries and searches them — a "decompose and answer" approach. LangMindSearch goes further: it maps the topic space around your question, discovers what you didn't know to ask, and produces a report that covers the territory, not just the literal query.

The entire backend runs on local LLMs via [llama.cpp](https://github.com/ggml-org/llama.cpp), with no dependency on cloud APIs for inference.

### What changed from upstream MindSearch

The original MindSearch used a Python code-generation planner (the LLM writes Python to build a search graph) backed by the Lagent framework. LangMindSearch replaces this entirely:

- **LangGraph-based DAG planner** — the planner emits JSON describing search nodes with dependencies, categories, and relevance rationales. No code generation, no Lagent dependency.
- **Exploratory research prompts** — the system is prompted to explore beyond the literal question: adjacent topics, historical context, emerging trends, controversies, and limitations. A node budget and category targets keep exploration disciplined.
- **Seed search** — before planning, a broad web survey discovers the topic landscape, giving the planner real-world context about what exists.
- **Reflection loop** — after each search batch, an LLM-powered reflection step assesses coverage gaps and feeds actionable guidance back to the planner.
- **Structured leads** — each searcher outputs prioritised leads (topics discovered during search that weren't the focus), which the planner can follow up on.
- **Concurrency cap** — the dispatcher limits parallel searchers to avoid overwhelming a single local LLM endpoint.
- **Execution tracer** — a structured JSON trace captures every state transition, LLM call, and routing decision for debugging.

### Architecture

```
                 seed_searcher
                      │
                      ▼
         ┌──── planner ◄──────────────┐
         │         │                  │
    has_pending  finalize → END       │
         │                            │
         ▼                            │
    dispatcher                        │
      │      │                        │
   fan_out  (empty → planner)         │
      │                               │
      ▼                               │
  searcher (×N parallel, capped)      │
      │                               │
      ▼                               │
   reflect ───────────────────────────┘
     (writes reflection_notes to state,
      routes to dispatcher or planner)
```

The planner builds a DAG of search nodes with five categories: **core**, **context**, **adjacent**, **emerging**, and **critical**. Independent nodes run in parallel; dependent nodes wait for their prerequisites. The system targets comprehensive topic coverage, not just answering the question.

## Quick Start

### Prerequisites

- Python 3.10+
- A running [llama-server](https://github.com/ggml-org/llama.cpp) instance with an OpenAI-compatible API
- Node.js 18+ (for the React frontend)

### 1. Install dependencies

```bash
git clone https://github.com/ggl314/LangMindSearch
cd LangMindSearch
pip install -r requirements.txt
```

### 2. Start llama-server

LangMindSearch is designed for local inference. Start llama-server with parallel slots and prompt caching for best performance:

```bash
llama-server \
  -m your-model.gguf \
  -ngl 99 \
  -fa \
  -c 32768 \
  -np 3 \
  --cont-batching \
  -cram 512 \
  -sps 0.3 \
  --cache-reuse 128
```

Key flags:
- `-np 3` — three parallel slots, matching the default searcher concurrency cap
- `-cram 512` — 512 MiB host RAM for prompt caching (the planner and searcher system prompts are reused across calls)
- `-sps 0.3` — lower similarity threshold for slot matching (searchers share a system prompt but differ in user content)
- `--cache-reuse 128` — minimum chunk size for KV shift-based reuse

### 3. Start the backend

```bash
python -m mindsearch.app --llm_url http://localhost:8080/v1
```

Options:
- `--llm_url` — llama-server endpoint (default: `http://localhost:8080/v1`, or set `LLM_URL` env var)
- `--search_engine` — `DuckDuckGoSearch` (default, no API key), `GoogleSearch` (requires `SERPER_API_KEY`), or `BraveSearch` (requires `BRAVE_API_KEY`)

Environment variables for tuning:
- `MAX_TURNS=8` — maximum planner iterations before forcing finalize
- `MAX_NODES=15` — total node budget (exploration cap)
- `MAX_CONCURRENT_SEARCHERS=3` — parallel searcher limit
- `ENABLE_SEED_SEARCH=true` — run a landscape survey before planning
- `ENABLE_REFLECTION=true` — LLM-powered coverage assessment between rounds
- `DEBUG=true` — enable execution tracing (writes JSON traces to `TRACE_DIR`)
- `TRACE_DIR=./traces` — output directory for trace files

### 4. Start the frontend

```bash
cd frontend/React
npm install
npm start
```

The frontend defaults to `http://localhost:5173`. Configure the backend proxy in `vite.config.ts` if the backend is on a different host/port.

### 5. Terminal mode (no frontend)

For debugging or scripting:

```bash
python -m mindsearch.terminal \
  --llm_url http://localhost:8080/v1 \
  --query "your research question here"
```

Add `--debug` to enable tracing with a printed execution summary.

## How It Works

A research run proceeds through several cycles:

1. **Seed search** — 2-3 broad web searches discover the topic landscape (key players, sub-areas, terminology, recent developments).

2. **Planning** — the planner reads the landscape survey, any prior reflection feedback, and the current research graph. It emits a batch of search nodes, each tagged with a category and a one-sentence relevance rationale. Each batch must include at least one non-core node.

3. **Dispatching** — the dispatcher selects unblocked nodes (dependencies satisfied, not already dispatched) and fans them out to parallel searchers, capped at `MAX_CONCURRENT_SEARCHERS`.

4. **Searching** — each searcher runs a ReAct loop (up to 3 rounds of tool calls), then outputs structured FINDINGS and prioritised LEADS.

5. **Reflection** — after all searchers in a batch complete, the reflection node assesses coverage, identifies gaps, and writes actionable guidance into the state for the planner's next turn.

6. **Repeat** until the planner signals finalize, or the node budget / turn limit is reached.

7. **Synthesis** — the finalize node groups all completed findings by category and generates a comprehensive report with sections for core answer, background, related developments, limitations, and areas for further investigation.

## Debugging

### Execution traces

Run with `DEBUG=true` (or `--debug` in terminal mode). The tracer records every node execution, LLM call (with full prompts and responses), tool invocation, routing decision, and budget check. Traces are saved as JSON to `TRACE_DIR`.

```bash
# Pretty-print route decisions from a trace
cat traces/trace_*.json | python3 -c "
import json, sys
t = json.load(sys.stdin)
for e in t['events']:
    if e['type'] == 'route':
        print(f\"{e['elapsed']:7.1f}s  {e['router']} → {e['decision']}\")
"
```

### Logs

The backend logs every state transition at `INFO` level with prefixed node names (`PLANNER`, `SEARCHER [n1]`, `DISPATCHER`, `REFLECT`, etc.). Key things to watch:

- `PLANNER emitting N new nodes` — how many nodes per turn
- `DISPATCHER dispatching N/M nodes (cap=3, deferred=[...])` — concurrency limiting in action
- `SEARCHER [nX] parsed: findings=N chars, leads=M chars` — lead extraction working
- `REFLECT notes:` — what the reflection told the planner
- `PLANNER hit node budget (15), forcing finalize` — budget enforcement

### State machine reference

See [`docs/state_machine_reference.md`](docs/state_machine_reference.md) for the authoritative specification of all valid transitions, state fields, node lifecycles, and failure modes.

## Search Engines

LangMindSearch supports three search backends:

| Engine | API Key Required | Env Variable |
|--------|:---:|---|
| DuckDuckGo | No | — |
| Google (via Serper) | Yes | `SERPER_API_KEY` or `WEB_SEARCH_API_KEY` |
| Brave Search | Yes | `BRAVE_API_KEY` or `WEB_SEARCH_API_KEY` |

DuckDuckGo is the default and requires no setup. For better result quality, Serper (Google) is recommended.

## llama-server Performance Notes

LangMindSearch sends many concurrent requests to a single LLM endpoint. Performance depends heavily on llama-server configuration:

- **Parallel slots (`-np`)** — must be ≥ `MAX_CONCURRENT_SEARCHERS` (default 3). Without this, searchers serialise and batches take 3× longer.
- **Host-memory prompt caching (`-cram`)** — the planner and searcher system prompts are identical across calls. With `-cram 512`, the server caches and reuses the computed KV for these prefixes, skipping re-processing on every call.
- **Flash attention (`-fa`)** — reduces memory usage and improves throughput.
- **Context size (`-c`)** — should be large enough for the planner's context (system prompt + graph summary + reflection, typically 8K-20K tokens). 32768 is a safe default.

Monitor the llama-server logs for `slot update_slots` messages showing cache hit rates. You should see `progress = 0.0xxxxx` (mostly cached) rather than `progress = 1.000000` (full reprocessing) on subsequent requests.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Citation

LangMindSearch builds on the MindSearch research. If you find this work useful, please cite the original paper:

```bibtex
@article{chen2024mindsearch,
  title={MindSearch: Mimicking Human Minds Elicits Deep AI Searcher},
  author={Chen, Zehui and Liu, Kuikun and Wang, Qiuchen and Liu, Jiangning and Zhang, Wenwei and Chen, Kai and Zhao, Feng},
  journal={arXiv preprint arXiv:2407.20183},
  year={2024}
}
```

## Related Projects

- [MindSearch](https://github.com/InternLM/MindSearch) — the original project this fork is based on
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — the inference backend
- [LangGraph](https://github.com/langchain-ai/langgraph) — the graph execution framework
- [Open Deep Research](https://github.com/langchain-ai/open_deep_research) — LangChain's deep research agent (architectural inspiration)
- [GPT-Researcher](https://github.com/assafelovic/gpt-researcher) — another deep research framework (prompt design inspiration)
