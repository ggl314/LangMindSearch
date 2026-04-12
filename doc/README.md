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

The entire backend runs on local LLMs via [llama.cpp](https://github.com/ggml-org/llama.cpp), with no dependency on cloud APIs for inference. Primary target models are **Qwen3.5** (27B dense or 35B-A3B MoE), with support for **Gemma 4** and **GPT-OSS-120B** (experimental).

### What changed from upstream MindSearch

The original MindSearch used a Python code-generation planner (the LLM writes Python to build a search graph) backed by the Lagent framework. LangMindSearch replaces this entirely:

- **LangGraph-based DAG planner** — the planner emits JSON describing search nodes with dependencies, categories, and relevance rationales. No code generation, no Lagent dependency.
- **Exploratory research prompts** — the system is prompted to explore beyond the literal question: adjacent topics, historical context, emerging trends, controversies, and limitations. A node budget and category targets keep exploration disciplined.
- **Seed search** — before planning, a broad web survey discovers the topic landscape, giving the planner real-world context about what exists.
- **Reflection loop** — after each search batch, an LLM-powered reflection step assesses coverage gaps and feeds actionable guidance back to the planner.
- **Selective thinking mode** — reasoning-heavy calls (reflection, synthesis, searcher summaries) use the model's thinking mode (`<think>` tags) for deeper analysis, while fast calls (planning, search queries) run with thinking disabled for speed.
- **Structured leads** — each searcher outputs prioritised leads (topics discovered during search that weren't the focus), which the planner can follow up on.
- **Substance-first reports** — the synthesis prompt is tuned for dense, fact-heavy output with specific numbers, benchmarks, and comparisons. No executive summaries, no speculative "future work" sections — just the substance the research found.
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

### Thinking Mode

LangMindSearch uses two LLM configurations pointing at the same llama-server, differing only in whether the model's thinking/reasoning mode is active:

| Node | Thinking | Why |
|------|:---:|-----|
| Seed searcher | off | Generating broad search queries — speed matters |
| Planner | off | Emitting JSON — structured output, no reasoning needed |
| Searcher (tool rounds) | off | Deciding what to search — latency-sensitive |
| Searcher (final summary) | **on** | Weighing evidence, extracting findings vs leads |
| Reflect | **on** | Assessing coverage, identifying gaps — pure reasoning |
| Finalize | **on** | Synthesising report from all nodes — deep reasoning |

For Qwen3.5 models, thinking mode is controlled via `enable_thinking` in the chat template (requires `--jinja` on llama-server). For Gemma 4, the equivalent mechanism uses `<start_of_thought>` tags. Thinking tokens are generated but stripped from visible output.

## Quick Start

### Prerequisites

- Python 3.10+
- A running [llama-server](https://github.com/ggml-org/llama.cpp) instance with an OpenAI-compatible API
- Node.js 18+ (for the React frontend)

### Recommended Models

| Model | VRAM | Best For |
|-------|------|----------|
| **Qwen3.5-27B** Q4_K_M | ~17 GB | Deep reasoning, complex research queries |
| **Qwen3.5-35B-A3B** Q4_K_M | ~21 GB | Fast throughput, agentic workflows |
| **Gemma 4** 27B Q4 | ~17 GB | Alternative reasoning model |
| **GPT-OSS-120B** (multi-GPU) | ~80 GB | Experimental, highest quality |

The Qwen3.5-27B dense model is the recommended default — it provides the best reasoning quality for the planner, reflection, and synthesis steps. The 35B-A3B MoE variant trades some reasoning depth for ~3× faster inference, which can be worthwhile when running many research cycles.

### 1. Install dependencies

```bash
git clone https://github.com/ggl314/LangMindSearch
cd LangMindSearch
pip install -r requirements.txt
```

### 2. Start llama-server

Start llama-server with parallel slots, prompt caching, and Jinja template support (required for thinking mode):

```bash
llama-server \
  -m qwen3.5-27b-q4_k_m.gguf \
  --jinja \
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
- `--jinja` — **required** for thinking mode to work (enables chat template kwargs)
- `-np 3` — three parallel slots, matching the default searcher concurrency cap
- `-cram 512` — 512 MiB host RAM for prompt caching (system prompts are reused across calls)
- `-sps 0.3` — lower similarity threshold for slot matching (searchers share a system prompt but differ in user content)
- `--cache-reuse 128` — minimum chunk size for KV shift-based reuse
- `-fa` — flash attention, reduces memory and improves throughput

### 3. Start the backend

```bash
python -m mindsearch.app --llm_url http://localhost:8080/v1
```

Options:
- `--llm_url` — llama-server endpoint (default: `http://localhost:8080/v1`, or set `LLM_URL` env var)
- `--search_engine` — `DuckDuckGoSearch` (default, no API key), `GoogleSearch` (requires `SERPER_API_KEY`), or `BraveSearch` (requires `BRAVE_API_KEY`)

Environment variables for tuning:

| Variable | Default | Effect |
|----------|---------|--------|
| `MAX_TURNS` | `8` | Maximum planner iterations before forcing finalize |
| `MAX_NODES` | `15` | Total node budget (exploration cap) |
| `MAX_CONCURRENT_SEARCHERS` | `3` | Parallel searcher limit |
| `ENABLE_SEED_SEARCH` | `true` | Run a landscape survey before planning |
| `ENABLE_REFLECTION` | `true` | LLM-powered coverage assessment between rounds |
| `ENABLE_THINKING` | `true` | Use thinking mode for reasoning calls (requires `--jinja`) |
| `ENABLE_COMPRESSION` | `false` | Compress findings before synthesis (for small-context models) |
| `DEBUG` | `false` | Enable execution tracing |
| `TRACE_DIR` | `./traces` | Output directory for trace files |

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

4. **Searching** — each searcher runs a ReAct loop (up to 3 rounds of tool calls). Tool-call rounds use fast mode (no thinking) for low latency. The final synthesis round uses thinking mode to reason about results, then outputs structured FINDINGS (detailed, data-dense) and prioritised LEADS.

5. **Reflection** — after all searchers in a batch complete, the reflection node uses thinking mode to assess coverage, identify gaps, and write actionable guidance into the state for the planner's next turn.

6. **Repeat** until the planner signals finalize, or the node budget / turn limit is reached.

7. **Synthesis** — the finalize node receives all findings grouped by category and uses thinking mode to produce a substantive report organised by topic, dense with specific data, and free of speculative filler.

## Report Quality

The synthesis step is specifically tuned to avoid common LLM report-writing failure modes:

- **No executive summaries or abstracts** — the report starts with substance
- **No speculative "Areas for Further Investigation"** — the system reports what it found, not what it imagines you might want to know next
- **No skeleton reports** — the prompt demands specific numbers, benchmark scores, prices, and named sources rather than vague summaries
- **Topic-organised, not template-organised** — headings reflect the actual content (e.g., "27B vs 35B-A3B: Intelligence vs Speed") rather than mandatory report sections

If reports feel too thin, the most impactful lever is the model choice. The Qwen3.5-27B dense model produces substantially richer synthesis than the 35B-A3B MoE due to its higher per-token reasoning quality.

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

- **`--jinja`** — required for thinking mode. Without this, `enable_thinking` kwargs are silently ignored and all calls run in the model's default mode.
- **Parallel slots (`-np`)** — must be ≥ `MAX_CONCURRENT_SEARCHERS` (default 3). Without this, searchers serialise and batches take 3× longer.
- **Host-memory prompt caching (`-cram`)** — the planner and searcher system prompts are identical across calls. With `-cram 512`, the server caches and reuses the computed KV for these prefixes, skipping re-processing on every call.
- **Flash attention (`-fa`)** — reduces memory usage and improves throughput.
- **Context size (`-c`)** — should be large enough for the planner's context (system prompt + graph summary + reflection, typically 8K-20K tokens). 32768 is a safe default.

Monitor the llama-server logs for `slot update_slots` messages showing cache hit rates. You should see `progress = 0.0xxxxx` (mostly cached) rather than `progress = 1.000000` (full reprocessing) on subsequent requests.

### Thinking mode token budget

When thinking is enabled, reasoning calls generate hidden `<think>` tokens that are stripped before output. A reflection call producing 200 visible tokens may generate 500+ total. The reasoning LLM is configured with 2× the `max_tokens` of the fast LLM to accommodate this. For the finalize step (which produces long reports), you may need to increase this further if reports are being truncated.

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
