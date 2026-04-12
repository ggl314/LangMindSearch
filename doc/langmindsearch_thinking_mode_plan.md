# LangMindSearch: Selective Thinking Mode — Claude Code Plan

## Concept

LangMindSearch makes two kinds of LLM calls:

**Fast calls (structured output, no reasoning needed):**
- Planner: emit JSON DAG nodes — needs to be fast, output is structured
- Seed searcher: generate broad search queries — speed matters
- Searcher tool-call rounds: decide what to search next — latency-sensitive

**Reasoning calls (quality matters more than speed):**
- Reflect: assess coverage gaps, weigh what's missing — this IS reasoning
- Finalize: synthesise 10-15 nodes of findings into a coherent report — deep reasoning
- Searcher final summary: weigh search results, extract findings vs leads — moderate reasoning

The plan is to use two LLM configurations: a fast one (thinking disabled) and
a reasoning one (thinking enabled). Both point to the same llama-server endpoint
and model — the difference is in the request parameters that control thinking.

## How Thinking Mode Works in Target Models

### Qwen3.5 (primary target)
- Thinking mode controlled by `enable_thinking` parameter in chat template
- With `--jinja` on llama-server, pass via `chat_template_kwargs`:
  `{"enable_thinking": true}` or `{"enable_thinking": false}`
- When enabled, model outputs `<think>...</think>` before the actual response
- The thinking content appears in the response but can be stripped
- Default behaviour (no flag): thinking enabled (Qwen3.5 "thinks by default")

### Gemma 4 (secondary target)
- Uses `<start_of_thought>...<end_of_thought>` tags
- Controlled via system prompt or template kwargs depending on build
- llama-server with `--jinja` handles this through the model's chat template

### GPT-OSS-120B (experimental)
- Thinking mode support varies by quant/build
- Typically controlled via system prompt instruction

## Implementation

### Changes to `lg_agent.py`

#### Create two LLM instances in `build_graph`:

```python
def build_graph(
    llm_url: str,
    search_engine: str,
    ...
    enable_thinking: bool = True,
):
    # Fast LLM — thinking disabled, used for planner, seed, searcher tool rounds
    fast_llm = ChatOpenAI(
        base_url=llm_url,
        api_key="sk-no-key-required",
        model="local",
        temperature=0.0,
        max_tokens=max_tokens,
        streaming=False,
        max_retries=6,
        model_kwargs={
            "cache_prompt": True,
            **({"chat_template_kwargs": {"enable_thinking": False}} if enable_thinking else {}),
        },
    )

    # Reasoning LLM — thinking enabled, used for reflect, finalize, searcher synthesis
    reason_llm = ChatOpenAI(
        base_url=llm_url,
        api_key="sk-no-key-required",
        model="local",
        temperature=0.0,
        max_tokens=max_tokens * 2,  # thinking tokens need more headroom
        streaming=False,
        max_retries=6,
        model_kwargs={
            "cache_prompt": True,
            **({"chat_template_kwargs": {"enable_thinking": True}} if enable_thinking else {}),
        },
    )
```

#### Wire each node to the appropriate LLM:

```python
    graph.add_node("planner",    make_planner_node(fast_llm, planner_system))
    graph.add_node("searcher",   make_searcher_node(fast_llm, reason_llm, search_tool))
    graph.add_node("finalize",   make_finalize_node(reason_llm, enable_compression))

    if enable_seed_search:
        graph.add_node("seed_searcher", make_seed_searcher_node(fast_llm, search_tool))

    if enable_reflection:
        graph.add_node("reflect", make_reflect_node(reason_llm))
```

Note: `make_searcher_node` gets BOTH LLMs — fast for tool-call rounds, reasoning
for the final summary.

#### Update `make_searcher_node` to use two LLMs:

```python
def make_searcher_node(fast_llm: ChatOpenAI, reason_llm: ChatOpenAI, search_tool):
    fast_with_tools = fast_llm.bind_tools([search_tool])
    reason_with_tools = reason_llm.bind_tools([search_tool])

    def searcher_node(state: SearcherState) -> dict:
        messages = [
            SystemMessage(content=SEARCHER_SYSTEM),
            HumanMessage(content=f"Search task: {state['query']}\n\n"
                                 f"Parent question for context: {state['question']}"),
        ]

        final_response = None
        for round_i in range(3):
            # Use fast LLM for tool-call rounds, reasoning LLM for final synthesis
            is_likely_final = (round_i == 2)  # last allowed round
            llm = reason_with_tools if is_likely_final else fast_with_tools

            response = llm.invoke(messages)
            messages.append(response)
            final_response = response

            if not response.tool_calls:
                break

            # If this was supposed to be the final round but model wants more tools,
            # we still process them — but the NEXT response won't have thinking
            for tc in response.tool_calls:
                result = search_tool.invoke(tc["args"])
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        # If we exited the loop because of tool calls on the last round,
        # do one more reasoning call for synthesis
        if final_response and final_response.tool_calls:
            final_response = reason_with_tools.invoke(messages)
            messages.append(final_response)

        raw_output = (final_response.content or "No results found.") if final_response \
                     else "Search failed."

        # Strip thinking tags from response if present
        raw_output = _strip_thinking(raw_output)

        findings, leads = _parse_searcher_output(raw_output)
        ...
```

#### Add thinking tag stripper:

```python
import re

def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output.
    
    Qwen3.5 wraps reasoning in <think>...</think>.
    Gemma 4 uses <start_of_thought>...<end_of_thought>.
    We strip both to get clean output for parsing and display.
    """
    # Qwen3.5 style
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    # Gemma 4 style
    text = re.sub(r'<start_of_thought>.*?<end_of_thought>', '', text, flags=re.DOTALL).strip()
    return text
```

Apply `_strip_thinking` to the output of every reasoning LLM call:
- `make_reflect_node`: strip before writing to `reflection_notes`
- `make_finalize_node`: strip before writing to `final_answer`
- `make_searcher_node`: strip before parsing findings/leads (shown above)

The planner and seed searcher use fast_llm (thinking disabled), so no stripping needed.

### Changes to `__init__.py`

```python
def init_agent(
    ...
    enable_thinking: bool = True,
    **kwargs,
):
    return build_graph(
        ...
        enable_thinking=enable_thinking,
    )
```

### Changes to `app.py`

```python
agent_graph = init_agent(
    ...
    enable_thinking=os.getenv("ENABLE_THINKING", "true").lower() == "true",
)
```

### llama-server requirements

The server MUST be started with `--jinja` for thinking mode to work:

```bash
llama-server \
  -m qwen3.5-27b-q4_k_m.gguf \
  --jinja \
  -ngl 99 -fa -c 32768 -np 3 \
  -cram 512 -sps 0.3 --cache-reuse 128
```

For Qwen3.5 models, `--jinja` is required for the `enable_thinking` chat
template kwarg to take effect. Without `--jinja`, the kwarg is silently ignored.

### Which calls use which LLM (summary)

| Node | LLM | Thinking | Why |
|------|-----|:---:|-----|
| seed_searcher | fast | off | Generating broad search queries — speed matters |
| planner | fast | off | Emitting JSON — structured output, no reasoning needed |
| searcher (tool rounds) | fast | off | Deciding what to search — latency-sensitive |
| searcher (final summary) | reason | ON | Weighing evidence, extracting findings vs leads |
| reflect | reason | ON | Assessing coverage, identifying gaps — pure reasoning |
| finalize | reason | ON | Synthesising report from 15 nodes — deep reasoning |

### Config

| Env Variable | Default | Effect |
|---|---|---|
| `ENABLE_THINKING` | `true` | Toggle thinking mode for reasoning calls |

When `enable_thinking=False`, both LLMs are identical (both use fast config).
This is the fallback for models that don't support thinking mode.

### Token budget note

Thinking tokens are "hidden" — they're generated but stripped from the output.
A reflection call that produces 200 tokens of visible output may generate 500+
tokens including thinking. Set `max_tokens` on the reasoning LLM to at least
2× what you'd use for fast calls. The plan uses `max_tokens * 2` for reason_llm.

For finalize (which produces a 10K+ char report), this may need to be higher.
Consider making it separately configurable: `max_tokens_reason=16384`.
