# LangMindSearch: Narrative Synthesis & Citation Quality — Claude Code Plan

## Problem

The current synthesis produces a report that reads as a sequence of independent
sections — one per research stage — rather than a connected narrative. Transitions
between sections are abrupt or absent. The report structure mirrors the research
plan structure rather than the natural flow of the topic.

Citations are also weak: URLs appear sporadically, often clustered at the end of
a section or missing entirely. Individual claims lack attribution. The reader
can't trace a specific fact back to its source.

## Target Quality

The report should read like a well-written research briefing:
- Sections flow into each other with transition sentences
- Later sections reference earlier findings naturally ("As noted in the
  benchmarks above, the 0.6B draft achieved...")
- Every factual claim (numbers, benchmarks, quotes, product names, dates)
  has an inline numbered citation: [1], [2], etc.
- A numbered reference list at the end with full URLs
- The structure follows the topic's logic, not the research plan's stage order

## Changes

### 1. Stage-Level: Preserve Source URLs in Structured Form

The current searcher outputs URLs mixed into prose. The synthesis step can't
reliably extract them. Fix this at the searcher level by requiring structured
source output.

#### Update `SEARCHER_SYSTEM` in `lg_prompts.py`

Add a third section to the searcher output format, after FINDINGS and LEADS:

```python
SEARCHER_SYSTEM = f"""{_DATE}

You are a focused web research agent assigned a single search task.

Steps:
1. Call the search tool with 1-3 query variants to maximize recall.
2. If results are insufficient, refine and search again (3 rounds max)
3. Synthesise a response with EXACTLY three sections, using these exact headings:

FINDINGS:
Write a detailed summary (6-12 sentences) of what you found. Include ALL specific
numbers, benchmarks, prices, performance metrics, dates, and named sources.
When stating a fact, note which source it came from in brackets, e.g.
"The RTX 5090 achieves 213 tok/s [TechPowerUp benchmark]".

LEADS:
List 2-4 specific topics with importance ratings:
- [high] Topic: why it matters
- [medium] Topic: relevant but not critical
- [low] Topic: tangentially interesting

SOURCES:
List every URL that contributed to the findings above, one per line, with a
short description of what the source provided:
- https://example.com/article : RTX 5090 benchmark data, inference speeds
- https://reddit.com/r/LocalLLaMA/... : community discussion on VRAM requirements

You MUST use the exact headings "FINDINGS:", "LEADS:", and "SOURCES:".
"""
```

#### Update `_parse_searcher_output` to extract sources

```python
def _parse_searcher_output(raw: str) -> tuple[str, str, str]:
    """Split searcher output into (findings, leads, sources).

    Returns three strings. If a section isn't found, returns "".
    """
    raw_upper = raw.upper()

    # Find section boundaries
    leads_idx = raw_upper.rfind("LEADS")
    sources_idx = raw_upper.rfind("SOURCES")

    if sources_idx != -1:
        sources_start = raw.rfind("\n", 0, sources_idx)
        sources_start = 0 if sources_start == -1 else sources_start + 1
        sources = raw[sources_idx:].strip()
        # Strip heading
        for prefix in ("SOURCES:", "SOURCES"):
            if sources.upper().startswith(prefix):
                sources = sources[len(prefix):].strip()
                break
    else:
        sources = ""
        sources_start = len(raw)

    if leads_idx != -1 and leads_idx < sources_start:
        leads_start = raw.rfind("\n", 0, leads_idx)
        leads_start = 0 if leads_start == -1 else leads_start + 1
        leads_end = sources_start if sources_idx != -1 else len(raw)
        leads = raw[leads_idx:leads_end].strip()
        for prefix in ("LEADS:", "LEADS"):
            if leads.upper().startswith(prefix):
                leads = leads[len(prefix):].strip()
                break
        findings_end = leads_start
    else:
        leads = ""
        findings_end = sources_start

    findings = raw[:findings_end].strip()
    for prefix in ("FINDINGS:", "FINDINGS"):
        if findings.upper().startswith(prefix):
            findings = findings[len(prefix):].strip()
            break

    return findings, leads, sources
```

#### Update `SearchNode` to include sources

```python
class SearchNode(TypedDict):
    id: str
    query: str
    depends_on: list[str]
    status: str
    summary: str      # FINDINGS
    leads: str        # LEADS
    sources: str      # SOURCES (URLs with descriptions)
    category: str
    rationale: str
```

### 2. Stage Finalize: Preserve Sources in Stage Summary

The stage-level finalize (which produces the rich summary per plan stage) must
carry the sources through. Update `STAGE_FINALIZE_SYSTEM`:

```python
STAGE_FINALIZE_SYSTEM = f"""{_DATE}

You are a research summariser for one stage of a multi-stage research project.

Write a detailed summary with two parts:

SUMMARY:
A dense, fact-packed summary preserving ALL specific numbers, benchmarks,
named entities, and technical details. When stating a fact, include an inline
reference number [N] corresponding to the source list below.

REFERENCES:
A numbered list of all source URLs from the search findings, in the format:
[1] Short description - URL
[2] Short description - URL

CRITICAL:
- Every factual claim in the summary MUST have a [N] reference
- Number references sequentially starting from [1]
- Include ALL source URLs from the search results — do not drop any
- This summary will be fed to a final synthesis agent that needs the references
  to produce a properly cited report
"""
```

This means each stage summary arrives at the narrative synthesis step with
its own numbered reference list. The synthesis step can then build a unified
reference list.

### 3. Narrative Synthesis: Connected Prose with Unified Citations

This is the main change. The `NARRATIVE_SYNTHESIS_SYSTEM` prompt needs to be
completely rewritten to produce connected narrative with inline citations.

```python
NARRATIVE_SYNTHESIS_SYSTEM = f"""{_DATE}

You are a research report writer producing a single, connected narrative from
multiple research stage summaries. The report must read as one coherent text,
not as a collection of independent sections.

NARRATIVE RULES:
- Write in continuous prose. Each section should flow into the next.
- Use transition sentences between sections that connect ideas:
  "Building on the architectural foundations above, the practical deployment..."
  "As the benchmark data shows, this theoretical advantage translates to..."
  "While the previous section established the cost structure, community
   experience reveals additional considerations..."
- Reference earlier sections naturally when relevant:
  "As noted in the benchmarks above...", "The distillation approach
   described earlier...", "Returning to the expert routing question..."
- Do NOT write sections that could stand alone as independent documents.
  Every section should assume the reader has read what came before.
- Do NOT start with an executive summary or abstract. Begin directly with
  the first substantive topic.
- Do NOT end with a conclusion, "future work", or "areas for investigation"
  section unless the research plan explicitly included one.

SECTION STRUCTURE:
- Use the stage titles as a starting point but reorganise freely to create
  the best narrative flow. Merge small related stages. Split large stages
  if they cover distinct ideas.
- Use ## for section headings. Keep headings specific and descriptive
  (not generic like "Background" or "Analysis").
- Within sections, use paragraphs. Reserve bullet lists and tables for
  genuinely structured data (comparisons, specifications, benchmarks).

CITATION RULES (critical):
- Build a single unified reference list from all stage summaries.
- Assign each unique URL a sequential number [1], [2], [3]... across the
  entire report. If the same URL appears in multiple stage summaries, it
  gets ONE number.
- Place inline citations immediately after the specific claim they support:
  "The draft model achieved 2.1x speedup on coding tasks [15]."
  "RTX 5090 delivers 213 tok/s for 8B models [3], compared to 35 tok/s
   on the RTX 4090 [7]."
- Every factual claim (number, benchmark, date, product spec, quote,
  community observation) MUST have at least one citation.
- Uncitable claims (your own synthesis, logical deductions connecting
  facts from different sources) do not need citations but should be
  clearly framed as synthesis: "Taken together, these benchmarks suggest..."
- End the report with a ## References section listing all cited sources:
  [1] Description - URL
  [2] Description - URL
  Numbers must be sequential with no gaps.

CONTENT RULES:
- Include all specific numbers, benchmarks, and technical details from
  the stage summaries. Do not generalise.
- Do not add information not present in the stage summaries.
- Use tables for comparative data (benchmarks, prices, specifications).
- Preserve the specific language of community quotes where relevant
  (e.g. "users describe it as 'ridiculously fast'").
"""

NARRATIVE_SYNTHESIS_USER = """Original research question:
{query}

Research plan that was executed:
{plan_outline}

Stage summaries with per-stage references:

{all_stage_summaries}

Write a comprehensive narrative report. Build a unified reference list
from all stage references — deduplicate URLs that appear in multiple stages.
Every factual claim must have an inline [N] citation."""
```

### 4. Reference Deduplication Helper

Before passing stage summaries to the synthesis LLM, preprocess to build a
unified URL-to-number mapping. This helps the LLM but isn't strictly required
(the LLM can deduplicate, but providing a pre-built map reduces errors):

```python
def _build_reference_map(stage_summaries: list[dict]) -> tuple[str, dict]:
    """Extract all URLs from stage summaries, deduplicate, assign numbers.

    Returns:
        annotated_summaries: stage summaries with a note about the global
                            reference mapping
        ref_list: {url: number} mapping
    """
    import re
    all_urls = []
    seen = set()

    for stage in stage_summaries:
        # Extract URLs from the REFERENCES section of each stage summary
        urls = re.findall(r'https?://[^\s\]]+', stage.get("summary", ""))
        for url in urls:
            clean = url.rstrip('.,;:)')
            if clean not in seen:
                seen.add(clean)
                all_urls.append(clean)

    # Build numbered reference list
    ref_map = {url: i + 1 for i, url in enumerate(all_urls)}
    ref_list_text = "\n".join(f"[{n}] {url}" for url, n in ref_map.items())

    return ref_list_text, ref_map
```

Then in the synthesis user prompt, append:

```python
pre_built_refs, _ = _build_reference_map(stage_summaries)

NARRATIVE_SYNTHESIS_USER = f"""...

Pre-built reference list (use these numbers for inline citations):
{pre_built_refs}

..."""
```

This gives the LLM a pre-numbered reference list to work from, dramatically
reducing citation numbering errors.

### 5. Thinking Mode for Synthesis

The narrative synthesis is the most reasoning-heavy call in the entire pipeline.
It must:
- Read 8-10 stage summaries (potentially 20K+ tokens)
- Plan a narrative structure
- Cross-reference between stages
- Build a unified citation system
- Write connected prose

This MUST use the reasoning LLM (thinking enabled). The thinking phase lets
the model plan the narrative structure, resolve cross-references, and build
the citation map before writing. Without thinking, the model tends to process
stages sequentially and produce the disconnected chapter-per-stage output.

The `max_tokens` for this call should be generous — at least 16384, possibly
higher for complex topics. The thinking tokens alone may consume 2-4K.

---

## What These Changes Fix

### Before (current output style):

```markdown
## Fundamentals of Speculative Decoding
Speculative decoding uses a draft model to propose tokens...
Source: https://example.com/article1

## DeepSeek Draft Models
DeepSeek created V2-Lite with 16B parameters...
Source: https://example.com/article2

## Expert Budgeting
MoE-Spec introduces expert budgeting to limit...
```

Problems: no transitions, independent sections, citations as raw URLs at
section end, no cross-referencing.

### After (target output style):

```markdown
## The Memory Bandwidth Problem in Sparse Models

Speculative decoding addresses the fundamental bottleneck in autoregressive
generation: memory bandwidth [1]. For a 230B MoE like DeepSeek-V2, even
with sparse activation of only 21B parameters per token, the weight-loading
cost dominates inference time [3]. The technique works by decoupling token
proposal from verification — a small draft model generates K candidate
tokens, which the master model verifies in a single parallel forward
pass [1][7].

The mathematical foundation rests on rejection sampling: a token x is
accepted with probability min(1, p(x)/q(x)), ensuring the final output
follows the target distribution exactly [1]. This is why users consistently
report that "quality cannot degrade" under speculative decoding [4] — any
draft error is corrected by the verifier.

## From Theory to Architecture: Building the Draft

Given this theoretical foundation, the practical question becomes how to
build an effective draft model. The DeepSeek team's approach illustrates
the key design choices. DeepSeek-V2-Lite, a 16B total parameter model
with 2.4B active, mirrors the V2 architecture using the same Multi-head
Latent Attention and DeepSeekMoE primitives [3]. This architectural
alignment yields higher acceptance rates than a generic dense model
because the draft's routing logic matches the master's [3][18].

However, community experience suggests that even simpler approaches
work well. Jukofyork's 0.6B draft models for DeepSeek-R1 [10] used a
completely different base (Qwen-2.5-0.5B) with vocabulary transplant
and QLoRA fine-tuning on 2.3B tokens [11]. The key insight from this
work is that draft model size has diminishing returns — a 0.5B model
is often faster end-to-end than a 7B model despite lower acceptance
rates, because the speed advantage outweighs the acceptance penalty [8].

...

## References

[1] Decoding Speculative Decoding - https://ar5iv.org/html/2402.01528v4
[3] DeepSeek-V2 paper - https://arxiv.org/html/2405.04434v3
[4] llama.cpp speculative decoding discussion - https://www.reddit.com/...
[7] NVIDIA speculative decoding blog - https://developer.nvidia.com/...
[8] Yan et al. draft size analysis - ...
[10] jukofyork DeepSeek-V3 draft - https://huggingface.co/jukofyork/...
[11] jukofyork training details - ...
[18] DeepSeek-V2-Lite - https://huggingface.co/deepseek-ai/...
```

Key differences: transitions connect sections, inline numbered citations on
every claim, single unified reference list, narrative flows as one continuous
argument.

---

## Implementation Order

1. **Update `SEARCHER_SYSTEM`** to add SOURCES section and inline source
   attribution in findings
2. **Update `_parse_searcher_output`** to handle three sections
3. **Update `SearchNode`** to include `sources` field
4. **Update `STAGE_FINALIZE_SYSTEM`** to preserve numbered references
5. **Update `NARRATIVE_SYNTHESIS_SYSTEM`** with the full rewrite above
6. **Add `_build_reference_map` helper** for pre-deduplication
7. **Ensure synthesis uses reasoning LLM** with generous max_tokens
8. **Test** with the speculative decoding query — compare output structure
   against the ChatGPT PDF
