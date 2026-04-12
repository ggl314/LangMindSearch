# mindsearch/agent/__init__.py

import os
from .lg_agent import build_graph


def init_agent(
    llm_url: str = "http://localhost:8080/v1",
    search_engine: str = "DuckDuckGoSearch",
    api_key: str = None,
    max_turns: int = 8,
    max_nodes: int = 15,
    max_concurrent_searchers: int = 3,
    enable_seed_search: bool = True,
    enable_reflection: bool = True,
    enable_compression: bool = False,
    enable_thinking: bool = True,
    debug: bool = False,
    trace_dir: str = "./traces",
    **kwargs,   # absorb any legacy kwargs (lang, model_format, use_async) silently
):
    """
    Build and return a compiled LangGraph MindSearch agent.

    Returns:
        (graph, tracer) — compiled LangGraph and its ResearchTracer. When
        debug=False the tracer is a no-op wrapper but still returned.

    Args:
        llm_url:            Base URL of llama-server OpenAI-compatible endpoint.
                            e.g. "http://localhost:8080/v1"
        search_engine:      "DuckDuckGoSearch" | "GoogleSearch" | "BraveSearch"
        api_key:            Search API key. Not needed for DuckDuckGo.
                            Reads WEB_SEARCH_API_KEY or SERPER_API_KEY from env
                            if not passed explicitly.
        max_turns:          Maximum planner iterations before forcing finalize.
        max_nodes:          Maximum total search nodes (exploration budget).
        enable_seed_search: Run broad landscape search before planning.
        enable_reflection:  LLM-powered coverage assessment between rounds.
        enable_compression: Compress findings before final synthesis
                            (for small-context models; not needed with 128K).
        debug:              Enable structured execution tracing.
        trace_dir:          Output directory for trace JSON files.
    """
    if api_key is None:
        api_key = os.getenv("WEB_SEARCH_API_KEY") or os.getenv("SERPER_API_KEY")

    return build_graph(
        llm_url=llm_url,
        search_engine=search_engine,
        api_key=api_key,
        max_turns=max_turns,
        max_nodes=max_nodes,
        max_concurrent_searchers=max_concurrent_searchers,
        enable_seed_search=enable_seed_search,
        enable_reflection=enable_reflection,
        enable_compression=enable_compression,
        enable_thinking=enable_thinking,
        debug=debug,
        trace_dir=trace_dir,
    )
