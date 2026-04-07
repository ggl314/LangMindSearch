# mindsearch/agent/__init__.py

import os
from .lg_agent import build_graph


def init_agent(
    llm_url: str = "http://localhost:8080/v1",
    search_engine: str = "DuckDuckGoSearch",
    api_key: str = None,
    **kwargs,   # absorb any legacy kwargs (lang, model_format, use_async) silently
):
    """
    Build and return a compiled LangGraph MindSearch agent.

    Args:
        llm_url:       Base URL of llama-server OpenAI-compatible endpoint.
                       e.g. "http://localhost:8080/v1"
        search_engine: "DuckDuckGoSearch" | "GoogleSearch" | "BraveSearch"
        api_key:       Search API key. Not needed for DuckDuckGo.
                       Reads WEB_SEARCH_API_KEY or SERPER_API_KEY from env
                       if not passed explicitly.
    """
    if api_key is None:
        api_key = os.getenv("WEB_SEARCH_API_KEY") or os.getenv("SERPER_API_KEY")

    return build_graph(
        llm_url=llm_url,
        search_engine=search_engine,
        api_key=api_key,
    )
