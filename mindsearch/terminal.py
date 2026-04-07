# mindsearch/terminal.py

import argparse
import os

from mindsearch.agent import init_agent

parser = argparse.ArgumentParser(description="MindSearch terminal")
parser.add_argument("--llm_url", default="http://localhost:8080/v1",
                    help="llama-server base URL")
parser.add_argument("--search_engine", default="DuckDuckGoSearch",
                    choices=["DuckDuckGoSearch", "GoogleSearch", "BraveSearch"])
parser.add_argument("--query", default=None,
                    help="Run a single query and exit")
args = parser.parse_args()

graph = init_agent(
    llm_url=args.llm_url,
    search_engine=args.search_engine,
)


def run_query(query: str):
    initial_state = {
        "question": query,
        "nodes": [],
        "final_answer": "",
        "turns": 0,
    }

    print(f"\n[MindSearch] {query}\n")

    for event in graph.stream(initial_state, stream_mode="updates"):
        for node_name, update in event.items():
            if node_name == "planner":
                for n in update.get("nodes", []):
                    print(f"  -> [{n['id']}] {n['query']}")
            elif node_name == "searcher":
                for n in update.get("nodes", []):
                    if n["status"] == "done":
                        print(f"  done [{n['id']}]")
            elif node_name == "finalize":
                answer = update.get("final_answer", "")
                print("\n" + "=" * 60)
                print(answer)
                print("=" * 60)


if args.query:
    run_query(args.query)
else:
    print("MindSearch (LangGraph) — Ctrl-C to exit")
    while True:
        try:
            query = input("\nQuery> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if query:
            run_query(query)
