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
parser.add_argument("--debug", action="store_true",
                    help="Enable execution tracing")
parser.add_argument("--trace-dir", default="./traces",
                    help="Directory for trace output files")
parser.add_argument("--max-turns", type=int, default=8,
                    help="Maximum planner iterations")
parser.add_argument("--max-nodes", type=int, default=15,
                    help="Maximum total search nodes (exploration budget)")
parser.add_argument("--no-seed-search", action="store_true",
                    help="Disable preliminary landscape search")
parser.add_argument("--no-reflection", action="store_true",
                    help="Disable coverage reflection between rounds")
parser.add_argument("--enable-compression", action="store_true",
                    help="Enable findings compression (for small-context models)")
args = parser.parse_args()

graph, tracer = init_agent(
    llm_url=args.llm_url,
    search_engine=args.search_engine,
    max_turns=args.max_turns,
    max_nodes=args.max_nodes,
    enable_seed_search=not args.no_seed_search,
    enable_reflection=not args.no_reflection,
    enable_compression=args.enable_compression,
    debug=args.debug,
    trace_dir=args.trace_dir,
)


def run_query(query: str):
    initial_state = {
        "question": query,
        "nodes": [],
        "dispatched_ids": [],
        "to_dispatch_ids": [],
        "final_answer": "",
        "turns": 0,
        "seed_context": "",
        "reflection_notes": "",
    }

    print(f"\n[MindSearch] {query}\n")

    for event in graph.stream(
        initial_state,
        config={"recursion_limit": 200},
        stream_mode="updates",
    ):
        for node_name, update in event.items():
            if node_name == "planner":
                for n in update.get("nodes", []):
                    cat = n.get("category", "core")
                    print(f"  -> [{n['id']}] ({cat}) {n['query']}")
            elif node_name == "searcher":
                for n in update.get("nodes", []):
                    if n["status"] == "done":
                        leads = n.get("leads", "")
                        leads_preview = f" | leads: {leads[:80]}" if leads else ""
                        summary_len = len(n.get("summary", "") or "")
                        print(f"  done [{n['id']}] ({summary_len}ch)"
                              f"{leads_preview}")
            elif node_name == "seed_searcher":
                ctx_len = len(update.get("seed_context", "") or "")
                print(f"  [seed] Landscape survey done ({ctx_len} chars)")
            elif node_name == "reflect":
                notes = update.get("reflection_notes", "") or ""
                if notes:
                    print(f"  [reflect] {notes[:120]}...")
            elif node_name == "finalize":
                answer = update.get("final_answer", "")
                print("\n" + "=" * 60)
                print(answer)
                print("=" * 60)

    # Print trace summary and save
    tracer.print_summary()
    trace_path = tracer.save(question=query)
    if trace_path:
        print(f"\nTrace saved to: {trace_path}")


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
