import argparse

from mindsearch.agent import init_agent

parser = argparse.ArgumentParser()
parser.add_argument("--lang", default="en", choices=["cn", "en"])
parser.add_argument("--model_format", default="llamacpp_server")
parser.add_argument("--search_engine", default="DuckDuckGoSearch")
parser.add_argument("--query", default=None, help="Query to run (omit for interactive mode)")
args = parser.parse_args()

agent = init_agent(lang=args.lang, model_format=args.model_format,
                   search_engine=args.search_engine, use_async=False)

def run_query(query):
    for agent_return in agent(query):
        pass
    print(agent_return.sender)
    print(agent_return.content)
    if hasattr(agent_return, "formatted") and agent_return.formatted:
        print(agent_return.formatted.get("ref2url", ""))

if args.query:
    run_query(args.query)
else:
    print("MindSearch terminal (Ctrl-C to exit)")
    while True:
        try:
            query = input("\nQuery> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not query:
            continue
        run_query(query)
