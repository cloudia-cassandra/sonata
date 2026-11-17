"""
Sonata's command line.

  python -m sonata ask "<request>" [--no-audio]   RAG recommendations with reasons and retrieval similarity
  python -m sonata recommend [-k 5]               content-based recommendations from audio features
  python -m sonata evaluate [--fresh]             run the evaluation and print the summary table

All three work from the committed data (data/library.json and data/features.parquet):
no Spotify login and no MP3 previews are needed. The LLM provider comes from .env
(LLM_PROVIDER, mock by default). The first "ask" downloads the small embedding model
(all-MiniLM-L6-v2) from Hugging Face once.

Rebuilding the data is separate and does need Spotify and the previews:
  python -m sonata.fetch_spotify_taste && python -m sonata.build_library && python -m sonata.analyze
"""

import argparse
import logging
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m sonata", description="Sonata: songs picked from your taste")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    ask = commands.add_parser("ask", help="ask for songs in plain language (RAG)")
    ask.add_argument("request", help='for example: "upbeat songs for a night drive"')
    ask.add_argument("--no-audio", action="store_true", help="rank by text similarity only")

    recommend = commands.add_parser("recommend", help="new songs that sound like the ones you like")
    recommend.add_argument("-k", type=int, default=5, help="number of recommendations (default 5)")

    evaluate = commands.add_parser("evaluate", help="run the evaluation and print the summary table")
    evaluate.add_argument("--fresh", action="store_true",
                          help="bypass the response cache for every run (costs Gemini calls)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    # Sonata's own notices at INFO; library chatter (Hugging Face, httpx, Gemini SDK) only at WARNING
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s", stream=sys.stderr)

    # Imported per command so each one loads only what it needs
    if args.command == "ask":
        from sonata import rag
        rag.log.setLevel(logging.INFO)
        result = rag.ask(args.request, use_audio=not args.no_audio)
        if result["source"] == "rejected":
            print(result["message"], file=sys.stderr)
            return 2
        print(rag.format_result(result))
    elif args.command == "recommend":
        from sonata.recommender import format_recommendations, load_and_recommend
        if args.k < 1:
            print("-k must be at least 1", file=sys.stderr)
            return 2
        print(format_recommendations(load_and_recommend(args.k)))
    elif args.command == "evaluate":
        from sonata import evaluate, rag
        # The report lists every notice; don't also print each fallback warning per run
        rag.log.setLevel(logging.ERROR)
        print(evaluate.run(fresh=args.fresh))
    return 0


if __name__ == "__main__":
    sys.exit(main())
