"""Command-line interface for the Research Copilot RAG project."""

from __future__ import annotations

import argparse

from src.rag import RAGPipeline


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="research-copilot-rag",
        description="Run the Research Copilot RAG pipeline.",
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Question to answer.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved contexts.",
    )

    parser.add_argument(
        "--max-contexts",
        type=int,
        default=None,
        help="Maximum contexts included in the prompt.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum generated tokens.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature.",
    )

    return parser


def run_pipeline(
    pipeline: RAGPipeline,
    args: argparse.Namespace,
) -> str:
    """Execute the pipeline and return the answer."""
    return pipeline.answer(
        query=args.query,
        top_k=args.top_k,
        max_contexts=args.max_contexts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )


def main() -> int:
    """CLI entry point.

    Pipeline construction is intentionally left to the application
    bootstrap that will be added in later sprints.
    """
    parser = build_parser()
    parser.parse_args()

    raise NotImplementedError(
        "Pipeline bootstrap will be implemented in a later sprint."
    )


if __name__ == "__main__":
    raise SystemExit(main())
