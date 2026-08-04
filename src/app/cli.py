"""Command-line interface for the Research Copilot RAG project."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from textwrap import fill
from typing import Sequence

from src.app.bootstrap import PipelineConfig, build_indexed_pipeline
from src.rag import RAGPipeline, RAGResponse


DEFAULT_VERSION = "1.0.0"
MIN_TERMINAL_WIDTH = 60
MAX_TERMINAL_WIDTH = 100


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="research-copilot-rag",
        description=(
            "Index a PDF or text document and answer a question "
            "with retrieval-augmented generation."
        ),
    )

    parser.add_argument(
        "--document",
        type=Path,
        required=True,
        help="Path to a .pdf or .txt document.",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Question to answer from the document.",
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
        help="Maximum number of contexts included in the prompt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Document chunk size in characters.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=64,
        help="Character overlap between adjacent chunks.",
    )
    parser.add_argument(
        "--embedding-model",
        default=PipelineConfig().embedding_model_name,
        help="Sentence Transformers embedding model.",
    )
    parser.add_argument(
        "--generation-model",
        default=PipelineConfig().generation_model_name,
        help="Hugging Face causal language model.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Execution device such as "auto", "cpu", or "cuda".',
    )
    parser.add_argument(
        "--include-scores",
        action="store_true",
        help="Include retrieval scores in the generated prompt.",
    )
    parser.add_argument(
        "--show-contexts",
        action="store_true",
        help="Print retrieved contexts after the answer.",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print the final prompt after the answer.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable the Phase 19 presentation banner and progress messages.",
    )

    return parser


def build_config(args: argparse.Namespace) -> PipelineConfig:
    """Create a pipeline configuration from parsed CLI arguments."""
    return PipelineConfig(
        embedding_model_name=args.embedding_model,
        generation_model_name=args.generation_model,
        device=args.device,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        include_scores=args.include_scores,
    )


def run_pipeline(
    pipeline: RAGPipeline,
    args: argparse.Namespace,
) -> RAGResponse:
    """Execute the configured RAG pipeline."""
    return pipeline.run(
        query=args.query,
        top_k=args.top_k,
        max_contexts=args.max_contexts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )


def terminal_width() -> int:
    """Return a safe width for terminal presentation."""
    width = shutil.get_terminal_size(
        fallback=(80, 24),
    ).columns

    return max(
        MIN_TERMINAL_WIDTH,
        min(width, MAX_TERMINAL_WIDTH),
    )


def read_project_version(
    repository_root: Path | None = None,
) -> str:
    """Read the project version from VERSION when available."""
    root = repository_root or Path.cwd()
    version_path = root / "VERSION"

    if not version_path.is_file():
        return DEFAULT_VERSION

    version = version_path.read_text(
        encoding="utf-8",
    ).strip()

    return version or DEFAULT_VERSION


def print_banner(
    *,
    version: str | None = None,
) -> None:
    """Print the Phase 19 CLI banner."""
    width = terminal_width()
    resolved_version = version or read_project_version()
    title = "Research Copilot RAG"
    subtitle = f"v{resolved_version}"

    print()
    print("═" * width)
    print(title.center(width))
    print(subtitle.center(width))
    print("═" * width)
    print()


def print_step(
    index: int,
    total: int,
    message: str,
) -> None:
    """Print one numbered execution step."""
    print(f"[{index}/{total}] {message}")


def print_success(message: str) -> None:
    """Print one successful step result."""
    print(f"✓ {message}")


def print_section(title: str) -> None:
    """Print a visible terminal section heading."""
    width = terminal_width()

    print()
    print("─" * width)
    print(title)
    print("─" * width)


def format_response(
    response: RAGResponse,
    *,
    show_contexts: bool = False,
    show_prompt: bool = False,
) -> str:
    """Format a RAG response for terminal output.

    The default output intentionally preserves the original
    ``Answer:\n...`` format for backwards compatibility.
    """
    sections = [
        "Answer:",
        response.answer,
    ]

    if show_contexts:
        sections.extend(
            [
                "",
                "Retrieved contexts:",
            ]
        )

        for index, result in enumerate(
            response.contexts,
            start=1,
        ):
            source = result.chunk.source
            sections.extend(
                [
                    "",
                    (
                        f"[{index}] source={source} "
                        f"score={result.score:.6f}"
                    ),
                    result.chunk.text,
                ]
            )

    if show_prompt:
        sections.extend(
            [
                "",
                "Prompt:",
                response.prompt,
            ]
        )

    return "\n".join(sections)


def format_professional_response(
    response: RAGResponse,
    *,
    show_contexts: bool = False,
    show_prompt: bool = False,
) -> str:
    """Format a response using the Phase 19 terminal presentation."""
    width = terminal_width()
    answer_width = max(40, width - 2)

    sections = [
        "─" * width,
        "Answer",
        "─" * width,
        fill(
            response.answer,
            width=answer_width,
            replace_whitespace=False,
        ),
    ]

    if show_contexts:
        sections.extend(
            [
                "",
                "─" * width,
                "Retrieved contexts",
                "─" * width,
            ]
        )

        for index, result in enumerate(
            response.contexts,
            start=1,
        ):
            source = result.chunk.source
            header = (
                f"[{index}] source={source} "
                f"score={result.score:.6f}"
            )

            sections.extend(
                [
                    "",
                    header,
                    fill(
                        result.chunk.text,
                        width=answer_width,
                        replace_whitespace=False,
                    ),
                ]
            )

    if show_prompt:
        sections.extend(
            [
                "",
                "─" * width,
                "Prompt",
                "─" * width,
                response.prompt,
            ]
        )

    return "\n".join(sections)


def print_summary(
    *,
    args: argparse.Namespace,
    response: RAGResponse,
) -> None:
    """Print a compact successful-execution summary."""
    print_section("Summary")
    print(f"Document   : {args.document.name}")
    print(f"Query      : {args.query.strip()}")
    print(f"Contexts   : {len(response.contexts)}")
    print(f"Top-k      : {args.top_k}")
    print(f"Embedder   : {args.embedding_model}")
    print(f"Generator  : {args.generation_model}")
    print(f"Device     : {args.device}")
    print("Status     : SUCCESS")
    print()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line application."""
    parser = build_parser()
    args = parser.parse_args(argv)
    quiet = args.quiet

    try:
        if not quiet:
            print_banner()
            print_step(
                1,
                3,
                "Loading the document and building the retrieval index...",
            )

        config = build_config(args)
        pipeline = build_indexed_pipeline(
            args.document,
            config=config,
        )

        if not quiet:
            print_success(
                f"Indexed document: {args.document.name}"
            )
            print_step(
                2,
                3,
                "Retrieving evidence and generating the answer...",
            )

        response = run_pipeline(
            pipeline,
            args,
        )

        if not quiet:
            print_success(
                f"Retrieved {len(response.contexts)} context(s)"
            )
            print_step(
                3,
                3,
                "Formatting the final response...",
            )
            print_success("Response ready")
    except (
        FileNotFoundError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as exc:
        print(
            f"Error: {exc}",
            file=sys.stderr,
        )
        return 1

    if quiet:
        print(
            format_response(
                response,
                show_contexts=args.show_contexts,
                show_prompt=args.show_prompt,
            )
        )
    else:
        print()
        print(
            format_professional_response(
                response,
                show_contexts=args.show_contexts,
                show_prompt=args.show_prompt,
            )
        )
        print_summary(
            args=args,
            response=response,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
