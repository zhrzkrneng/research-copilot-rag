"""Application layer exports."""

from __future__ import annotations

from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Lazily invoke the CLI without preloading ``src.app.cli``."""
    from .cli import main as cli_main

    return cli_main(argv)


__all__ = [
    "main",
]
