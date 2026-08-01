"""Recursive character-based document splitter."""

from collections.abc import Sequence

from src.core.document import Document
from src.preprocessing.base_splitter import BaseSplitter
from src.preprocessing.chunk import Chunk


class RecursiveSplitter(BaseSplitter):
    """Split documents recursively while preserving natural boundaries."""

    DEFAULT_SEPARATORS: tuple[str, ...] = (
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    )

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: Sequence[str] | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        chosen_separators = tuple(
            separators or self.DEFAULT_SEPARATORS
        )

        if not chosen_separators:
            raise ValueError("At least one separator is required.")

        if chosen_separators[-1] != "":
            chosen_separators = (*chosen_separators, "")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = chosen_separators

    def split(self, document: Document) -> list[Chunk]:
        """Split a document into ordered chunks."""
        text = document.text

        if not text or not text.strip():
            return []

        atomic_parts = self._recursive_split(
            text=text,
            separators=self.separators,
        )

        chunk_texts = self._merge_parts(atomic_parts)

        chunks: list[Chunk] = []
        search_start = 0

        for chunk_id, chunk_text in enumerate(chunk_texts):
            start = text.find(chunk_text, search_start)

            if start == -1:
                start = max(
                    0,
                    search_start - self.chunk_overlap,
                )

            end = start + len(chunk_text)

            metadata = dict(document.metadata)
            metadata.update(
                {
                    "start": start,
                    "end": end,
                    "length": len(chunk_text),
                }
            )

            chunks.append(
                Chunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    source=document.source,
                    metadata=metadata,
                )
            )

            search_start = max(
                start + 1,
                end - self.chunk_overlap,
            )

        return chunks

    def _recursive_split(
        self,
        text: str,
        separators: Sequence[str],
    ) -> list[str]:
        """Recursively split text until each part fits the size limit."""
        if len(text) <= self.chunk_size:
            return [text]

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            return [
                text[index : index + self.chunk_size]
                for index in range(
                    0,
                    len(text),
                    self.chunk_size,
                )
            ]

        raw_parts = text.split(separator)

        if len(raw_parts) == 1:
            return self._recursive_split(
                text=text,
                separators=remaining_separators,
            )

        parts: list[str] = []

        for index, raw_part in enumerate(raw_parts):
            if not raw_part:
                continue

            part = raw_part

            if index < len(raw_parts) - 1:
                part += separator

            if len(part) <= self.chunk_size:
                parts.append(part)
            else:
                parts.extend(
                    self._recursive_split(
                        text=part,
                        separators=remaining_separators,
                    )
                )

        return parts

    def _merge_parts(
        self,
        parts: Sequence[str],
    ) -> list[str]:
        """Merge smaller units into overlapping chunks."""
        chunks: list[str] = []
        current = ""

        for part in parts:
            if not part:
                continue

            candidate = current + part

            if len(candidate) <= self.chunk_size:
                current = candidate
                continue

            if current.strip():
                chunks.append(current.strip())

            overlap_text = (
                current[-self.chunk_overlap :]
                if self.chunk_overlap and current
                else ""
            )

            current = overlap_text + part

            while len(current) > self.chunk_size:
                chunks.append(
                    current[: self.chunk_size].strip()
                )

                step = self.chunk_size - self.chunk_overlap
                current = current[step:]

        if current.strip():
            chunks.append(current.strip())

        return [
            chunk
            for chunk in chunks
            if chunk
        ]
