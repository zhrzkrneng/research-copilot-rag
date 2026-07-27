
from dataclasses import dataclass, field


@dataclass(slots=True)
class Chunk:
    """
    Represents a chunk of a document.
    """

    text: str
    chunk_id: int
    source: str
    metadata: dict = field(default_factory=dict)
