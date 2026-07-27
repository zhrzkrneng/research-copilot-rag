"""
Core document data model.
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class Document:
    """
    Represents a document loaded into the system.
    """

    text: str
    source: str
    file_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
