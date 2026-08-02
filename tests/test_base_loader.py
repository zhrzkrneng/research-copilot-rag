from src.ingestion.base_loader import BaseLoader


def test_base_loader_is_abstract():
    assert hasattr(BaseLoader, "load")
