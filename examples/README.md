# Example

The sample document in this directory is intentionally small and is suitable
for a first end-to-end CLI run.

```bash
python -m src.app.cli   --document examples/sample_document.txt   --query "What does retrieval-augmented generation do?"   --device cpu   --show-contexts
```

The first run may download the configured Hugging Face models.
