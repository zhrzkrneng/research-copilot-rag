# Quickstart

This guide shows how to install, test, and run Research Copilot RAG.

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

For PDF ingestion and FAISS retrieval, the environment must include:

```bash
pip install PyMuPDF faiss-cpu
```

## 2. Run the test suite

From the repository root:

```bash
pytest -q
```

## 3. Run the CLI

Use a PDF or plain-text document:

```bash
python -m src.app.cli   --document examples/sample_document.txt   --query "What is retrieval-augmented generation?"
```

Useful options:

```text
--top-k
--max-contexts
--max-new-tokens
--temperature
--chunk-size
--chunk-overlap
--embedding-model
--generation-model
--device
--include-scores
--show-contexts
--show-prompt
```

Example with additional output:

```bash
python -m src.app.cli   --document examples/sample_document.txt   --query "Why does RAG improve factual grounding?"   --top-k 3   --max-contexts 2   --show-contexts   --show-prompt
```

## 4. Use a configuration file

The default project configuration is stored at:

```text
configs/default.yaml
```

The typed configuration API can be used from Python:

```python
from src.config import load_config

config = load_config("configs/default.yaml")
pipeline_config = config.to_pipeline_config()
```

## 5. Save and restore a FAISS index

```python
from src.retrieval import FAISSVectorStore

store = FAISSVectorStore(dimension=384)
store.save("outputs/index")

restored = FAISSVectorStore(dimension=384)
restored.load("outputs/index")
```

The persistence directory contains:

```text
index.faiss
chunks.json
store.json
```

## 6. Core architecture

```text
Document
  ↓
Loader
  ↓
RecursiveSplitter
  ↓
SentenceTransformerEmbedder
  ↓
FAISSVectorStore
  ↓
DenseRetriever
  ↓
PromptBuilder
  ↓
HFGenerator
  ↓
RAGResponse
```

## 7. Notes

The first CLI execution may download embedding and generation model weights.
For CPU-only environments, use:

```bash
--device cpu
```

For CUDA-enabled environments, use:

```bash
--device cuda
```
