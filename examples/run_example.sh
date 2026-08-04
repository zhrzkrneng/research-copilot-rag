#!/usr/bin/env bash
set -euo pipefail

python -m src.app.cli   --document examples/sample_document.txt   --query "How does retrieval-augmented generation improve grounding?"   --device cpu   --top-k 3   --max-contexts 2   --show-contexts
