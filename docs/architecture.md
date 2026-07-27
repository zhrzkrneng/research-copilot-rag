
# ResearchCopilot Architecture

## Goal

Build an AI Research Assistant capable of:

- Loading scientific papers
- Loading source code
- Chunking documents
- Building embeddings
- Retrieving relevant context
- Answering research questions
- Comparing papers
- Explaining source code
- Generating summaries

---

## High-Level Pipeline

Document
    ↓
Loader
    ↓
Chunker
    ↓
Embedding
    ↓
Vector Store
    ↓
Retriever
    ↓
LLM
    ↓
Answer
