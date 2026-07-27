# ResearchCopilot RAG

A retrieval-augmented research assistant for scientific text and source code question answering.

## Project Overview

ResearchCopilot is an NLP-based assistant designed to analyse scientific documents and their associated source code.

The system retrieves relevant evidence from scientific text and code, then uses a language model to generate grounded answers with source references.

## Planned Features

- Scientific text question answering
- Source code question answering
- Text and code chunking
- Semantic retrieval with FAISS
- Retrieval-Augmented Generation
- Fine-tuning with QLoRA
- Evidence and source citation
- Paper and code comparison
- Evaluation of retrieval and generated answers
- Interactive Gradio interface
The project is currently under development.

## Planned Pipeline

```text
Scientific Text + Source Code
              |
              v
         Preprocessing
              |
              v
           Chunking
              |
              v
          Embeddings
              |
              v
         FAISS Index
              |
              v
          Retrieval
              |
              v
       Fine-tuned LLM
              |
              v
   Answer + Evidence + Sources

##Technologies
Python
PyTorch
Hugging Face Transformers
Sentence Transformers
FAISS
PEFT
QLoRA
Gradio

##Author: zohreh zakeran
