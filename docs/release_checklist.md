# Release Checklist — v1.0.0

Use this checklist before creating the first stable release.

## Code quality

- [ ] `python scripts/final_qa.py` completes successfully.
- [ ] The full test suite passes.
- [ ] No Git conflict markers remain.
- [ ] No unfinished rebase or merge is active.
- [ ] The working tree is clean after the release commit.

## Functional coverage

- [ ] PDF ingestion works.
- [ ] Text ingestion works.
- [ ] Recursive chunking works.
- [ ] Sentence Transformer embeddings work.
- [ ] FAISS indexing and retrieval work.
- [ ] FAISS save/load round trip works.
- [ ] Prompt construction works.
- [ ] Hugging Face generation works.
- [ ] RAGPipeline returns answer, prompt, and contexts.
- [ ] Exact Match, Token F1, ROUGE-L, and Evaluator work.
- [ ] Typed configuration loading works.
- [ ] Production CLI parses and executes correctly.

## Documentation

- [ ] `README.md` describes the project and architecture.
- [ ] `docs/quickstart.md` is current.
- [ ] `examples/README.md` is current.
- [ ] The example command uses valid paths and arguments.
- [ ] Installation dependencies are documented.

## Git and release

- [ ] Sprint 18 changes are committed.
- [ ] The branch is pushed successfully.
- [ ] The release branch or `main` contains the final commit.
- [ ] Tag `v1.0.0` is created.
- [ ] Tag `v1.0.0` is pushed.
- [ ] GitHub Release notes summarize features and test status.

## Suggested release commands

```bash
python scripts/final_qa.py

git status
git add scripts/final_qa.py docs/release_checklist.md VERSION
git commit -m "Finalize project for v1.0.0"
git push origin feature/vector-store

git tag -a v1.0.0 -m "Research Copilot RAG v1.0.0"
git push origin v1.0.0
```
