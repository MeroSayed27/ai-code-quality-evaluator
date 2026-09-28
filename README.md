# AI Code Quality Evaluator

Automated evaluator for AI-generated code: detects bugs, scores correctness, and benchmarks competing solutions to the same coding task. Built as a small-scale version of the exact evaluation work behind training and auditing AI coding agents (Claude Code, Cursor, Copilot, Codex) — the same category of task used by AI-training platforms like Mindrift, Alignerr, and Mercor's SWE-Bench auditing programs.

## Why this project

AI coding agents now write a huge share of production code. Automated tests catch the obvious failures, but subtler issues — off-by-one errors, wrong comparison operators, missing edge cases, silently swapped variables — often slip through. Someone (or something) has to review the code a model produces and decide: is this actually correct? This project builds that evaluation pipeline end to end, starting from data and finishing at a queryable API.

## Architecture

```
Mutation-Based Dataset  →  AST Feature Extraction  →  PyTorch Classifier  →  FastAPI Evaluation Service
     (Phase 1 ✅)              (Phase 2)                  (Phase 2)              (Phase 3)
```

1. **Dataset generation** — seed algorithms + mutation testing produce labeled correct/buggy code pairs
2. **Feature extraction** — parse each snippet's AST into structural features (branch count, loop depth, comparison operators used, etc.)
3. **Classifier** — a small feedforward neural network (PyTorch) trained on those features to predict correctness
4. **Service** — a FastAPI endpoint that takes AI-generated code, runs it through the pipeline, and returns a score + flags — with an optional LLM call for qualitative review

## Status

- [x] **Phase 1 — Dataset generator** (this repo, ready to run)
- [ ] Phase 2 — AST feature extraction + PyTorch classifier
- [ ] Phase 3 — FastAPI service + LLM qualitative review layer

## Phase 1: Dataset Generator

Uses **mutation testing** — a real, established software-testing technique — to programmatically inject common real-world bug patterns into a set of 16 seed algorithms (binary search, bubble sort, palindrome check, GCD, and others), producing labeled correct/buggy pairs.

Bug patterns covered: off-by-one errors, comparison-operator flips, boundary-condition removal, wrong arithmetic/logical operators, swapped loop variables, dropped increments.

### Setup

```bash
git clone https://github.com/MeroSayed27/ai-code-quality-evaluator.git
cd ai-code-quality-evaluator
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run

```bash
cd data
python generate_dataset.py
```

Produces `data/code_quality_dataset.jsonl` — one labeled example per line:

```json
{"id": 1, "function_name": "is_palindrome", "code": "def is_palindrome(s):\n    ...\n    return s != s[::-1]", "label": 0, "bug_type": "comparison_flip"}
```

Current dataset: **42 examples** (16 correct, 26 buggy across 6 bug types). Scaling it up is a matter of adding more entries to the `SEEDS` dict in `generate_dataset.py` — the mutation operators apply automatically to any new function added.

## Roadmap

- Expand seed functions to 50+ for a larger training set
- Phase 2: `features/extract_features.py` — AST-based structural feature extraction
- Phase 2: `model/train_nn.py` — PyTorch feedforward classifier trained on extracted features
- Phase 3: `api/main.py` — FastAPI service exposing `/evaluate` for single-snippet scoring and `/compare` for ranking multiple AI-generated solutions to the same prompt

## Author

**Amir M. Sayed** — AI/ML engineer, 3+ years freelance experience in LLM evaluation, RLHF preference ranking, and Python backend development (FastAPI, SQL).
GitHub: [github.com/MeroSayed27](https://github.com/MeroSayed27)
