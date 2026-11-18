# Decoder-taxonomy-discovery-ft

**Fine-tuning decoder-only language models for taxonomy discovery on MatOnto and OBI, adapted from IRIS.**

This repository contains the supervised fine-tuning, inference, and evaluation code used for **LLMs4OL 2025 Task C** in two domains: materials science (**MatOnto**) and biomedicine (**OBI**). Given a directed child–parent pair, a binary classifier predicts whether the child is a subclass of the parent.

The implementation builds on the [IRIS team's LLMs4OL 2025 solution](https://github.com/AFigaro/LLMs4OL_2025), adapting its DeBERTa-based pipeline to **Llama 3.2–3B** and **Gemma 3–4B–PT**, with additional controls for deterministic execution. It supports experiments with base models and compatible checkpoints obtained through continual pretraining (CPT).

## Code and data

| Domain | Training and inference | Evaluation | Documentation | Data |
| --- | --- | --- | --- | --- |
| MatOnto | [Llama](src/MatOnto/matonto_train_and_infer_configurable_llama.py) · [Gemma](src/MatOnto/matonto_train_and_infer_configurable_gemma.py) | [Evaluator](src/MatOnto/evaluation.py) | [MatOnto README](src/MatOnto/readme.md) | [Train/test files](data/MatOnto/) |
| OBI | [Llama](src/OBI/obi_train_and_inference_llama.py) · [Gemma](src/OBI/obi_train_and_inference_gemma.py) | [Evaluator](src/OBI/evaluation.py) | [OBI README](src/OBI/readme.md) | [Train/test files](data/OBI/) |

Each source directory also contains semantic candidate lists inherited from IRIS and concept definitions. The `data/` directories contain training pairs, concept lists, and test gold references.

## Workflow

1. **Build supervised examples:** use annotated child–parent relations as positives and sample negatives from semantic candidates or the concept list. Split training children into train, development, and internal-test partitions.
2. **Fine-tune:** train the language model and binary classification head, optionally including concept definitions in the input. Select the checkpoint with the highest development F1.
3. **Calibrate and infer:** tune the positive-class threshold on development data, then predict relations for official-test children using the configured candidate source.
4. **Evaluate:** compare predicted relations against the test reference and report precision, recall, and F1.

Predictions are obtained from classification logits. Inference uses candidate lists or random samples according to configuration; it does not enumerate all possible concept pairs.

## Main changes from IRIS

- **Decoder-only backbones:** replace DeBERTa with Llama or Gemma and adapt tokenization, padding, and model loading for binary sequence classification.
- **Determinism controls:** use fixed seeds, stable ordering of children and parents, dedicated sampling generators, deterministic PyTorch algorithms, and explicit CUDA/attention settings.
- **Input and training adjustments:** introduce an instruction-style pair template and revise precision, optimization, gradient accumulation, and loss settings for these models. Exact settings are documented in each domain directory.

## Running experiments

Install compatible versions of PyTorch, Transformers, Accelerate, Datasets, NumPy, SciPy, scikit-learn, and tqdm. Follow the domain README for model-specific requirements; the Llama and Gemma scripts use different Transformers APIs.

Before running, configure the constants in the selected script: model/checkpoint, data paths, definition usage, negative source, and output locations. Paths are resolved relative to the working directory. Point training inputs to the files under `data/` and each evaluator to the corresponding `*_test_gold.json`; some defaults retain earlier filenames or local paths.

For example, after configuring paths, run from the relevant source directory:

```bash
cd src/MatOnto
PYTHONHASHSEED=42 python matonto_train_and_infer_configurable_llama.py
```

Set distinct output locations for each model/checkpoint and record dependency versions, hardware, and GPU count. Deterministic settings target repeatability within a fixed environment.

## Related repositories and attribution

- [IRIS / LLMs4OL 2025](https://github.com/AFigaro/LLMs4OL_2025): original implementation and candidate-generation pipeline. Credit the original authors and cite their paper when using this adaptation.
- [OntoSelect4CPT](https://github.com/gplsi/OntoSelect4CPT): ontology-corpus characterization, metric–F1 association analysis, and corpus scoring for data selection.
- [LMTK — deterministic branch](https://github.com/gplsi/LMTK/tree/deterministic): CPT code used to produce the ontology-adapted checkpoints evaluated through this fine-tuning workflow.
