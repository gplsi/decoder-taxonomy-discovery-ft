
# MatOnto · Taxonomy discovery with decoder-only models

Fine-tuning and inference for **LLMs4OL 2025 Task C** on the **MatOnto materials-science ontology**. Given a directed child–parent pair, the model predicts whether the child is a subclass of the parent.

This implementation adapts the [IRIS team's original code](https://github.com/AFigaro/LLMs4OL_2025), specifically its [MatOnto Task C pipeline](https://github.com/AFigaro/LLMs4OL_2025/tree/main/LLMs4OL/src/TaskC/matonto), to **Llama and Gemma**. Please credit the original authors and cite the IRIS paper when using this work.

## Files

| File | Purpose |
| --- | --- |
| [matonto_train_and_infer_configurable_llama.py](matonto_train_and_infer_configurable_llama.py) | Fine-tune Llama, select the best checkpoint, calibrate the decision threshold, and predict test relations. |
| [matonto_train_and_infer_configurable_gemma.py](matonto_train_and_infer_configurable_gemma.py) | Equivalent workflow adapted to Gemma 3, using its native sequence-classification implementation. |
| [candidates_matonto_train.json](candidates_matonto_train.json) | Training-split semantic candidates inherited from IRIS, used to sample negatives. |
| [candidates_matonto_test.json](candidates_matonto_test.json) | Test-split semantic candidates inherited from IRIS, used during inference. |
| [matonto_defs.json](matonto_defs.json) | Concept definitions; corresponds to the original `matonto_types_leads.json`. |
| [evaluation.py](evaluation.py) | Calculate precision, recall, and F1 against reference test pairs. |

## Main changes from IRIS

- **Decoder-only models:** replace `microsoft/deberta-v3-large` with Llama 3.2–3B or Gemma 3–4B–PT. Both use a binary classification head and fine-tune the language-model parameters. Despite the instruction-style input, predictions come from classification logits rather than generated answers. Gemma uses `Gemma3ForSequenceClassification` with text inputs; its unused vision components are frozen.
- **Stronger determinism controls:** seed Python, NumPy, PyTorch, and CUDA; sort children and parent lists before splitting and constructing examples; use a dedicated seeded generator for sampling; set training and data seeds; and use zero DataLoader workers. Enable strict deterministic PyTorch algorithms, configure cuBLAS, disable cuDNN benchmarking and TF32, and use eager attention with Flash and memory-efficient SDP disabled.
- **Input format:** replace IRIS's `[CHILD]`/`[PARENT]` format with an instruction describing the parent–child decision, optionally including concept definitions.
- **Training configuration:** use 10 epochs, a learning rate of `1e-5`, bfloat16, gradient accumulation, and a per-device training batch size of 1. Add label smoothing of `0.1` and cap the positive-class loss weight at 10. Both scripts use eight gradient accumulation steps. Gradient checkpointing is currently disabled; the Gemma script includes distributed execution support.

The core IRIS workflow is retained: split by child concept, sample negatives, train a weighted binary classifier, select the checkpoint with the highest development F1, tune a probability threshold on development data, and infer test relations. Both scripts sample **up to nine negatives per child**, excluding that child and its annotated direct parents.

## Configuration and execution

Configure constants near the top of each script before running:

| Setting | Purpose |
| --- | --- |
| `MODEL_ID` | Base model identifier or compatible local checkpoint, including one obtained through continual pretraining. |
| `USE_DEFS` | Include concept definitions. Currently `False` for both models (128-token inputs); enabling definitions uses 512-token inputs. |
| `NEG_SRC` | `"candidates"` for semantic candidates, or `"random"` for random sampling. Both currently default to `"candidates"`. |
| `PAIRS_JSON`, `TRAIN_TYPES`, `TEST_TYPES` | Paths to the original MatOnto training pairs and train/test concept lists. Supply these benchmark files separately. |
| `DEFS_JSON` | Set to `matonto_defs.json` when definitions are enabled; the Llama script already uses this name, while Gemma currently references `matonto_types_leads.json`. |
| `OUT_DIR`, `PRED_JS` | Model and prediction output locations. Use distinct locations for each experiment. |

Run from this directory with the required data paths configured:

```bash
PYTHONHASHSEED=42 python matonto_train_and_infer_configurable_llama.py
# Or, in the Gemma environment:
PYTHONHASHSEED=42 python matonto_train_and_infer_configurable_gemma.py
```

Dependencies include PyTorch, Transformers, Accelerate, Datasets, NumPy, SciPy, scikit-learn, and tqdm. The Gemma script specifies `transformers>=5.2,<6` and uses the newer `processing_class` Trainer API; the Llama script uses the older `tokenizer` API. Use and record compatible dependency versions for each script. Configure Hugging Face authentication if the selected checkpoint requires access.

Determinism controls target repeatability within a fixed environment. Record library and CUDA versions, hardware, GPU count, and launch configuration; identical results across different environments are not guaranteed.

## Outputs and evaluation

Each training script saves the selected model and tokenizer, `best_threshold.txt`, and a prediction JSON containing `parent`, `child`, and `score` fields. Development and internal-test partitions each contain approximately 10% of training children. Threshold calibration searches from `0.05` to `0.95` in steps of `0.01`.

With `NEG_SRC="candidates"`, official-test inference evaluates the supplied candidate lists. With `"random"`, it evaluates sampled test concepts. Neither mode exhaustively evaluates all possible pairs.

Before running the evaluator, set its reference JSON, prediction JSON, output filename, and `output_dir` to your local paths:

```bash
python evaluation.py
```

The evaluator matches case-insensitive directed pairs. Correct matches are counted as unique pairs, while precision and recall denominators use the original list lengths; duplicate records therefore affect the reported scores.
