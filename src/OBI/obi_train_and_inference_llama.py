# ───────────────────────────────────────────────────────────────────────────
# One file → four variants
#   USE_DEFS:     include type definitions or not
#   NEG_SRC :     "candidates"  (Top-100 semantic list)  or  "random"
# ───────────────────────────────────────────────────────────────────────────

# ╭─ choose your variant here ──────────────────────────────────────────────╮
USE_DEFS      = True               # True = add definitions, False = raw strings
NEG_SRC       = "candidates"        # "candidates" or "random"
NEG_PER_CHILD = 9 if NEG_SRC == "candidates" else 9
# ╰──────────────────────────────────────────────────────────────────────────╯

# ═══════════ CONFIGURACIÓN DETERMINISMO ULTRA ESTRICTA ══════════════════════
SEED = 42 #42
import os
# Para fijar también el hash: PYTHONHASHSEED=42 python obi_determinista.py
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["MKL_NUM_THREADS"]        = "1"
os.environ["NUMEXPR_NUM_THREADS"]    = "1"
import json, math, re, warnings
import numpy as np
import random
import torch
import torch.nn as nn
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32    = False
# FIX: deshabilitar Memory Efficient Attention (no tiene implementación determinista en CUDA)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)   # math SDP sí es determinista
torch.use_deterministic_algorithms(True, warn_only=False)  # FIX: warn_only=False, estricto
random_gen = random.Random(SEED)
np_gen     = np.random.RandomState(SEED)
import os, json, random, math, numpy as np
# Simple multi-GPU configuration without DDP
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings
import torch, torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, TrainingArguments, Trainer)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from scipy.special import softmax

# ───────── PATHS ──────────────────────────────────────────────────────────
ROOT         = Path("LLMs4OL")
# To run the script add original data files (2025) to LLMs4OL folder
PAIRS_JSON = Path(r'obi_train_pairs.json')
TRAIN_TYPES  = Path(r'obi_train_types.txt')
TEST_TYPES   = Path(r'obi_test_types.txt')
DEFS_JSON    = Path(r'obi_types_leads.json')
CAND_TRAIN   = Path(r'candidates_obi_train.json')
CAND_TEST    = Path(r'candidates_obi_test.json')
TAG       = f"obi_{'def' if USE_DEFS else 'str'}_{NEG_SRC}"
OUT_DIR   = Path(rf'ckpt_llama_{TAG}')
PRED_JS   = Path(rf'pred_edges_{TAG}.json')
MODEL_ID  = "meta-llama/Llama-3.2-3B"
#

# ───────── HYPERPARAMS ────────────────────────────────────────────────────
SEED       = 42
BATCH_SIZE = 1  # Reduced batch size for stability
EPOCHS     = 10  # Keep original epochs
LR         = 2e-5  # Lower learning rate for stability
MAX_LEN    = 512 if USE_DEFS else 128  # Reduced max length
DEV_FR, TEST_FR = 0.10, 0.10
INF_BATCH  = 32  # Reduced inference batch
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────── LOAD DATA ──────────────────────────────────────────────────────
pairs        = json.loads(PAIRS_JSON.read_text("utf-8"))
train_types  = [t.strip() for t in TRAIN_TYPES.read_text("utf-8").splitlines() if t.strip()]
test_types   = [t.strip() for t in TEST_TYPES.read_text("utf-8").splitlines() if t.strip()]
defs_map     = json.loads(DEFS_JSON.read_text("utf-8")) if USE_DEFS else {}
cand_train   = json.loads(CAND_TRAIN.read_text("utf-8")) if NEG_SRC=="candidates" else {}
cand_test    = json.loads(CAND_TEST.read_text("utf-8"))  if NEG_SRC=="candidates" else {}
# gold: child → lista de padres ordenada y sin duplicados
gold = {}
for e in pairs:
    gold.setdefault(e["child"], set()).add(e["parent"])
# Orden estable, independiente del hash del proceso.
gold = {child: sorted(parents) for child, parents in gold.items()}

# ───────── SPLIT CHILDREN ─────────────────────────────────────────────────
children = sorted(gold.keys())
random_gen.shuffle(children)
n_tot = len(children); n_dev = math.floor(DEV_FR*n_tot); n_tst = math.floor(TEST_FR*n_tot)
dev_children   = set(children[:n_dev])
test_ch_train  = set(children[n_dev:n_dev+n_tst])      # held-out inside train corpus
train_children = set(children[n_dev+n_tst:])
print(f"[INFO] split train/dev/test children = "
      f"{len(train_children)}/{len(dev_children)}/{len(test_ch_train)}")

# ───────── PROMPT MAKER ───────────────────────────────────────────────────

def prompt(c, p):
    if USE_DEFS:
        return (f"Determine if there is a parent-child relationship between these concepts:\n\n"
               f"Child: {c}\nDefinition: {defs_map.get(c,'')}\n\n"
               f"Parent: {p}\nDefinition: {defs_map.get(p,'')}\n\n"
               f"Answer with 'yes' if {p} is a parent of {c}, 'no' otherwise.")
    else:
        return (f"Determine if there is a parent-child relationship between these concepts:\n\n"
               f"Child: {c}\nParent: {p}\n\n"
               f"Answer with 'yes' if {p} is a parent of {c}, 'no' otherwise.")

# ───────── DATASET BUILDER ────────────────────────────────────────────────

def build(child_set, tag):
    data = []
    for c in sorted(child_set):
        pos = gold[c]
        if NEG_SRC == "candidates":
            pool = [t for t in cand_train.get(c, []) if t not in pos and t != c]
        else:
            pool = [t for t in train_types if t not in pos and t != c]
        negs = random_gen.sample(pool, min(NEG_PER_CHILD, len(pool)))
        data += [{"text": prompt(c, p), "label": 1} for p in pos]
        data += [{"text": prompt(c, p), "label": 0} for p in negs]
    print(f"[{tag}] examples = {len(data):,}")
    return data
train_data = build(train_children, "train")
dev_data   = build(dev_children,   "dev")
test_data  = build(test_ch_train,  "test")

# ───────── TOKENISE ───────────────────────────────────────────────────────
try:
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True,
                                        use_auth_token="")
    print(f"✓ Fast tokenizer cargado desde: {MODEL_ID}")
except Exception as e:
    print(f"✗ Error al cargar tokenizer: {e}")
    base_model = "meta-llama/Llama-3.2-3B"
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True,
                                        use_auth_token="")
    print(f"✓ Tokenizer fallback cargado desde: {base_model}")

if tok.pad_token is None:
    tok.pad_token = tok.eos_token

def enc(b): 
    return tok(b["text"], truncation=True, max_length=MAX_LEN, padding=False)
train_ds = Dataset.from_list(train_data).map(enc, batched=True, remove_columns=["text"])
dev_ds   = Dataset.from_list(dev_data  ).map(enc, batched=True, remove_columns=["text"])
test_ds  = Dataset.from_list(test_data ).map(enc, batched=True, remove_columns=["text"])
collator = DataCollatorWithPadding(tok)

# ───────── MODEL & TRAINER ────────────────────────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, 
    num_labels=2,
    torch_dtype=torch.bfloat16,  # Use bfloat16 instead of float16
    attn_implementation="eager"
)
# Disable cache to avoid issues with multiple GPUs
model.config.use_cache = False
# Properly set pad token
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
model.config.pad_token_id = tok.pad_token_id
model.resize_token_embeddings(len(tok))
# Use multiple GPUs without DDP - let Transformers handle it
print(f"[INFO] Available GPUs: {torch.cuda.device_count()}")
if torch.cuda.device_count() > 1:
    print(f"[INFO] Will use {torch.cuda.device_count()} GPUs automatically")
else:
    print(f"[INFO] Using single GPU: {torch.cuda.get_device_name(0)}")
pos = sum(e["label"] for e in train_data); neg = len(train_data) - pos
# Cap the positive weight to avoid extreme imbalance
pos_w = min(neg / pos, 10.0) if pos > 0 else 1.0
print(f"[INFO] Class weights: negative=1.0, positive={pos_w:.2f}")

class WTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **_):
        labels = inputs.pop("labels")
        outs = model(**inputs)
        # Use label smoothing to prevent overconfident predictions
        weights = torch.tensor([1.0, pos_w], device=outs.logits.device, dtype=outs.logits.dtype)
        loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
        loss = loss_fn(outs.logits, labels)
        return (loss, outs) if return_outputs else loss

def metrics(pred):
    logits, labels = pred
    preds = logits.argmax(-1)
    p, r, f, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"accuracy": accuracy_score(labels, preds), "precision": p, "recall": r, "f1": f}
args = TrainingArguments(
    output_dir=str(OUT_DIR),
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=max(1, BATCH_SIZE // 2),
    gradient_accumulation_steps=4,
    num_train_epochs=EPOCHS,
    warmup_steps=100,
    weight_decay=0.01,
    bf16=True,
    fp16=False,
    seed=SEED,
    data_seed=SEED,
    torch_compile=False,
    optim="adamw_torch",
    adam_epsilon=1e-8,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    report_to="none",
    save_total_limit=1,
    gradient_checkpointing=False,
    dataloader_pin_memory=False,
    max_grad_norm=1.0,
    eval_accumulation_steps=None,
    remove_unused_columns=False,
    dataloader_num_workers=0,
    dataloader_persistent_workers=False,
    eval_on_start=False,
    # Important: Do NOT set any DDP settings for simple multi-GPU
    local_rank=-1,  # Disable distributed training detection
    prediction_loss_only=False,
)
trainer = WTrainer(model=model, args=args,
                   train_dataset=train_ds, eval_dataset=dev_ds,
                   tokenizer=tok, data_collator=collator,
                   compute_metrics=metrics)
trainer.train()

# ───────── SHOW BEST EPOCH INFO ───────────────────────────────────────────
print(f"[INFO] Best model checkpoint: {trainer.state.best_model_checkpoint}")
if hasattr(trainer.state, 'best_metric') and trainer.state.best_metric is not None:
    print(f"[INFO] Best metric: {trainer.state.best_metric:.4f}")
model.eval()
model.save_pretrained(OUT_DIR)
tok.save_pretrained(OUT_DIR)

# ───────── τ tuning & save ────────────────────────────────────────────────
# Fix BFloat16 issue in predictions
dev_predictions = np.asarray(trainer.predict(dev_ds).predictions, dtype=np.float32)
dev_probs = softmax(dev_predictions, axis=-1)[:, 1]
best_tau, _ = max(
    ((thr, precision_recall_fscore_support(dev_ds["label"],
                                           (dev_probs >= thr).astype(int),
                                           average="binary", zero_division=0)[2])
     for thr in np.arange(0.05, 0.951, 0.01)),
    key=lambda x: x[1]
)
(OUT_DIR / "best_threshold.txt").write_text(f"{best_tau:.4f}\n", "utf-8")
print(f"[DEV] best τ = {best_tau:.2f}")

# ───────── internal test metrics ──────────────────────────────────────────
if len(test_ds) > 0:
    # Fix BFloat16 issue in test predictions
    test_predictions = np.asarray(trainer.predict(test_ds).predictions, dtype=np.float32)
    probs_test = softmax(test_predictions, axis=-1)[:, 1]
    test_pred = (probs_test >= best_tau).astype(int)
    p, r, f, _ = precision_recall_fscore_support(test_ds["label"], test_pred, average="binary", zero_division=0)
    print(f"[TEST internal] F1={f:.3f}  P={p:.3f}  R={r:.3f}")
else:
    print("[TEST internal] No test data available")

# ───────── OFFICIAL TEST INFERENCE ────────────────────────────────────────
print("[INFO] inference on official OBI test children …")
pred_edges, texts, pairs = [], [], []

def flush():
    if not texts:
        return
    with torch.no_grad():
        # Direct tokenization without chat template  
        inputs = tok(texts, padding=True, truncation=True, max_length=MAX_LEN,
                    return_tensors="pt").to(device)
        # Get predictions and fix BFloat16 issue
        outputs = model(**inputs)
        logits = outputs.logits
        # Convert BFloat16 to float32 before calling .numpy()
        logits_float = logits.float().detach().cpu().numpy()
        probs = softmax(logits_float, axis=-1)[:, 1]
    for (c, p), s in zip(pairs, probs):
        if s >= best_tau:
            pred_edges.append({"parent": p, "child": c, "score": float(s)})
    texts.clear(); pairs.clear()
for c in tqdm(test_types, unit="child"):
    if NEG_SRC == "candidates":
        parents = cand_test.get(c, [])
    else:  # random negatives sampled from *test* types
        pool = [t for t in test_types if t != c]
        parents = random_gen.sample(pool, min(NEG_PER_CHILD, len(pool)))
    for p in parents:
        texts.append(prompt(c, p)); pairs.append((c, p))
        if len(texts) >= INF_BATCH:
            flush()
flush()

# ───────── SAVE PREDICTIONS ───────────────────────────────────────────────
PRED_JS.parent.mkdir(parents=True, exist_ok=True)
PRED_JS.write_text(json.dumps(pred_edges, indent=2, ensure_ascii=False), "utf-8")
print(f"[DONE] {len(pred_edges):,} edges saved → {PRED_JS}")
