# Adaptado a google/gemma-3-4b-pt (clasificación binaria, texto solamente).
# Dependencias: transformers>=5.2,<6; accelerate; datasets; scikit-learn; scipy; tqdm.
# Autenticación: token configurado en HF_TOKEN, como en el script original.
# Conserva particiones, negativos, prompts, hiperparámetros y calibración originales.
# El determinismo requiere también fijar versiones, hardware y número de procesos.
# PYTHONHASHSEED debe fijarse ANTES de arrancar Python si se necesita fijar su hash.

# ───────────────────────────────────────────────────────────────────────────
# One file → four variants
#   USE_DEFS:     include type definitions or not
#   NEG_SRC :     "candidates"  (Top-100 semantic list)  or  "random"
# ───────────────────────────────────────────────────────────────────────────

# ╭─ choose your variant here ──────────────────────────────────────────────╮
USE_DEFS      = False               # True = add definitions, False = raw strings
NEG_SRC       = "candidates"        # "candidates" or "random"
NEG_PER_CHILD = 9 if NEG_SRC == "candidates" else 9
# ╰──────────────────────────────────────────────────────────────────────────╯

# ═══════════ CONFIGURACIÓN DETERMINISMO ULTRA ESTRICTA ══════════════════════
SEED = 42 #42

import os
os.environ['PYTHONHASHSEED']         = str(SEED)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["MKL_NUM_THREADS"]        = "1"
os.environ["NUMEXPR_NUM_THREADS"]    = "1"

import json, math, re, warnings
import sys
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

from pathlib import Path
from tqdm import tqdm
from datasets import Dataset
from transformers import (AutoTokenizer, Gemma3ForSequenceClassification,
                          DataCollatorWithPadding, TrainingArguments, Trainer)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from scipy.special import softmax

# ───────── PATHS ──────────────────────────────────────────────────────────
ROOT        = Path("LLMs4OL")
PAIRS_JSON  = Path(r'matonto_train_pairs.json')
TRAIN_TYPES = Path(r'matonto_train_types.txt')
TEST_TYPES  = Path(r'matonto_test_types.txt')
DEFS_JSON   = Path(r'matonto_types_leads.json')
CAND_TRAIN  = Path(r'candidates_matonto_train.json')
CAND_TEST   = Path(r'candidates_matonto_test.json')

TAG      = f"matonto_{'def' if USE_DEFS else 'str'}_{NEG_SRC}"
MODEL_ID = "google/gemma-3-4b-pt"
# Para otro experimento, cambia RUN_NAME y/o MODEL_ID por el checkpoint Gemma.
RUN_NAME = "gemma3_4b"
OUT_DIR = CAND_TRAIN.parent / f"ckpt_{RUN_NAME}_{TAG}"
PRED_JS = CAND_TRAIN.parent / f"pred_edges_{RUN_NAME}_{TAG}.json"
HF_TOKEN = ''

# ───────── HYPERPARAMS ────────────────────────────────────────────────────
# BATCH_SIZE = 1
EPOCHS     = 10
# LR         = 2e-5
MAX_LEN    = 512 if USE_DEFS else 128
DEV_FR, TEST_FR = 0.10, 0.10
INF_BATCH  = 32

BATCH_SIZE = 1
# gradient_accumulation_steps = 8
LR = 1e-5
# EPOCHS = 5
# warmup_ratio = 0.06


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DETERMINISM] Using device: {device}, SEED: {SEED}")

# ───────── LOAD DATA ──────────────────────────────────────────────────────
raw_pairs   = json.loads(PAIRS_JSON.read_text("utf-8"))
train_types = [t.strip() for t in TRAIN_TYPES.read_text("utf-8").splitlines() if t.strip()]
test_types  = [t.strip() for t in TEST_TYPES.read_text("utf-8").splitlines() if t.strip()]

defs_map   = json.loads(DEFS_JSON.read_text("utf-8"))  if USE_DEFS              else {}
cand_train = json.loads(CAND_TRAIN.read_text("utf-8")) if NEG_SRC == "candidates" else {}
cand_test  = json.loads(CAND_TEST.read_text("utf-8"))  if NEG_SRC == "candidates" else {}

# gold: child → lista ordenada de parents (lista, no set → orden estable)
gold: dict[str, list[str]] = {}
for e in raw_pairs:
    child, parent = e["child"], e["parent"]
    if child not in gold:
        gold[child] = []
    if parent not in gold[child]:
        gold[child].append(parent)
for child in gold:
    gold[child] = sorted(gold[child])

# ───────── SPLIT CHILDREN ─────────────────────────────────────────────────
children = sorted(gold.keys())   # orden léxico fijo antes del shuffle
random_gen.shuffle(children)

n_tot = len(children)
n_dev = math.floor(DEV_FR  * n_tot)
n_tst = math.floor(TEST_FR * n_tot)
dev_children   = set(children[:n_dev])
test_ch_train  = set(children[n_dev:n_dev + n_tst])
train_children = set(children[n_dev + n_tst:])

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
    for c in sorted(child_set):   # sorted() sobre set para orden estable
        pos = gold[c]              # lista ya ordenada
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
# Sin fallback a LLaMA: el vocabulario debe ser el del checkpoint Gemma.
tok = AutoTokenizer.from_pretrained(
    MODEL_ID, use_fast=True, token=HF_TOKEN, padding_side="right",
)
if tok.pad_token_id is None:
    raise ValueError("El tokenizador de Gemma debe incluir su token <pad>.")
print(f"[INFO] Tokenizer: {MODEL_ID}; pad_token_id={tok.pad_token_id}")

def enc(b):
    encoded = tok(
        b["text"],
        truncation=True,
        max_length=MAX_LEN,
        padding=False,
    )
    encoded["token_type_ids"] = [
        [0] * len(ids) for ids in encoded["input_ids"]
    ]
    return encoded

train_ds = Dataset.from_list(train_data).map(enc, batched=True, remove_columns=["text"])
dev_ds   = Dataset.from_list(dev_data  ).map(enc, batched=True, remove_columns=["text"])
test_ds  = Dataset.from_list(test_data ).map(enc, batched=True, remove_columns=["text"])
collator = DataCollatorWithPadding(tok)

# ───────── MODEL & TRAINER ────────────────────────────────────────────────
# Gemma 3 de 4B tiene configuración multimodal. La clase nativa acepta texto
# y añade una cabeza de clasificación de dos clases, inicializada desde cero.
# No usar GemmaForSequenceClassification (corresponde a otra arquitectura).
model = Gemma3ForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=2,
    dtype=torch.bfloat16,
    token=HF_TOKEN,
    attn_implementation="eager",
    id2label={0: "no_relation", 1: "parent_child"},
    label2id={"no_relation": 0, "parent_child": 1},
)
model.config.pad_token_id = tok.pad_token_id
model.config.text_config.pad_token_id = tok.pad_token_id
model.config.use_cache = False
model.config.text_config.use_cache = False
# Sin imágenes, estos parámetros no participan en el cálculo de la pérdida.
# Se ajustan todos los parámetros del modelo de lenguaje y de la cabeza.
for name, parameter in model.named_parameters():
    if "vision_tower" in name or "multi_modal_projector" in name:
        parameter.requires_grad_(False)
print(f"[INFO] Classifier: {type(model).__name__}")
# No se añaden tokens ni se redimensionan los embeddings.

n_pos = sum(e["label"] for e in train_data)
n_neg = len(train_data) - n_pos
pos_w = min(n_neg / n_pos, 10.0) if n_pos > 0 else 1.0
print(f"[INFO] Class weights: negative=1.0, positive={pos_w:.2f}")

class WTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **_):
        labels = inputs["labels"]
        outs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        weights = torch.tensor([1.0, pos_w],
                               device=outs.logits.device,
                               dtype=torch.float32)
        loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
        loss = loss_fn(outs.logits.float(), labels)
        return (loss, outs) if return_outputs else loss

def metrics(pred):
    logits, labels = pred
    preds = logits.argmax(-1)
    p, r, f, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0)
    return {"accuracy": accuracy_score(labels, preds),
            "precision": p, "recall": r, "f1": f}

args = TrainingArguments(
    output_dir=str(OUT_DIR),
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=8, ### antes 1
    num_train_epochs=EPOCHS,
    warmup_steps=100,
    warmup_ratio=0.06, ### nuevo
    weight_decay=0.01,
    bf16=True,
    fp16=False,
    seed=SEED,
    data_seed=SEED,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none",
    save_total_limit=1,
    gradient_checkpointing=False,
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
    dataloader_persistent_workers=False,
    max_grad_norm=1.0,
    eval_accumulation_steps=None,
    torch_compile=False,
    ddp_find_unused_parameters=True,
    optim="adamw_torch",
    adam_epsilon=1e-8,
    remove_unused_columns=False,
)

trainer = WTrainer(
    model=model, args=args,
    train_dataset=train_ds, eval_dataset=dev_ds,
    processing_class=tok, data_collator=collator,
    compute_metrics=metrics
)

# Pérdida media propia: Trainer gestiona la acumulación de gradientes.
trainer.model_accepts_loss_kwargs = False

# ══════════ VERIFICACIÓN FINAL DE DETERMINISMO ══════════════════════════════
print(f"[DETERMINISM CHECK] PyTorch version: {torch.__version__}")
print(f"[DETERMINISM CHECK] CUDA deterministic: {torch.backends.cudnn.deterministic}")
print(f"[DETERMINISM CHECK] CUDA benchmark: {torch.backends.cudnn.benchmark}")
print(f"[DETERMINISM CHECK] TF32 disabled: {not torch.backends.cuda.matmul.allow_tf32}")
print(f"[DETERMINISM CHECK] Flash SDP: {torch.backends.cuda.flash_sdp_enabled()}")
print(f"[DETERMINISM CHECK] MemEfficient SDP: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
print(f"[DETERMINISM CHECK] Math SDP: {torch.backends.cuda.math_sdp_enabled()}")
print(f"[DETERMINISM CHECK] Random seed set to: {SEED}")
print(f"[DETERMINISM CHECK] DataLoader workers: {args.dataloader_num_workers}")
print("[DETERMINISM CHECK] Starting deterministic training...")

trainer.train()

# ───────── SHOW BEST EPOCH INFO ───────────────────────────────────────────
print(f"[INFO] Best model checkpoint: {trainer.state.best_model_checkpoint}")
print(f"[INFO] Best metric (F1): {trainer.state.best_metric:.4f}")
if trainer.state.best_model_checkpoint:
    match = re.search(r"checkpoint-(\d+)", trainer.state.best_model_checkpoint)
    if match:
        best_step = int(match.group(1))
        for record in trainer.state.log_history:
            if record.get("step") == best_step and "eval_f1" in record:
                print(f"[INFO] Best epoch: {record.get('epoch')}")
                break

trainer.save_model(str(OUT_DIR))
if trainer.is_world_process_zero():
    tok.save_pretrained(OUT_DIR)

# ───────── τ tuning & save ────────────────────────────────────────────────
dev_probs = softmax(trainer.predict(dev_ds).predictions, axis=-1)[:, 1]

best_tau, _ = max(
    ((thr, precision_recall_fscore_support(
        dev_ds["label"],
        (dev_probs >= thr).astype(int),
        average="binary", zero_division=0)[2])
     for thr in np.linspace(0.05, 0.95, 91)),
    key=lambda x: x[1]
)
if trainer.is_world_process_zero():
    (OUT_DIR / "best_threshold.txt").write_text(f"{best_tau:.4f}\n", "utf-8")
print(f"[DEV] best τ = {best_tau:.4f}")

# ───────── internal test metrics ──────────────────────────────────────────
if len(test_ds) > 0:
    probs_test = softmax(trainer.predict(test_ds).predictions, axis=-1)[:, 1]
    test_pred  = (probs_test >= best_tau).astype(int)
    p, r, f, _ = precision_recall_fscore_support(
        test_ds["label"], test_pred, average="binary", zero_division=0)
    print(f"[TEST internal] F1={f:.3f}  P={p:.3f}  R={r:.3f}")
else:
    print("[TEST internal] No test data available")

# ───────── OFFICIAL TEST INFERENCE ────────────────────────────────────────
# Todos los procesos completan trainer.predict antes de la inferencia manual.
trainer.accelerator.wait_for_everyone()
if not trainer.is_world_process_zero():
    trainer.accelerator.wait_for_everyone()
    sys.exit(0)
model = trainer.accelerator.unwrap_model(trainer.model)
model.eval()

print("[INFO] inference on official MatOnto test children …")
pred_edges, infer_texts, infer_pairs = [], [], []

def flush():
    if not infer_texts:
        return
    with torch.no_grad():
        inputs = tok(infer_texts, padding=True, truncation=True,
                     max_length=MAX_LEN, return_tensors="pt").to(next(model.parameters()).device)
        probs = softmax(model(**inputs).logits.float().cpu().numpy(), axis=-1)[:, 1]
    for (c, p), s in zip(infer_pairs, probs):
        if s >= best_tau:
            pred_edges.append({"parent": p, "child": c, "score": float(s)})
    infer_texts.clear()
    infer_pairs.clear()

for c in tqdm(test_types, unit="child"):
    if NEG_SRC == "candidates":
        parents = cand_test.get(c, [])
    else:
        pool    = [t for t in test_types if t != c]
        parents = random_gen.sample(pool, min(NEG_PER_CHILD, len(pool)))
    for p in parents:
        infer_texts.append(prompt(c, p))
        infer_pairs.append((c, p))
        if len(infer_texts) >= INF_BATCH:
            flush()
flush()

# ───────── SAVE PREDICTIONS ───────────────────────────────────────────────
PRED_JS.parent.mkdir(parents=True, exist_ok=True)
PRED_JS.write_text(json.dumps(pred_edges, indent=2, ensure_ascii=False), "utf-8")
print(f"[DONE] {len(pred_edges):,} edges saved → {PRED_JS}")
trainer.accelerator.wait_for_everyone()
