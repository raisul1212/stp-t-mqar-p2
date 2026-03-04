"""
MQAR Benchmark Config — STP-T White Paper v3.0
RISE Lab, Purdue University  |  March 2026

════════════════════════════════════════════════════════════════════════════════
ARCHITECTURE: Standard Zoology "Based Recipe" Hybrid Block
════════════════════════════════════════════════════════════════════════════════

Every model uses the identical block structure from Based (Arora et al. 2024):

    TransformerBlock(
        LN → [BaseConv(l_max) || sequence_mixer(d_model)]  → residual
        Identity state_mixer  (no FFN — matches Zoology published configs)
    )

Only the sequence_mixer slot differs between models.

════════════════════════════════════════════════════════════════════════════════
CONFIGURATION: run_configs.csv
════════════════════════════════════════════════════════════════════════════════

ALL run parameters (model, d_model, lr, batch_size, stp hyperparams, ...) come
from run_configs.csv, NOT from hardcoded constants here.  Edit that file to
change the sweep — no Python changes needed.

CSV path resolution order:
  1. RUN_CONFIG env var  (set this to use a custom path)
  2. Same directory as this file (run_configs.csv)

════════════════════════════════════════════════════════════════════════════════
DATA: Matches Phase 1 published setup exactly
════════════════════════════════════════════════════════════════════════════════

  module    : zoology.data.multiquery_ar.MQARConfig   (same as P1)
  vocab_size: 8192  (fixed)
  Train     : varying seq_len per kv difficulty (matches P1 exactly):
               kv=4  → seq_len=64,  100K examples
               kv=8  → seq_len=128, 20K examples
               kv=16 → seq_len=256, 20K examples
               kv=32 → seq_len=256, 20K examples
               kv=64 → seq_len=256, 20K examples
  Test      : stress-test at longer seq_len per kv (7 difficulties)
  batch_size: (train_bs, train_bs // 8)  ← tuple required by Zoology DataConfig

Note: batch_size in CSV = TRAIN batch. Test batch auto-set to 1/8.
      Reduce to 32 if OOM on <40GB GPU at d=256 (P1 OOM was at batch_size=256).

════════════════════════════════════════════════════════════════════════════════
FILTERS
════════════════════════════════════════════════════════════════════════════════

  STP_MODELS=stp_light,stp_t   → only those model rows from CSV
  RUN_CONFIG=/path/to/file.csv → use a custom CSV
  Both can be combined.

════════════════════════════════════════════════════════════════════════════════
ZOOLOGY API NOTES (from P1 debugging)
════════════════════════════════════════════════════════════════════════════════

  • TrainConfig uses run_id (not run_name) + slice_keys for per-KV wandb tracking
  • LoggerConfig() needs pydantic v2 patch (setup.sh handles this)
  • return_embeddings must be False in model.py (setup.sh handles this)
  • BaseConv is at zoology.mixers.base_conv.BaseConv (underscore, not convolution)
  • state_mixer must be Identity (not BaseConv) to match P1 published configs;
    BaseConv goes into the sequence_mixer slot via Hybrid wrapper
  • fla classes take hidden_size=, not d_model= → use fla_wrappers.py

DEPENDENCY ON ZOOLOGY INTERNALS:
  • zoology.mixers.hybrid.Hybrid  — wraps [BaseConv, sequence_mixer] in one call
  • zoology.mixers.base_conv.BaseConv  — short causal conv
  • These are stable across Zoology versions used in P1
"""

import csv
import os
import sys
import uuid
from pathlib import Path

from zoology.config import (
    TrainConfig, ModelConfig, DataConfig, LoggerConfig, ModuleConfig
)
from zoology.data.multiquery_ar import MQARConfig


# ── CSV resolution ────────────────────────────────────────────────────────────

_HERE = Path(__file__).parent
_DEFAULT_CSV = _HERE / "run_configs.csv"
_CSV_PATH = Path(os.environ.get("RUN_CONFIG", str(_DEFAULT_CSV)))

if not _CSV_PATH.exists():
    print(f"[mqar_p2] ERROR: run_configs.csv not found at {_CSV_PATH}", file=sys.stderr)
    print(f"[mqar_p2]   Set RUN_CONFIG=/path/to/run_configs.csv", file=sys.stderr)
    sys.exit(1)

# ── Model filter ──────────────────────────────────────────────────────────────

_ACTIVE = set(os.environ.get("STP_MODELS", "").split(",")) - {""}
def _include(name: str) -> bool:
    return (not _ACTIVE) or (name in _ACTIVE)

# ── Read CSV ──────────────────────────────────────────────────────────────────

def _load_csv(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        reader = csv.DictReader(row for row in f if not row.lstrip().startswith("#"))
        for row in reader:
            if row.get("enabled", "1").strip() != "1":
                continue
            rows.append({k.strip(): v.strip() for k, v in row.items()})
    return rows

_ALL_ROWS = _load_csv(_CSV_PATH)
print(f"[mqar_p2] Loaded {len(_ALL_ROWS)} enabled rows from {_CSV_PATH.name}", file=sys.stderr)

# ── Fixed constants (not per-run tunable) ─────────────────────────────────────

VOCAB_SIZE  = 8192
SWEEP_NAME  = "stp-t-mqar-p2-" + uuid.uuid4().hex[:6]

# ── Data: exact P1 schema ─────────────────────────────────────────────────────
# seq_len MUST scale with kv_pairs — using a single flat seq_len=512 for all
# difficulties would mismatch P1 published data and inflate easy-kv accuracy.
# batch_size tuple: (train_bs, test_bs) — Zoology DataConfig requires this.

# max_seq_len across all test configs (drives l_max for BaseConv / positional emb)
INPUT_SEQ_LEN_MAX = 1024   # from P1: test kv=256 uses seq_len=1024

_TRAIN_CONFIGS = [
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64,  num_examples=100_000, num_kv_pairs=4),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128, num_examples=20_000,  num_kv_pairs=8),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000,  num_kv_pairs=16),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000,  num_kv_pairs=32),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000,  num_kv_pairs=64),
]
_TEST_CONFIGS = [
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64,   num_examples=1_000, num_kv_pairs=4),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64,   num_examples=1_000, num_kv_pairs=8),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64,   num_examples=1_000, num_kv_pairs=16),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128,  num_examples=1_000, num_kv_pairs=32),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256,  num_examples=1_000, num_kv_pairs=64),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=512,  num_examples=1_000, num_kv_pairs=128),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=1024, num_examples=1_000, num_kv_pairs=256),
]

def _make_data(train_batch_size: int) -> DataConfig:
    # Zoology requires a tuple (train_bs, test_bs) for DataConfig.batch_size
    # when train and test configs have different seq_lens.
    # test_bs = train_bs // 8 mirrors P1 exactly: (256, 32)→here: (bs, bs//8)
    return DataConfig(
        train_configs=_TRAIN_CONFIGS,
        test_configs=_TEST_CONFIGS,
        batch_size=(train_batch_size, max(1, train_batch_size // 8)),
        cache_dir="/workspace/zoology_cache",
    )

# ── Hybrid mixer builder ──────────────────────────────────────────────────────
# Zoology's Hybrid(configs=[conv_cfg, seq_cfg]) runs BaseConv first, then the
# sequence mixer, and sums their outputs.  This is the "Based recipe" applied
# to every model for fair comparison (per Zoology README, P1 configs).
#
# BaseConv path: zoology.mixers.base_conv.BaseConv  ← confirmed from P1 configs
# NOT zoology.mixers.convolution.BaseConv (that was a P2 draft error).

def _baseconv_cfg(l_max: int) -> dict:
    return {
        "name": "zoology.mixers.base_conv.BaseConv",
        "kwargs": {
            "l_max": l_max,
            "kernel_size": 3,
            "implicit_long_conv": True,
        },
    }

def _hybrid_mixer(seq_cfg: dict, l_max: int) -> ModuleConfig:
    return ModuleConfig(
        name="zoology.mixers.hybrid.Hybrid",
        kwargs={"configs": [_baseconv_cfg(l_max), seq_cfg]},
    )

# ── Per-model sequence mixer config dicts ─────────────────────────────────────

def _seq_cfg_for(row: dict) -> dict:
    """Return the inner sequence mixer config dict for a given CSV row."""
    model   = row["model"]
    n_heads = int(row["num_heads"])

    if model == "attention":
        return {
            "name": "zoology.mixers.attention.MHA",
            "kwargs": {"num_heads": n_heads},
        }
    elif model == "based":
        # Based uses its own head count (num_heads=1, num_key_value_heads=1)
        # matching Zoology's add_based factory from models_repo.py.
        # The CSV num_heads column is ignored for Based — these are hardcoded
        # to match the published Zoology MQAR config exactly.
        return {
            "name": "zoology.mixers.based.Based",
            "kwargs": {
                "l_max": INPUT_SEQ_LEN_MAX,
                "feature_dim": 16,
                "feature_name": "taylor_exp",
                "num_key_value_heads": 1,
                "num_heads": 1,
                "train_view": "quadratic",
            },
        }
    elif model == "retnet":
        # fla_wrappers.RetNetWrapper bridges fla's hidden_size= API to Zoology's d_model= API.
        # We do NOT use "fla.layers.MultiScaleRetention" directly because fla takes
        # hidden_size as first positional arg while Zoology passes d_model positionally.
        return {
            "name": "fla_wrappers.RetNetWrapper",
            "kwargs": {"num_heads": n_heads, "mode": "fused_chunk"},
        }
    elif model == "gla":
        return {
            "name": "fla_wrappers.GLAWrapper",
            "kwargs": {
                "num_heads":       n_heads,
                "mode":            "chunk",
                "expand_k":        0.5,
                "expand_v":        1.0,
                "use_output_gate": True,
            },
        }
    elif model == "stp_light":
        return {
            "name": "stp_v3.STPTLight",
            "kwargs": {
                "num_heads":    n_heads,
                "gamma_init":   float(row.get("gamma_init",  0.9)),
                "lambda_init":  float(row.get("lambda_init", 0.1)),
            },
        }
    elif model == "stp_t":
        return {
            "name": "stp_v3.STPT",
            "kwargs": {
                "num_heads":    n_heads,
                "gamma_init":   float(row.get("gamma_init",  0.9)),
                "lambda_init":  float(row.get("lambda_init", 0.1)),
            },
        }
    else:
        raise ValueError(f"[mqar_p2] Unknown model '{model}' in CSV row: {row}")


# ── TrainConfig builder ───────────────────────────────────────────────────────

def _make_train_config(row: dict) -> TrainConfig:
    model      = row["model"]
    d_model    = int(row["d_model"])
    n_layers   = int(row["n_layers"])
    lr         = float(row["lr"])
    max_epochs = int(row["max_epochs"])
    batch_size = int(row["batch_size"])
    run_id     = row["run_id"]

    # max_position_embeddings = 0 for ALL models, matching P1 and Zoology's
    # published MQAR configs.  MHA on MQAR does not need positional embeddings
    # (the task is retrieval by content, not position), and giving attention
    # pos-emb that baselines lack would create an unfair comparison.
    max_pos_emb = 0

    seq_cfg = _seq_cfg_for(row)
    hybrid  = _hybrid_mixer(seq_cfg, l_max=INPUT_SEQ_LEN_MAX)

    return TrainConfig(
        model=ModelConfig(
            d_model=d_model,
            n_layers=n_layers,
            max_position_embeddings=max_pos_emb,
            vocab_size=VOCAB_SIZE,
            sequence_mixer=hybrid,
            # state_mixer = Identity matches P1 and Zoology published configs.
            # FFN is NOT added here — the Hybrid wrapper already adds BaseConv.
            state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
        ),
        data=_make_data(batch_size),
        learning_rate=lr,
        max_epochs=max_epochs,
        # run_id format matches P1 extract_results.py regex: model-dN-lrX.Xe-NN
        run_id=run_id,
        sweep_id=SWEEP_NAME,
        # slice_keys enables per-KV accuracy tracking in wandb (critical for paper)
        slice_keys=["num_kv_pairs"],
        logger=LoggerConfig(),
    )


# ── Build configs list ────────────────────────────────────────────────────────

configs = []
skipped = []

for row in _ALL_ROWS:
    model = row.get("model", "")
    if not _include(model):
        skipped.append(row["run_id"])
        continue
    try:
        configs.append(_make_train_config(row))
    except Exception as e:
        print(f"[mqar_p2] ERROR building config for {row.get('run_id', '?')}: {e}",
              file=sys.stderr)
        raise

# ── Summary ──────────────────────────────────────────────────────────────────

print(f"[mqar_p2] Sweep: {SWEEP_NAME}", file=sys.stderr)
print(f"[mqar_p2] Configs: {len(configs)} active, {len(skipped)} skipped by STP_MODELS filter",
      file=sys.stderr)

if configs:
    models_in_sweep = sorted(set(c.run_id.split("-")[0] for c in configs))
    print(f"[mqar_p2] Models in sweep: {', '.join(models_in_sweep)}", file=sys.stderr)

if not configs:
    print("[mqar_p2] WARNING: No configs generated. Check STP_MODELS and CSV enabled column.",
          file=sys.stderr)