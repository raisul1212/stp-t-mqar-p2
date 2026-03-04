# DEPRECATED — use scripts/extract_results.py instead.
# collect_results.py reads only WandB offline directories and has no
# best-epoch tracking, no per-run JSON support, and no comparison table.
# extract_results.py (copied from Phase 1) handles all three result sources
# (per-run JSONs, WandB, log file) and produces summary.csv + comparison_table.txt.
#
# This file is kept only for reference. Do not call it directly.

#!/usr/bin/env python3
"""
collect_results.py — Aggregate WandB offline runs into CSV
RISE Lab, Purdue University

Reads all wandb offline runs in the zoology/ directory and emits:
  results/summary.csv       — best accuracy per run
  results/per_kv.csv        — per-KV-difficulty accuracy matrix

Usage:
    python3 scripts/collect_results.py [--wandb-dir /workspace/zoology/wandb]
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

# ── Parse args ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--wandb-dir", default="/workspace/zoology/wandb",
                    help="Path to wandb/ directory inside zoology workspace")
parser.add_argument("--out-dir", default="results",
                    help="Directory for output CSVs")
args = parser.parse_args()

WANDB_DIR = Path(args.wandb_dir)
OUT_DIR   = Path(args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not WANDB_DIR.exists():
    print(f"[ERROR] wandb dir not found: {WANDB_DIR}")
    sys.exit(1)

# ── Scan runs ─────────────────────────────────────────────────────────────────
KV_VALUES = [4, 8, 16, 32, 64, 128, 256]

runs_data = []

for run_dir in sorted(WANDB_DIR.glob("offline-run-*")):
    files_dir = run_dir / "files"
    if not files_dir.exists():
        continue

    # Load run config
    config_path = files_dir / "config.yaml"
    if not config_path.exists():
        continue

    # Load wandb-summary.json for final metrics
    summary_path = files_dir / "wandb-summary.json"
    if not summary_path.exists():
        continue

    try:
        with open(summary_path) as f:
            summary = json.load(f)
    except Exception as e:
        print(f"[WARN] Could not load {summary_path}: {e}")
        continue

    # Parse run name from config.yaml (look for run_name field)
    run_name = run_dir.name
    try:
        with open(config_path) as f:
            for line in f:
                if "run_name" in line and "value:" in line:
                    run_name = line.split("value:")[-1].strip().strip('"').strip("'")
                    break
    except Exception:
        pass

    # Parse model family and dimension from run_name
    # e.g. "stp_t_d256_lr1e-03", "retnet_d128_lr3e-04"
    m_model = re.match(r"^([a-z_]+?)_d(\d+)_lr(.+)$", run_name)
    model_family = m_model.group(1) if m_model else "unknown"
    d_model      = int(m_model.group(2)) if m_model else -1
    lr_str       = m_model.group(3) if m_model else "?"

    # Extract metrics
    valid_acc     = summary.get("valid/accuracy", None)
    train_loss    = summary.get("train/loss",     None)
    best_epoch    = summary.get("_step",          None)

    kv_accs = {}
    for kv in KV_VALUES:
        key = f"valid/num_kv_pairs/accuracy-{kv}"
        kv_accs[kv] = summary.get(key, None)

    runs_data.append({
        "run_name":     run_name,
        "model":        model_family,
        "d_model":      d_model,
        "lr":           lr_str,
        "valid_acc":    valid_acc,
        "train_loss":   train_loss,
        "best_step":    best_epoch,
        **{f"acc_kv{kv}": kv_accs[kv] for kv in KV_VALUES},
    })

if not runs_data:
    print("[WARN] No runs found. Have you run the benchmark yet?")
    sys.exit(0)

print(f"[collect] Found {len(runs_data)} runs.")

# ── Write summary.csv ────────────────────────────────────────────────────────
summary_path = OUT_DIR / "summary.csv"
fieldnames = ["run_name", "model", "d_model", "lr", "valid_acc", "train_loss", "best_step"]
with open(summary_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    for row in sorted(runs_data, key=lambda r: (r["model"], r["d_model"], r["lr"])):
        w.writerow(row)
print(f"[collect] Written: {summary_path}")

# ── Write per_kv.csv (best run per model × d_model) ──────────────────────────
# Best = highest valid_acc across LR sweep
from collections import defaultdict
best = {}
for row in runs_data:
    key = (row["model"], row["d_model"])
    cur = best.get(key)
    if cur is None or (row["valid_acc"] or -1) > (cur["valid_acc"] or -1):
        best[key] = row

per_kv_path = OUT_DIR / "per_kv.csv"
kv_fields   = [f"acc_kv{kv}" for kv in KV_VALUES]
fieldnames2 = ["model", "d_model", "valid_acc", "best_lr"] + kv_fields
with open(per_kv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames2, extrasaction="ignore")
    w.writeheader()
    for (model, d), row in sorted(best.items()):
        row2 = {k: row[k] for k in fieldnames2 if k != "best_lr"}
        row2["best_lr"] = row["lr"]
        row2["model"]   = model
        row2["d_model"] = d
        w.writerow(row2)
print(f"[collect] Written: {per_kv_path}")

# ── Print table ──────────────────────────────────────────────────────────────
print("\n── Best accuracy by model × d_model ──────────────────────")
print(f"{'Model':<15} {'d_model':<8} {'valid_acc':<12} {'best_lr':<10}")
print("-" * 48)
for (model, d), row in sorted(best.items()):
    acc = f"{row['valid_acc']:.4f}" if row["valid_acc"] is not None else "N/A"
    print(f"{model:<15} {d:<8} {acc:<12} {row['lr']:<10}")
