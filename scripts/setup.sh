#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════════
# setup.sh — STP-T MQAR Phase 2 environment setup
# RISE Lab, Purdue University  |  March 2026
#
# DESIGN PRINCIPLE (inherited from P1):
#   RunPod pods ship with a matched PyTorch stack. We NEVER let pip modify it.
#   All third-party packages are installed with --no-deps where they could
#   pull in a conflicting torch version.
#
# What this script does:
#   1.  GPU + PyTorch version check
#   2.  System packages (git, tmux)
#   3.  Safe Python deps (einops, wandb, pyyaml, etc.)
#   4.  flash-linear-attention with --no-deps + BitNet conflict fix
#   5.  Zoology clone + install (--no-deps)
#   6.  Zoology patches: return_embeddings=False, pydantic v2 LoggerConfig,
#       BaseConv implicit_long_conv, fla head_first compatibility
#   7.  Register mixer directory on sys.path (.pth file)
#   8.  Copy stp_train.py (checkpoint + JSON results + best-epoch tracking)
#   9.  Full verification (imports, forward passes, logits shape, patch checks)
# ════════════════════════════════════════════════════════════════════════════════
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="${WORKSPACE}/stp-t-mqar-p2"
ZOOLOGY_DIR="${WORKSPACE}/zoology"
MIXER_DIR="${REPO_DIR}/mixers"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}⚠${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; }

echo "════════════════════════════════════════════════════════════════"
echo " STP-T MQAR Phase 2 Setup"
echo " Workspace: ${WORKSPACE}"
echo "════════════════════════════════════════════════════════════════"

# ── 1. GPU check ─────────────────────────────────────────────────────────────
echo "[1/9] GPU check..."
if ! nvidia-smi &>/dev/null; then fail "No GPU detected"; exit 1; fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
ok "GPU: ${GPU_NAME} (${GPU_MEM})"

# Record torch version — must not change after installs
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "none")
if [ "${TORCH_VER}" = "none" ]; then
    fail "PyTorch not found. Use a PyTorch pod template."; exit 1
fi
ok "PyTorch ${TORCH_VER} (pod-installed, will not modify)"

# ── 2. System packages ───────────────────────────────────────────────────────
echo "[2/9] System packages..."
apt-get update -qq && apt-get install -y -qq git tmux >/dev/null 2>&1 && ok "git, tmux" || warn "apt failed"

# ── 3. Safe Python deps ──────────────────────────────────────────────────────
echo "[3/9] Python dependencies..."

# Pure Python packages — safe to install normally
for pkg in einops wandb pyyaml pandas tqdm rich opt-einsum scipy; do
    pymod="${pkg//-/_}"
    python3 -c "import ${pymod}" 2>/dev/null || \
        pip install "${pkg}" --break-system-packages -q 2>/dev/null
done

# pydantic v2 — install without deps to avoid pulling new torch
python3 -c "import pydantic" 2>/dev/null || \
    pip install pydantic --break-system-packages -q --no-deps 2>/dev/null

# transformers — already on most pods; --no-deps if missing
python3 -c "import transformers" 2>/dev/null || \
    pip install transformers --break-system-packages -q --no-deps 2>/dev/null

ok "Python deps"

# ── 4. flash-linear-attention (fla) — with PyTorch guard ────────────────────
echo "[4/9] flash-linear-attention..."
FLA_OK=0

if python3 -c "from fla.layers import GatedLinearAttention, MultiScaleRetention" 2>/dev/null; then
    ok "fla (existing, import OK)"
    FLA_OK=1
fi

if [ "${FLA_OK}" -eq 0 ]; then
    # Install with --no-deps so torch is NEVER touched.
    # fla-core is a separate package required by flash-linear-attention for
    # some sub-modules (CUDA kernels / triton ops).  P1 found both are needed.
    pip install "flash-linear-attention>=0.4.1" \
        --break-system-packages -q --no-deps 2>/dev/null || true
    pip install fla-core \
        --break-system-packages -q --no-deps 2>/dev/null || true

    if python3 -c "from fla.layers import GatedLinearAttention, MultiScaleRetention" 2>/dev/null; then
        ok "fla installed (--no-deps)"
        FLA_OK=1
    else
        # BitNet AutoConfig.register conflict — fix by adding exist_ok=True
        warn "fla import failing — attempting BitNet registration patch"
        FLA_BITNET=$(python3 -c \
            "import fla, os; print(os.path.join(os.path.dirname(fla.__file__), 'models', 'bitnet', '__init__.py'))" \
            2>/dev/null || echo "")
        if [ -n "${FLA_BITNET}" ] && [ -f "${FLA_BITNET}" ]; then
            sed -i 's/\.register(\([^)]*\))/\.register(\1, exist_ok=True)/g' "${FLA_BITNET}"
            sed -i 's/exist_ok=True, exist_ok=True/exist_ok=True/g' "${FLA_BITNET}"
            if python3 -c "from fla.layers import GatedLinearAttention, MultiScaleRetention" 2>/dev/null; then
                ok "fla fixed (patched BitNet registration)"
                FLA_OK=1
            else
                warn "fla still broken — RetNet/GLA baselines will fail at runtime"
            fi
        else
            warn "Cannot locate fla BitNet file — RetNet/GLA baselines may fail"
        fi
    fi
fi

# Guard: verify torch was not changed
TORCH_AFTER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
if [ "${TORCH_VER}" != "${TORCH_AFTER}" ]; then
    fail "PyTorch version changed! (${TORCH_VER} → ${TORCH_AFTER}) — reinstall pod"
    exit 1
fi
ok "PyTorch version preserved: ${TORCH_VER}"

# ── 5. Zoology ───────────────────────────────────────────────────────────────
echo "[5/9] Zoology framework..."
if [ ! -d "${ZOOLOGY_DIR}" ]; then
    echo "  Cloning Zoology..."
    git clone -q https://github.com/HazyResearch/zoology.git "${ZOOLOGY_DIR}"
    ok "Cloned Zoology"
fi
cd "${ZOOLOGY_DIR}"
if ! python3 -c "import zoology" 2>/dev/null; then
    pip install -e . --break-system-packages -q --no-deps 2>/dev/null \
        && ok "Zoology installed (--no-deps)" \
        || { fail "Zoology install failed"; exit 1; }
else
    ok "Zoology (existing)"
fi

# ── 6. Zoology patches ───────────────────────────────────────────────────────
echo "[6/9] Zoology patches..."

# Patch A: return_embeddings=False
# Without this ALL models plateau at ~25% accuracy (embeddings instead of logits)
MODEL_PY="${ZOOLOGY_DIR}/zoology/model.py"
if grep -q "return_embeddings.*=.*True" "${MODEL_PY}" 2>/dev/null; then
    sed -i 's/return_embeddings.*=.*True/return_embeddings: bool = False/g' "${MODEL_PY}"
    ok "Patched: return_embeddings=False"
else
    ok "return_embeddings=False already applied"
fi

# Patch B: pydantic v2 LoggerConfig
# Zoology defines `project_name: str = None` which pydantic v2 rejects with
# "PydanticUserError: A non-annotated attribute was detected".
# Fix: change to Optional[str] = None for both project_name and entity.
PYDANTIC_OK=$(python3 -c "
from zoology.config import LoggerConfig
lc = LoggerConfig()
assert lc.project_name is None and lc.entity is None
print('ok')
" 2>/dev/null || echo "fail")

if [ "${PYDANTIC_OK}" = "ok" ]; then
    ok "pydantic v2 LoggerConfig already OK"
else
    CFG_PY="${ZOOLOGY_DIR}/zoology/config.py"
    # Ensure Optional is imported
    if ! grep "from typing import" "${CFG_PY}" | grep -q "Optional"; then
        sed -i 's/from typing import \(.*\)/from typing import Optional, \1/' "${CFG_PY}"
    fi
    sed -i 's/project_name:.*$/project_name: Optional[str] = None/' "${CFG_PY}"
    sed -i 's/entity:.*$/entity: Optional[str] = None/' "${CFG_PY}"
    PYDANTIC_CHECK=$(python3 -c "
from zoology.config import LoggerConfig
lc = LoggerConfig()
assert lc.project_name is None and lc.entity is None
print('ok')
" 2>/dev/null || echo "fail")
    if [ "${PYDANTIC_CHECK}" = "ok" ]; then
        ok "Patched: pydantic v2 LoggerConfig"
    else
        fail "Could not fix LoggerConfig — check zoology/config.py manually"
        exit 1
    fi
fi

# Patch C: fla head_first compatibility
# fla 0.4.1+ removed the head_first=False argument from GLA, DeltaNet, etc.
# Zoology's built-in fla wrappers still pass it → TypeError at runtime.
echo "  Patching fla head_first compatibility..."
for fla_wrapper in gla.py delta_net.py gated_delta_net.py rwkv7.py; do
    fpath="${ZOOLOGY_DIR}/zoology/mixers/${fla_wrapper}"
    [ ! -f "${fpath}" ] && continue
    if grep -q "head_first" "${fpath}" 2>/dev/null; then
        sed -i '/head_first=False/d' "${fpath}"
        # Remove trailing commas left by deletion: ", )" → ")"
        python3 -c "
import re
t = open('${fpath}').read()
t = re.sub(r',(\s*\))', r'\1', t)
open('${fpath}', 'w').write(t)
" 2>/dev/null || true
        ok "  Fixed: ${fla_wrapper} (removed head_first)"
    fi
done

# ── 7. Register mixer directory on sys.path ──────────────────────────────────
echo "[7/9] Registering mixers on sys.path..."
SITE_PKG=$(python3 -c "import site; print(site.getsitepackages()[0])")
PTH_FILE="${SITE_PKG}/stp_t_p2.pth"
echo "${MIXER_DIR}" > "${PTH_FILE}"
ok "Registered: ${MIXER_DIR}"
ok "  → ${PTH_FILE}"

# ── 8. Install stp_train.py ──────────────────────────────────────────────────
echo "[8/9] Installing stp_train.py..."
if [ -f "${REPO_DIR}/scripts/stp_train.py" ]; then
    if [ ! -f "${ZOOLOGY_DIR}/zoology/train.py.orig" ]; then
        cp "${ZOOLOGY_DIR}/zoology/train.py" "${ZOOLOGY_DIR}/zoology/train.py.orig"
    fi
    cp "${REPO_DIR}/scripts/stp_train.py" "${ZOOLOGY_DIR}/zoology/train.py"
    ok "stp_train.py installed (checkpoints + JSON results + best-epoch tracking)"
else
    warn "stp_train.py not found at ${REPO_DIR}/scripts/stp_train.py"
    warn "Results will use WandB only (no per-run JSON files)"
fi

# Create results and checkpoint directories
mkdir -p "${WORKSPACE}/results/runs"
mkdir -p "${WORKSPACE}/checkpoints"
ok "Directories: results/runs/, checkpoints/"

# ── 9. Full verification ─────────────────────────────────────────────────────
echo "[9/9] Verification..."
python3 << 'PYEOF'
import sys, os, torch

errors = []

# GPU
if not torch.cuda.is_available():
    errors.append("CUDA not available")
else:
    print(f"  ✓ PyTorch {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}")

# Zoology config + pydantic v2 fix
try:
    from zoology.config import TrainConfig, ModelConfig, DataConfig, LoggerConfig, ModuleConfig
    lc = LoggerConfig()
    assert lc.project_name is None and lc.entity is None, "LoggerConfig pydantic fix not applied"
    print("  ✓ zoology.config (pydantic v2 OK)")
except Exception as e:
    errors.append(f"zoology.config: {e}")

# P1 data module (multiquery_ar, not associative_recall)
try:
    from zoology.data.multiquery_ar import MQARConfig
    _ = MQARConfig(vocab_size=8192, input_seq_len=64, num_examples=10, num_kv_pairs=4)
    print("  ✓ zoology.data.multiquery_ar.MQARConfig")
except Exception as e:
    errors.append(f"zoology.data.multiquery_ar: {e}")

# BaseConv path (underscore, not convolution)
try:
    from zoology.mixers.base_conv import BaseConv
    print("  ✓ zoology.mixers.base_conv.BaseConv")
except Exception as e:
    errors.append(f"zoology.mixers.base_conv: {e}")

# Hybrid wrapper
try:
    from zoology.mixers.hybrid import Hybrid
    print("  ✓ zoology.mixers.hybrid.Hybrid")
except Exception as e:
    errors.append(f"zoology.mixers.hybrid: {e}")

# MHA + Based
try:
    from zoology.mixers.attention import MHA
    from zoology.mixers.based import Based
    print("  ✓ zoology.mixers.attention.MHA, based.Based")
except Exception as e:
    errors.append(f"zoology mixers: {e}")

# fla
try:
    from fla.layers import GatedLinearAttention, MultiScaleRetention
    print("  ✓ fla.layers (GatedLinearAttention, MultiScaleRetention)")
except Exception as e:
    errors.append(f"fla.layers: {e}")

# fla_wrappers
try:
    from fla_wrappers import RetNetWrapper, GLAWrapper
    print("  ✓ fla_wrappers (RetNetWrapper, GLAWrapper)")
except Exception as e:
    errors.append(f"fla_wrappers: {e}")

# stp_v3
try:
    from stp_v3 import STPT, STPTLight
    print("  ✓ stp_v3 (STPT, STPTLight)")
except Exception as e:
    errors.append(f"stp_v3: {e}")

# Forward passes (GPU)
if not errors and torch.cuda.is_available():
    try:
        from stp_v3 import STPT, STPTLight
        device = "cuda"
        B, T, D = 2, 32, 64
        x = torch.randn(B, T, D).to(device)
        for cls, name in [(STPTLight, "STPTLight"), (STPT, "STPT")]:
            m = cls(d_model=D, num_heads=2).to(device)
            y = m(x)
            assert y.shape == (B, T, D), f"Shape mismatch: {y.shape}"
            del m
        print(f"  ✓ STP GPU forward passes (B={B}, T={T}, D={D})")
    except Exception as e:
        errors.append(f"STP forward pass: {e}")

# return_embeddings fix (must return logits not embeddings)
try:
    from zoology.model import LanguageModel
    from zoology.config import ModelConfig, ModuleConfig
    mixer = ModuleConfig(
        name="zoology.mixers.hybrid.Hybrid",
        kwargs={"configs": [
            {"name": "zoology.mixers.base_conv.BaseConv",
             "kwargs": {"l_max": 64, "kernel_size": 3, "implicit_long_conv": True}},
            {"name": "zoology.mixers.attention.MHA", "kwargs": {}},
        ]}
    )
    m = LanguageModel(ModelConfig(
        vocab_size=8192, d_model=64, n_layers=2,
        max_position_embeddings=0, sequence_mixer=mixer,
        state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
    )).cuda()
    out = m(torch.randint(0, 8192, (2, 64)).cuda())
    assert out.shape == (2, 64, 8192), f"Logits shape {out.shape} != (2, 64, 8192)"
    del m
    print("  ✓ Logits shape correct (return_embeddings=False fix working)")
except Exception as e:
    errors.append(f"return_embeddings check: {e}")

# stp_train.py patch check
try:
    from zoology.train import Trainer
    if hasattr(Trainer, '_save_checkpoint') and hasattr(Trainer, '_save_run_results'):
        print("  ✓ stp_train.py installed (checkpoint + JSON results)")
    else:
        print("  ⚠ stp_train.py NOT installed — WandB-only results")
except Exception as e:
    errors.append(f"train.py import: {e}")

# Results dirs
for d in ["/workspace/results/runs", "/workspace/checkpoints"]:
    if os.path.isdir(d):
        print(f"  ✓ {d}/")
    else:
        print(f"  ⚠ {d}/ missing")

if errors:
    print(f"\n  ERRORS ({len(errors)}):")
    for e in errors:
        print(f"    ✗ {e}")
    sys.exit(1)
else:
    print("\n  ALL CHECKS PASSED")
PYEOF

# Final summary
echo ""
echo "════════════════════════════════════════════════════════════════"
echo " SETUP COMPLETE"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo " Quick start:"
echo "   tmux new -s stp"
echo "   bash ${REPO_DIR}/scripts/run.sh stp_light,stp_t"
echo ""
echo " See RUNSHEET.md for full instructions and CSV editing guide."
echo "════════════════════════════════════════════════════════════════"
