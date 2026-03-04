#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════════
# setup.sh — STP-T MQAR Phase 2 environment setup
# RISE Lab, Purdue University  |  March 2026
#
# PLATFORM SUPPORT:
#   - Cloud GPU pods  (RunPod, Thunder Compute, Lambda Labs, Vast.ai)
#   - Purdue clusters (Anvil, Gilbreth) — set PURDUE=1 or auto-detected
#   - Any Linux machine with Python 3 + PyTorch + CUDA
#
# DESIGN PRINCIPLE:
#   We NEVER let pip modify the pre-installed PyTorch stack.
#   All packages that could pull in a conflicting torch are installed --no-deps.
#
# Steps:
#   1.  Platform detection
#   2.  GPU + PyTorch version check
#   3.  System packages (git, tmux) — skipped gracefully on clusters
#   4.  Safe Python deps (einops, wandb, pyyaml, etc.)
#   5.  flash-linear-attention with --no-deps + BitNet conflict fix
#   6.  Zoology clone + install (--no-deps)
#   7.  Zoology patches (return_embeddings, pydantic v2, fla head_first)
#   8.  Register mixer directory on sys.path (.pth file)
#   9.  Install stp_train.py (checkpoints + JSON results)
#   10. Full verification
# ════════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Workspace + repo paths ───────────────────────────────────────────────────
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
WORKSPACE="${WORKSPACE:-$(dirname "${_REPO_ROOT}")}"
REPO_DIR="${_REPO_ROOT}"
ZOOLOGY_DIR="${WORKSPACE}/zoology"
MIXER_DIR="${REPO_DIR}/mixers"

# ── Platform detection ───────────────────────────────────────────────────────
IS_PURDUE=0
if [ "${PURDUE:-0}" = "1" ] || \
   echo "${HOSTNAME:-}" | grep -qiE "anvil|gilbreth|bell|negishi|purdue" || \
   [ -n "${RCAC_CLUSTER:-}" ]; then
    IS_PURDUE=1
fi

# ── pip resolution ───────────────────────────────────────────────────────────
# ALWAYS prefer "python3 -m pip" — it uses the pip module matching the active
# Python interpreter and avoids the broken system pip3 binary on Python ≥3.12
# (Ubuntu/Debian ship /usr/bin/pip3 linked to a pip that uses the removed
# pkgutil.ImpImporter, causing "AttributeError: module 'pkgutil' has no
# attribute 'ImpImporter'" on every pip3 invocation).
if python3 -m pip --version &>/dev/null; then
    PIP="python3 -m pip"
elif command -v pip3 &>/dev/null && pip3 --version &>/dev/null 2>&1; then
    PIP="pip3"
elif command -v pip &>/dev/null && pip --version &>/dev/null 2>&1; then
    PIP="pip"
else
    # Last resort: bootstrap pip
    warn "No working pip found — attempting bootstrap"
    if python3 -m ensurepip --upgrade &>/dev/null; then
        PIP="python3 -m pip"
    elif curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py 2>/dev/null; then
        python3 /tmp/get-pip.py --break-system-packages 2>/dev/null
        PIP="python3 -m pip"
    else
        fail "Cannot find or bootstrap pip. Install pip manually and rerun."
        exit 1
    fi
fi

# On Purdue outside conda: add --user since system site-packages is read-only
PIP_EXTRA=""
if [ "${IS_PURDUE}" = "1" ] && [ -z "${CONDA_PREFIX:-}" ]; then
    PIP_EXTRA="--user"
fi

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}⚠${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; }

echo "════════════════════════════════════════════════════════════════"
echo " STP-T MQAR Phase 2 Setup"
echo " Workspace:  ${WORKSPACE}"
echo " Repo:       ${REPO_DIR}"
if [ "${IS_PURDUE}" = "1" ]; then
echo " Platform:   Purdue cluster (${RCAC_CLUSTER:-${HOSTNAME:-unknown}})"
else
echo " Platform:   Cloud/Linux pod"
fi
echo "════════════════════════════════════════════════════════════════"

# ── 1. Platform-specific pre-checks ─────────────────────────────────────────
echo "[1/10] Platform check..."
if [ "${IS_PURDUE}" = "1" ]; then
    # On Purdue, modules must be loaded before running this script.
    # We just check that Python and torch are available.
    if ! command -v python3 &>/dev/null; then
        fail "python3 not found. On Purdue, load modules first:"
        echo "        module load anaconda"
        echo "        module load cuda"
        echo "        conda activate your_env"
        exit 1
    fi
    ok "Purdue cluster detected — skipping apt-get steps"
    ok "python3: $(python3 --version 2>&1)"
else
    ok "Cloud pod — full setup mode"
fi

# ── 2. GPU check ─────────────────────────────────────────────────────────────
echo "[2/10] GPU check..."
if ! nvidia-smi &>/dev/null; then
    fail "No GPU / nvidia-smi not found"
    if [ "${IS_PURDUE}" = "1" ]; then
        echo "       On Purdue: make sure you have a GPU allocation."
        echo "       Example: salloc -A your_account -p gpu -N 1 --gpus-per-node=1"
    fi
    exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
ok "GPU: ${GPU_NAME} (${GPU_MEM})"

TORCH_VER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "none")
if [ "${TORCH_VER}" = "none" ]; then
    fail "PyTorch not found."
    if [ "${IS_PURDUE}" = "1" ]; then
        echo "       Load modules: module load anaconda cuda"
        echo "       Then activate a conda env with PyTorch installed."
    else
        echo "       Use a PyTorch pod template."
    fi
    exit 1
fi
ok "PyTorch ${TORCH_VER}"

# ── 3. System packages ───────────────────────────────────────────────────────
echo "[3/10] System packages..."
if [ "${IS_PURDUE}" = "1" ]; then
    # On clusters, git is always available; tmux may or may not be.
    command -v git  &>/dev/null && ok "git (system)" || warn "git not found — install manually"
    command -v tmux &>/dev/null && ok "tmux (system)" || warn "tmux not found — use 'screen' or 'nohup' instead"
else
    # Cloud pod — use apt
    if command -v apt-get &>/dev/null; then
        apt-get update -qq 2>/dev/null && \
        apt-get install -y -qq git tmux 2>/dev/null && \
        ok "git, tmux installed" || warn "apt-get failed — git/tmux may already be present"
    else
        warn "apt-get not available — skipping system packages"
    fi
fi

# ── 4. Python dependencies ───────────────────────────────────────────────────
echo "[4/10] Python dependencies..."
for pkg in einops wandb pyyaml pandas tqdm rich opt-einsum scipy; do
    pymod="${pkg//-/_}"
    if python3 -c "import ${pymod}" 2>/dev/null; then
        true  # already installed
    else
        ${PIP} install "${pkg}" --break-system-packages ${PIP_EXTRA} -q 2>/dev/null || \
        ${PIP} install "${pkg}" ${PIP_EXTRA} -q 2>/dev/null || true
    fi
done

python3 -c "import pydantic" 2>/dev/null || \
    ${PIP} install pydantic --break-system-packages ${PIP_EXTRA} -q --no-deps 2>/dev/null || \
    ${PIP} install pydantic ${PIP_EXTRA} -q --no-deps 2>/dev/null || true

python3 -c "import transformers" 2>/dev/null || \
    ${PIP} install transformers --break-system-packages ${PIP_EXTRA} -q --no-deps 2>/dev/null || \
    ${PIP} install transformers ${PIP_EXTRA} -q --no-deps 2>/dev/null || true

ok "Python deps"

# ── 5. flash-linear-attention ────────────────────────────────────────────────
echo "[5/10] flash-linear-attention..."
FLA_OK=0

if python3 -c "from fla.layers import GatedLinearAttention, MultiScaleRetention" 2>/dev/null; then
    ok "fla (existing, import OK)"
    FLA_OK=1
fi

if [ "${FLA_OK}" -eq 0 ]; then
    ${PIP} install "flash-linear-attention>=0.4.1" \
        --break-system-packages ${PIP_EXTRA} -q --no-deps 2>/dev/null || \
    ${PIP} install "flash-linear-attention>=0.4.1" \
        ${PIP_EXTRA} -q --no-deps 2>/dev/null || true

    ${PIP} install fla-core \
        --break-system-packages ${PIP_EXTRA} -q --no-deps 2>/dev/null || \
    ${PIP} install fla-core \
        ${PIP_EXTRA} -q --no-deps 2>/dev/null || true

    if python3 -c "from fla.layers import GatedLinearAttention, MultiScaleRetention" 2>/dev/null; then
        ok "fla installed (--no-deps)"
        FLA_OK=1
    else
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
    fail "PyTorch version changed! (${TORCH_VER} → ${TORCH_AFTER})"
    exit 1
fi
ok "PyTorch version preserved: ${TORCH_VER}"

# ── 6. Zoology ───────────────────────────────────────────────────────────────
echo "[6/10] Zoology framework..."
if [ ! -d "${ZOOLOGY_DIR}" ]; then
    echo "  Cloning Zoology..."
    git clone -q https://github.com/HazyResearch/zoology.git "${ZOOLOGY_DIR}"
    ok "Cloned Zoology → ${ZOOLOGY_DIR}"
fi
cd "${ZOOLOGY_DIR}"
if ! python3 -c "import zoology" 2>/dev/null; then
    ${PIP} install -e . --break-system-packages ${PIP_EXTRA} -q --no-deps 2>/dev/null || \
    ${PIP} install -e . ${PIP_EXTRA} -q --no-deps 2>/dev/null && \
    ok "Zoology installed (--no-deps)" || { fail "Zoology install failed"; exit 1; }
else
    ok "Zoology (existing)"
fi

# ── 7. Zoology patches ───────────────────────────────────────────────────────
echo "[7/10] Zoology patches..."

# Patch A: return_embeddings=False
MODEL_PY="${ZOOLOGY_DIR}/zoology/model.py"
if grep -q "return_embeddings.*=.*True" "${MODEL_PY}" 2>/dev/null; then
    sed -i 's/return_embeddings.*=.*True/return_embeddings: bool = False/g' "${MODEL_PY}"
    ok "Patched: return_embeddings=False"
else
    ok "return_embeddings=False already applied"
fi

# Patch B: pydantic v2 LoggerConfig
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
echo "  Patching fla head_first compatibility..."
for fla_wrapper in gla.py delta_net.py gated_delta_net.py rwkv7.py; do
    fpath="${ZOOLOGY_DIR}/zoology/mixers/${fla_wrapper}"
    [ ! -f "${fpath}" ] && continue
    if grep -q "head_first" "${fpath}" 2>/dev/null; then
        sed -i '/head_first=False/d' "${fpath}"
        python3 -c "
import re
t = open('${fpath}').read()
t = re.sub(r',(\s*\))', r'\1', t)
open('${fpath}', 'w').write(t)
" 2>/dev/null || true
        ok "  Fixed: ${fla_wrapper} (removed head_first)"
    fi
done

# ── 8. Register mixer directory on sys.path ──────────────────────────────────
echo "[8/10] Registering mixers on sys.path..."
# Try system site-packages first; fall back to user site-packages on clusters
SITE_PKG=$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || \
           python3 -c "import site; print(site.getusersitepackages())")
PTH_FILE="${SITE_PKG}/stp_t_p2.pth"
if echo "${MIXER_DIR}" > "${PTH_FILE}" 2>/dev/null; then
    ok "Registered: ${MIXER_DIR}"
    ok "  → ${PTH_FILE}"
else
    # Fallback: write to user site-packages
    USER_SITE=$(python3 -c "import site; print(site.getusersitepackages())")
    mkdir -p "${USER_SITE}"
    echo "${MIXER_DIR}" > "${USER_SITE}/stp_t_p2.pth"
    ok "Registered (user): ${MIXER_DIR}"
    ok "  → ${USER_SITE}/stp_t_p2.pth"
fi

# ── 9. Install stp_train.py ──────────────────────────────────────────────────
echo "[9/10] Installing stp_train.py..."
if [ -f "${REPO_DIR}/scripts/stp_train.py" ]; then
    if [ ! -f "${ZOOLOGY_DIR}/zoology/train.py.orig" ]; then
        cp "${ZOOLOGY_DIR}/zoology/train.py" "${ZOOLOGY_DIR}/zoology/train.py.orig"
    fi
    cp "${REPO_DIR}/scripts/stp_train.py" "${ZOOLOGY_DIR}/zoology/train.py"
    ok "stp_train.py installed"
else
    warn "stp_train.py not found — WandB-only results"
fi

mkdir -p "${WORKSPACE}/results/runs"
mkdir -p "${WORKSPACE}/checkpoints"
ok "Directories: ${WORKSPACE}/results/runs/, ${WORKSPACE}/checkpoints/"

# ── 10. Verification ─────────────────────────────────────────────────────────
echo "[10/10] Verification..."
WORKSPACE_FOR_PY="${WORKSPACE}" python3 << 'PYEOF'
import sys, os, torch

errors = []
ws = os.environ.get("WORKSPACE_FOR_PY", "")

if not torch.cuda.is_available():
    errors.append("CUDA not available")
else:
    print(f"  ✓ PyTorch {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}")

try:
    from zoology.config import TrainConfig, ModelConfig, DataConfig, LoggerConfig, ModuleConfig
    lc = LoggerConfig()
    assert lc.project_name is None and lc.entity is None
    print("  ✓ zoology.config (pydantic v2 OK)")
except Exception as e:
    errors.append(f"zoology.config: {e}")

try:
    from zoology.data.multiquery_ar import MQARConfig
    _ = MQARConfig(vocab_size=8192, input_seq_len=64, num_examples=10, num_kv_pairs=4)
    print("  ✓ zoology.data.multiquery_ar.MQARConfig")
except Exception as e:
    errors.append(f"zoology.data.multiquery_ar: {e}")

try:
    from zoology.mixers.base_conv import BaseConv
    print("  ✓ zoology.mixers.base_conv.BaseConv")
except Exception as e:
    errors.append(f"zoology.mixers.base_conv: {e}")

try:
    from zoology.mixers.attention import MHA
    from zoology.mixers.based import Based
    print("  ✓ zoology.mixers.attention.MHA, based.Based")
except Exception as e:
    errors.append(f"zoology mixers: {e}")

try:
    from fla.layers import GatedLinearAttention, MultiScaleRetention
    print("  ✓ fla.layers (GatedLinearAttention, MultiScaleRetention)")
except Exception as e:
    errors.append(f"fla.layers: {e}")

try:
    from fla_wrappers import RetNetWrapper, GLAWrapper
    print("  ✓ fla_wrappers (RetNetWrapper, GLAWrapper)")
except Exception as e:
    errors.append(f"fla_wrappers: {e}")

try:
    from stp_v3 import STPT, STPTLight
    print("  ✓ stp_v3 (STPT, STPTLight)")
except Exception as e:
    errors.append(f"stp_v3: {e}")

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
    print("  ✓ Logits shape correct (return_embeddings=False working)")
except Exception as e:
    errors.append(f"return_embeddings check: {e}")

try:
    from zoology.train import Trainer
    if hasattr(Trainer, '_save_checkpoint') and hasattr(Trainer, '_save_run_results'):
        print("  ✓ stp_train.py installed (checkpoint + JSON results)")
    else:
        print("  ⚠ stp_train.py NOT installed — WandB-only results")
except Exception as e:
    errors.append(f"train.py import: {e}")

for d in [f"{ws}/results/runs", f"{ws}/checkpoints"]:
    if d and os.path.isdir(d):
        print(f"  ✓ {d}/")
    elif d:
        print(f"  ⚠ {d}/ missing")

if errors:
    print(f"\n  ERRORS ({len(errors)}):")
    for e in errors:
        print(f"    ✗ {e}")
    sys.exit(1)
else:
    print("\n  ALL CHECKS PASSED")
PYEOF

echo ""
echo "════════════════════════════════════════════════════════════════"
echo " SETUP COMPLETE"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo " Quick start:"
if [ "${IS_PURDUE}" = "1" ]; then
echo "   screen -S stp   # or: tmux new -s stp (if tmux available)"
else
echo "   tmux new -s stp"
fi
echo "   cd ${ZOOLOGY_DIR}"
echo "   export WANDB_MODE=offline"
echo "   export RUN_CONFIG=${REPO_DIR}/configs/run_configs.csv"
echo "   STP_MODELS=stp_light,stp_t python3 -m zoology.launch ${REPO_DIR}/configs/mqar_p2.py"
echo ""
echo " See RUNSHEET.md for full instructions."
echo "════════════════════════════════════════════════════════════════"