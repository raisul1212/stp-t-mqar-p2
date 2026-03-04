# STP-T MQAR Phase 2

**Benchmarking STP-Transformer v3 (White Paper v3.0) on MQAR**  
RISE Lab, Purdue University | March 2026

Implements and benchmarks the two architectures from the **STP-Transformer v3 white paper**
against standard published baselines using the [Zoology](https://github.com/HazyResearch/zoology)
framework (Arora et al., ICLR 2024).

---

## Models

| Model | Retention | Overhead | Two-factor | Convex interp | Source |
|-------|-----------|----------|-----------|---------------|--------|
| Attention (MHA) | Full softmax | — | — | — | Zoology built-in |
| Based | Feature map | — | — | — | Zoology built-in |
| RetNet | Scalar/head (fixed) | H | No | No | fla (Sun et al. 2023) |
| GLA | Per-row adaptive | d² | No | No | fla (Yang et al. 2023) |
| **STP-T-Light** | **Scalar/head input-dep.** | **4H** | **Yes** | **Yes** | **This work** |
| **STP-T** | **Per-column input-dep.** | **4d** | **Yes** | **Yes** | **This work** |

Both STP-T variants implement the white paper spec:
- **Two-factor decomposition**: `ρ = (1−λ)·γ` — non-monotonic retention impossible with single sigmoid
- **Key-driven gates**: input to retention is `k(t)`, already computed — no extra `Wg·x` projection
- **Convex state interpolation**: `S(t) = ρ·S(t−1) + (1−ρ)·(v⊗kᵀ)` — bounded state, stable training

---

## Repo Structure

```
stp-t-mqar-p2/
├── configs/
│   ├── mqar_p2.py          # Zoology sweep config — reads all params from CSV
│   └── run_configs.csv     # Single source of truth for every run parameter
├── mixers/
│   ├── stp_v3.py           # STPTLight + STPT (white paper implementation)
│   └── fla_wrappers.py     # Zoology-compatible wrappers for fla RetNet and GLA
├── scripts/
│   ├── setup.sh            # One-shot environment setup
│   └── collect_results.py  # WandB → CSV aggregation (legacy; see RUNSHEET for extract_results.py)
├── results/                # Output directory (populated after runs)
├── RUNSHEET.md             # Complete step-by-step deployment and CSV editing guide
└── README.md
```

---

## Quick Start

```bash
# 1. Clone
cd /workspace
git clone https://github.com/raisul1212/stp-t-mqar-p2.git

# 2. Setup (~10 min — installs Zoology, fla, patches, verifies GPU)
bash stp-t-mqar-p2/scripts/setup.sh

# 3. (Optional) adjust batch_size for your GPU — see RUNSHEET.md
# Default: d=64/128 → batch_size=256, d=256 → batch_size=64

# 4. Open tmux (survives SSH disconnect)
tmux new -s stp

# 5. Run
cd /workspace/zoology
export WANDB_MODE=offline
export RUN_CONFIG=/workspace/stp-t-mqar-p2/configs/run_configs.csv

# STP models only (~8–12 h)
STP_MODELS=stp_light,stp_t \
  python3 -m zoology.launch /workspace/stp-t-mqar-p2/configs/mqar_p2.py \
  2>&1 | tee /workspace/experiment_log.txt

# Everything (~20–40 h)
python3 -m zoology.launch /workspace/stp-t-mqar-p2/configs/mqar_p2.py \
  2>&1 | tee /workspace/experiment_log.txt
```

Available `STP_MODELS` values: `attention`, `based`, `retnet`, `gla`, `stp_light`, `stp_t`

See **RUNSHEET.md** for the complete guide including CSV editing, monitoring, result extraction,
and troubleshooting.

---

## Configuration

All run parameters live in `configs/run_configs.csv` — **no Python edits needed**.

```
enabled,model,d_model,num_heads,n_layers,lr,max_epochs,batch_size,gamma_init,lambda_init,run_id
1,stp_t,128,2,2,1e-3,32,256,0.9,0.1,stp_t-d128-lr1.0e-03
...
```

Edit from the command line (examples — see RUNSHEET.md for full list):

```bash
# Reduce batch_size for d=256 rows (OOM on 24 GB GPU)
python3 -c "
lines=open('configs/run_configs.csv').readlines(); out=[]
for l in lines:
    f=l.rstrip().split(',')
    if len(f)>7 and f[2].strip()=='256': f[7]='32'
    out.append(','.join(f)+'\n' if len(f)>1 else l)
open('configs/run_configs.csv','w').writelines(out)
print('Done')
"

# Disable a model
sed -i 's/^1,attention/0,attention/g' configs/run_configs.csv

# Re-enable a model
sed -i 's/^0,attention/1,attention/g' configs/run_configs.csv
```

---

## Phase 1 → Phase 2 Changes

| Issue | Status |
|-------|--------|
| `return_embeddings=True` (25% plateau) | Fixed — `setup.sh` patches Zoology `model.py` |
| pydantic v2 `LoggerConfig` error | Fixed — `setup.sh` patches `zoology/config.py` |
| fla `head_first=False` kwarg crash | Fixed — `setup.sh` patches fla wrappers |
| fla tuple output (Zoology expects tensor) | Fixed — `fla_wrappers.py` unpacks `(output, *_)` |
| d=256 OOM at `batch_size=256` | Fixed — CSV defaults to `batch_size=64` for d=256 |
| All params hardcoded in Python | Fixed — `run_configs.csv` is single source of truth |
| No per-model param control | Fixed — every row in CSV is independently tunable |

---

## References

- Sun et al. "Retentive Network" arXiv:2307.08621 (2023)
- Yang et al. "Gated Linear Attention Transformers" ICML 2024, arXiv:2312.06635
- Arora et al. "Zoology" ICLR 2024, arXiv:2312.04927
- Arora et al. "Based" arXiv:2402.18668 (2024)
