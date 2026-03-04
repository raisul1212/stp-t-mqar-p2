# RUNSHEET: STP-T MQAR Phase 2

Step-by-step deployment guide.  
RISE Lab, Purdue University | March 2026

---

## Requirements

- 1× GPU ≥ 24 GB VRAM (A100 40 GB recommended; RTX 4090 24 GB works with `batch_size=32` for d=256)
- Linux + CUDA drivers, `nvidia-smi` working
- ~15 GB disk for packages, data cache, checkpoints

---

## Steps

### 1. Start GPU instance

Any cloud provider (RunPod, Lambda, Vast.ai) with a PyTorch template.  
Verify GPU: `nvidia-smi`

### 2. Clone repo

```bash
cd /workspace
git clone https://github.com/raisul1212/stp-t-mqar-p2.git
```

### 3. Run setup (once per pod)

```bash
bash /workspace/stp-t-mqar-p2/scripts/setup.sh
```

What setup does automatically:
- Verifies GPU, records PyTorch version (never modified)
- Installs: einops, wandb, pyyaml, pandas, tqdm, scipy, opt-einsum
- Installs flash-linear-attention + fla-core with `--no-deps` (protects pod's PyTorch)
- Handles the fla BitNet `AutoConfig.register` conflict if it appears
- Clones and installs Zoology with `--no-deps`
- **Patches Zoology `model.py`**: `return_embeddings=False` (without this all models plateau at ~25%)
- **Patches Zoology `config.py`**: pydantic v2 `LoggerConfig` (`Optional[str] = None`)
- **Patches fla wrappers**: removes `head_first=False` (fla ≥ 0.4.1 dropped it)
- Registers `mixers/` on `sys.path` so `stp_v3` and `fla_wrappers` are importable
- Installs `stp_train.py` (adds checkpoint saving + per-run JSON + best-epoch tracking)
- Runs full verification: imports, GPU forward passes, logits shape, patch checks

If setup prints `ALL CHECKS PASSED`, proceed.

---

## Running the Benchmark

All runs are controlled by **`configs/run_configs.csv`**.  
Open a tmux session first:

```bash
tmux new -s stp
```

### Option A — Direct launch (recommended)

```bash
cd /workspace/zoology
export WANDB_MODE=offline
export STP_RESULTS_DIR=/workspace/results
export STP_CHECKPOINT_DIR=/workspace/checkpoints
export STP_SAVE_CHECKPOINTS=best
export RUN_CONFIG=/workspace/stp-t-mqar-p2/configs/run_configs.csv

# STP models only (~8–12 h)
STP_MODELS=stp_light,stp_t \
  python3 -m zoology.launch /workspace/stp-t-mqar-p2/configs/mqar_p2.py \
  2>&1 | tee /workspace/experiment_log.txt

# STP vs direct competitors (~16–20 h)
STP_MODELS=stp_light,stp_t,retnet,gla \
  python3 -m zoology.launch /workspace/stp-t-mqar-p2/configs/mqar_p2.py \
  2>&1 | tee /workspace/experiment_log.txt

# Everything in CSV — 72 runs (~20–40 h)
python3 -m zoology.launch /workspace/stp-t-mqar-p2/configs/mqar_p2.py \
  2>&1 | tee /workspace/experiment_log.txt
```

Available `STP_MODELS` values: `attention`, `based`, `retnet`, `gla`, `stp_light`, `stp_t`

Detach tmux: `Ctrl+B`, then `D`  
Reconnect: `tmux attach -t stp`

### Option B — Custom CSV

```bash
cp /workspace/stp-t-mqar-p2/configs/run_configs.csv \
   /workspace/stp-t-mqar-p2/configs/ablation.csv
# edit ablation.csv, then:
RUN_CONFIG=/workspace/stp-t-mqar-p2/configs/ablation.csv \
  STP_MODELS=stp_t \
  python3 -m zoology.launch /workspace/stp-t-mqar-p2/configs/mqar_p2.py \
  2>&1 | tee /workspace/experiment_log.txt
```

---

## Editing run_configs.csv

`configs/run_configs.csv` is the single source of truth for every run.  
Edit it before launching — no Python changes needed.

### CSV columns

| Column | Description |
|--------|-------------|
| `enabled` | `1` = include, `0` = skip (row stays in file for reference) |
| `model` | `attention` \| `based` \| `retnet` \| `gla` \| `stp_light` \| `stp_t` |
| `d_model` | Model width |
| `num_heads` | Attention heads (`dk = d_model / num_heads`) |
| `n_layers` | Transformer depth |
| `lr` | Learning rate |
| `max_epochs` | Training epochs |
| `batch_size` | **Train** batch size. Test batch = `batch_size // 8` automatically |
| `gamma_init` | STP only — initial γ (retention strength). Ignored for baselines. |
| `lambda_init` | STP only — initial λ (forgetting rate). Ignored for baselines. |
| `run_id` | Unique ID used as results filename and WandB run name |

---

### Disable / re-enable a model

```bash
# Skip all attention rows (e.g. already have Phase 1 results)
sed -i 's/^1,attention/0,attention/g' \
  /workspace/stp-t-mqar-p2/configs/run_configs.csv

# Re-enable
sed -i 's/^0,attention/1,attention/g' \
  /workspace/stp-t-mqar-p2/configs/run_configs.csv
```

---

### Reduce batch_size for OOM on small GPU (d=256 rows)

Phase 1 OOM occurred at `batch_size=256`. Phase 2 defaults to `batch_size=64` for d=256.
On a 24 GB GPU, reduce d=256 rows to 32:

```bash
python3 -c "
lines = open('/workspace/stp-t-mqar-p2/configs/run_configs.csv').readlines()
out = []
for line in lines:
    if line.startswith('#') or line.startswith('enabled'):
        out.append(line); continue
    fields = line.rstrip().split(',')
    if len(fields) > 7 and fields[2].strip() == '256':
        fields[7] = '32'
        line = ','.join(fields) + '\n'
    out.append(line)
open('/workspace/stp-t-mqar-p2/configs/run_configs.csv', 'w').writelines(out)
print('Done — all d=256 rows now use batch_size=32')
"
```

Revert to 64:

```bash
python3 -c "
lines = open('/workspace/stp-t-mqar-p2/configs/run_configs.csv').readlines()
out = []
for line in lines:
    if line.startswith('#') or line.startswith('enabled'):
        out.append(line); continue
    fields = line.rstrip().split(',')
    if len(fields) > 7 and fields[2].strip() == '256' and fields[7].strip() == '32':
        fields[7] = '64'
        line = ','.join(fields) + '\n'
    out.append(line)
open('/workspace/stp-t-mqar-p2/configs/run_configs.csv', 'w').writelines(out)
print('Done — all d=256 rows restored to batch_size=64')
"
```

---

### Reduce max_epochs for a fast sanity check

```bash
python3 -c "
import csv, io
src = '/workspace/stp-t-mqar-p2/configs/run_configs.csv'
lines = open(src).readlines()
comments = [l for l in lines if l.lstrip().startswith('#')]
data = [l for l in lines if not l.lstrip().startswith('#')]
reader = csv.DictReader(data)
rows = list(reader); fields = reader.fieldnames
for r in rows: r['max_epochs'] = '8'
out = io.StringIO()
for c in comments: out.write(c)
w = csv.DictWriter(out, fieldnames=fields)
w.writeheader(); w.writerows(rows)
open(src, 'w').write(out.getvalue())
print(f'Done — {len(rows)} rows now use max_epochs=8')
"
```

---

### Change STP hyperparameters (gamma_init / lambda_init)

```bash
python3 -c "
import csv, io
src = '/workspace/stp-t-mqar-p2/configs/run_configs.csv'
lines = open(src).readlines()
comments = [l for l in lines if l.lstrip().startswith('#')]
data = [l for l in lines if not l.lstrip().startswith('#')]
reader = csv.DictReader(data)
rows = list(reader); fields = reader.fieldnames
for r in rows:
    if r['model'] in ('stp_t', 'stp_light'):
        r['gamma_init']  = '0.95'
        r['lambda_init'] = '0.05'
out = io.StringIO()
for c in comments: out.write(c)
w = csv.DictWriter(out, fieldnames=fields)
w.writeheader(); w.writerows(rows)
open(src, 'w').write(out.getvalue())
print('Done')
"
```

---

### Add a learning rate to the sweep

```bash
python3 -c "
import csv, io
src = '/workspace/stp-t-mqar-p2/configs/run_configs.csv'
lines = open(src).readlines()
comments = [l for l in lines if l.lstrip().startswith('#')]
data = [l for l in lines if not l.lstrip().startswith('#')]
reader = csv.DictReader(data)
rows = list(reader); fields = reader.fieldnames
new_rows = []
for r in rows:
    if r['model'] == 'stp_t' and r['lr'].strip() == '1e-3':
        nr = dict(r); nr['lr'] = '5e-4'
        d = r['d_model'].strip()
        nr['run_id'] = f'stp_t-d{d}-lr5.0e-04'
        new_rows.append(nr)
rows.extend(new_rows)
out = io.StringIO()
for c in comments: out.write(c)
w = csv.DictWriter(out, fieldnames=fields)
w.writeheader(); w.writerows(rows)
open(src, 'w').write(out.getvalue())
print(f'Added {len(new_rows)} rows at lr=5e-4')
"
```

---

### Dry-run: preview what configs will be built

```bash
cd /workspace/zoology
RUN_CONFIG=/workspace/stp-t-mqar-p2/configs/run_configs.csv \
STP_MODELS=stp_t \
python3 -c "
import sys; sys.argv=['']
exec(open('/workspace/stp-t-mqar-p2/configs/mqar_p2.py').read())
print(f'Total configs: {len(configs)}')
for c in configs:
    print(f'  {c.run_id}  lr={c.learning_rate}  d={c.model.d_model}  bs={c.data.batch_size[0]}')
" 2>/dev/null
```

---

## Monitoring

```bash
# Live training progress
tail -f /workspace/experiment_log.txt

# GPU utilization
watch -n 5 nvidia-smi

# Completed runs so far
ls /workspace/results/runs/ | wc -l

# Inspect a single completed run
python3 -m json.tool /workspace/results/runs/stp_t-d128-lr1.0e-03.json | head -40
```

---

## Extracting Results

```bash
python3 /workspace/stp-t-mqar-p2/scripts/collect_results.py \
    --wandb-dir /workspace/zoology/wandb \
    --out-dir   /workspace/results
```

Outputs:
```
/workspace/results/
├── summary.csv        ← one row per run, all metrics
└── per_kv.csv         ← best LR per (model, d_model) with per-KV accuracy
```

For richer output (best-epoch tracking, per-run JSONs, comparison table), use
`extract_results.py` from Phase 1 if available:

```bash
python3 /workspace/extract_results.py --output_dir /workspace/results
```

---

## Archive Before Stopping the Pod

```bash
TIMESTAMP=$(date +%Y%m%d_%H%M)
tar czf /workspace/stp_mqar_p2_${TIMESTAMP}.tar.gz \
    -C /workspace \
    results/ \
    experiment_log.txt \
    zoology/wandb/ \
    stp-t-mqar-p2/ \
    2>/dev/null
echo "Archive: /workspace/stp_mqar_p2_${TIMESTAMP}.tar.gz"

# Transfer
scp -P <port> root@<pod-ip>:/workspace/stp_mqar_p2_*.tar.gz .
# or
runpodctl send /workspace/stp_mqar_p2_*.tar.gz
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STP_MODELS` | *(empty = all)* | Comma-separated model names to run |
| `RUN_CONFIG` | `configs/run_configs.csv` | Path to CSV config file |
| `WANDB_MODE` | *(unset)* | Set to `offline` on RunPod |
| `STP_RESULTS_DIR` | `/workspace/results` | Per-run JSON output directory |
| `STP_CHECKPOINT_DIR` | `/workspace/checkpoints` | Checkpoint `.pt` files |
| `STP_SAVE_CHECKPOINTS` | `best` | `best` \| `all` \| `none` |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| All accuracies ~25% | `grep return_embeddings /workspace/zoology/zoology/model.py` must show `False`. Rerun `setup.sh`. |
| `No module named zoology` | `cd /workspace/zoology && pip install -e . --break-system-packages --no-deps` |
| `No module named fla` | `pip install flash-linear-attention fla-core --break-system-packages --no-deps` |
| `No module named stp_v3` | `setup.sh` adds a `.pth` file — rerun setup or: `export PYTHONPATH=/workspace/stp-t-mqar-p2/mixers:$PYTHONPATH` |
| `No module named fla_wrappers` | Same as above |
| `unexpected keyword head_first` | Rerun `setup.sh` — patches Zoology's fla wrappers |
| `AutoConfig.register` BitNet error | Rerun `setup.sh` — patches fla's BitNet `__init__.py` |
| pydantic validation error on `LoggerConfig` | Rerun `setup.sh` — patches `zoology/config.py` |
| OOM at d=256 | Reduce batch_size to 32 for d=256 rows (command above) |
| `run_configs.csv not found` | Set `RUN_CONFIG=/workspace/stp-t-mqar-p2/configs/run_configs.csv` |
| `STP_MODELS filter → 0 configs` | Typo in model name. Valid: `attention`, `based`, `retnet`, `gla`, `stp_light`, `stp_t` |
| fla returns tuple, Zoology crashes | Already handled by `fla_wrappers.py` (`output, *_ = self.retnet(x)`) |

---

## Performance Notes

- **STP recurrence**: `STPTLight` and `STPT` use a token-by-token Python loop (`for t in range(T)`). Plan ~3–4× longer wall time vs RetNet/GLA (which use Triton kernels via fla). Expected and known.
- **fla first-run compile**: Triton kernels compile on first use per `d_model`. Expect 2–5 min overhead at the start of the first run for each new `d_model`. Cached thereafter.
- **Data cache**: Zoology caches MQAR datasets in `/workspace/zoology_cache/`. First run generates; all subsequent load instantly.
