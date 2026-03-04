# RUNSHEET: STP-T MQAR Phase 2

Step-by-step deployment guide.
RISE Lab, Purdue University | March 2026

---

## Platform Support

| Platform | Notes |
|----------|-------|
| Thunder Compute (A100 80GB) | Primary dev platform — use home dir |
| RunPod (A100 40/80GB) | Use /workspace |
| Lambda Labs | Use Base template |
| Vast.ai | Filter hosts ≥95% reliability |
| Purdue Anvil | See Purdue section below |
| Purdue Gilbreth | See Purdue section below |

---

## Cloud GPU Pods (Thunder / RunPod / Lambda / Vast.ai)

### 1. Start GPU instance

Minimum: 1× GPU ≥24 GB VRAM. A100 80GB recommended. Verify: `nvidia-smi`

### 2. Open a persistent session

```bash
sudo apt-get update && sudo apt-get install -y tmux
tmux new -s stp
# Detach: Ctrl+B then D | Reconnect: tmux attach -t stp
```

### 3. Clone and setup

```bash
# Thunder Compute (home dir)
cd ~
git clone https://github.com/raisul1212/stp-t-mqar-p2.git
cd stp-t-mqar-p2
bash scripts/setup.sh 2>&1 | tee ~/setup_log.txt

# RunPod / Lambda (/workspace)
cd /workspace
git clone https://github.com/raisul1212/stp-t-mqar-p2.git
cd stp-t-mqar-p2
bash scripts/setup.sh 2>&1 | tee /workspace/setup_log.txt
```

setup.sh auto-detects workspace from its own location. No env vars needed.
Wait for **ALL CHECKS PASSED** before proceeding.

### 4. Run the benchmark

```bash
export WORKSPACE=~            # Thunder — adjust for your platform
# export WORKSPACE=/workspace # RunPod/Lambda

export WANDB_MODE=offline
export RUN_CONFIG=${WORKSPACE}/stp-t-mqar-p2/configs/run_configs.csv
export STP_RESULTS_DIR=${WORKSPACE}/results
export STP_CHECKPOINT_DIR=${WORKSPACE}/checkpoints
export STP_SAVE_CHECKPOINTS=best

cd ${WORKSPACE}/zoology

# STP models only (~8-12 h)
STP_MODELS=stp_light,stp_t \
  python3 -m zoology.launch ${WORKSPACE}/stp-t-mqar-p2/configs/mqar_p2.py \
  2>&1 | tee ${WORKSPACE}/experiment_log.txt

# STP + baselines (~16-20 h)
STP_MODELS=stp_light,stp_t,retnet,gla \
  python3 -m zoology.launch ${WORKSPACE}/stp-t-mqar-p2/configs/mqar_p2.py \
  2>&1 | tee ${WORKSPACE}/experiment_log.txt

# Full sweep - 72 runs (~20-40 h)
python3 -m zoology.launch ${WORKSPACE}/stp-t-mqar-p2/configs/mqar_p2.py \
  2>&1 | tee ${WORKSPACE}/experiment_log.txt
```

---

## Purdue Clusters (Anvil / Gilbreth)

### 1. Login and request GPU node

```bash
ssh username@anvil.rcac.purdue.edu
# or
ssh username@gilbreth.rcac.purdue.edu

# Interactive GPU session (adjust account/partition)
salloc -A your_account -p gpu -N 1 --gpus-per-node=1 --time=48:00:00
```

### 2. Load modules

```bash
# Anvil
module load anaconda cuda
conda activate your_pytorch_env

# Gilbreth
module load anaconda/2020.11-py38 cuda/11.4.0
conda activate your_pytorch_env

# Verify
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If you need to create a conda env with PyTorch:
```bash
conda create -n stp python=3.10 -y
conda activate stp
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### 3. Clone and setup

```bash
cd $SCRATCH
git clone https://github.com/raisul1212/stp-t-mqar-p2.git
cd stp-t-mqar-p2
bash scripts/setup.sh 2>&1 | tee ~/setup_log.txt
```

setup.sh auto-detects Purdue from hostname (anvil/gilbreth/purdue) and adjusts:
- Skips apt-get (no sudo on cluster)
- Uses --user pip installs if outside conda
- Falls back to user site-packages for .pth registration
- Shows screen as tmux alternative

### 4. Persistent session

```bash
# tmux (on login node)
tmux new -s stp

# screen (always available)
screen -S stp
# Detach: Ctrl+A then D | Reconnect: screen -r stp
```

### 5. Run the benchmark

```bash
export WORKSPACE=$SCRATCH/stp-workspace
mkdir -p $WORKSPACE

export WANDB_MODE=offline
export RUN_CONFIG=$SCRATCH/stp-t-mqar-p2/configs/run_configs.csv
export STP_RESULTS_DIR=$WORKSPACE/results
export STP_CHECKPOINT_DIR=$WORKSPACE/checkpoints

cd $SCRATCH/zoology
STP_MODELS=stp_light,stp_t \
  python3 -m zoology.launch $SCRATCH/stp-t-mqar-p2/configs/mqar_p2.py \
  2>&1 | tee $WORKSPACE/experiment_log.txt
```

### 6. SLURM batch job

```bash
cat > run_stp.sh << 'EOF'
#!/bin/bash
#SBATCH -A your_account
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH -o stp_%j.log
#SBATCH -e stp_%j.err
#SBATCH --job-name=stp_mqar

module load anaconda cuda
conda activate your_pytorch_env

export WORKSPACE=$SCRATCH/stp-workspace
export WANDB_MODE=offline
export RUN_CONFIG=$SCRATCH/stp-t-mqar-p2/configs/run_configs.csv
export STP_RESULTS_DIR=$WORKSPACE/results
export STP_CHECKPOINT_DIR=$WORKSPACE/checkpoints

cd $SCRATCH/zoology
python3 -m zoology.launch $SCRATCH/stp-t-mqar-p2/configs/mqar_p2.py \
  2>&1 | tee $WORKSPACE/experiment_log.txt
EOF

sbatch run_stp.sh
squeue -u $USER    # monitor
scancel <job_id>   # cancel
```

---

## Configuration (all platforms)

All run parameters live in `configs/run_configs.csv` — no Python edits needed.

### CSV columns

| Column | Description |
|--------|-------------|
| `enabled` | `1` = run, `0` = skip |
| `model` | `attention` / `based` / `retnet` / `gla` / `stp_light` / `stp_t` |
| `d_model` | Model width |
| `num_heads` | Attention heads |
| `n_layers` | Depth |
| `lr` | Learning rate |
| `max_epochs` | Training epochs |
| `batch_size` | Train batch (d=256 defaults to 64 to avoid OOM) |
| `gamma_init` | STP only — initial gamma |
| `lambda_init` | STP only — initial lambda |
| `run_id` | Unique ID for results file and WandB run name |

### Common CSV edits

```bash
# Disable a model
sed -i 's/^1,attention/0,attention/g' configs/run_configs.csv

# Re-enable
sed -i 's/^0,attention/1,attention/g' configs/run_configs.csv

# Reduce batch_size for d=256 (OOM on 24GB GPU)
python3 -c "
lines = open('configs/run_configs.csv').readlines(); out = []
for l in lines:
    f = l.rstrip().split(',')
    if len(f) > 7 and f[2].strip() == '256': f[7] = '32'
    out.append(','.join(f)+'\n' if len(f)>1 else l)
open('configs/run_configs.csv','w').writelines(out)
print('Done')
"

# Reduce max_epochs for sanity check
python3 -c "
import csv, io
src = 'configs/run_configs.csv'
lines = open(src).readlines()
comments = [l for l in lines if l.lstrip().startswith('#')]
data = [l for l in lines if not l.lstrip().startswith('#')]
reader = csv.DictReader(data); rows = list(reader); fields = reader.fieldnames
for r in rows: r['max_epochs'] = '4'
out = io.StringIO()
for c in comments: out.write(c)
w = csv.DictWriter(out, fieldnames=fields)
w.writeheader(); w.writerows(rows)
open(src,'w').write(out.getvalue())
print(f'Done — {len(rows)} rows now use max_epochs=4')
"
```

---

## Monitoring

```bash
tail -f ${WORKSPACE}/experiment_log.txt   # live log
watch -n 5 nvidia-smi                     # GPU util
ls ${WORKSPACE}/results/runs/ | wc -l    # completed runs
squeue -u $USER                           # Purdue job status
```

---

## Extracting Results

```bash
python3 stp-t-mqar-p2/scripts/collect_results.py \
    --wandb-dir ${WORKSPACE}/zoology/wandb \
    --out-dir   ${WORKSPACE}/results
```

Outputs `results/summary.csv` and `results/per_kv.csv`.

---

## Archive Before Stopping

```bash
TIMESTAMP=$(date +%Y%m%d_%H%M)
tar czf ~/stp_mqar_p2_${TIMESTAMP}.tar.gz \
    ${WORKSPACE}/results/ \
    ${WORKSPACE}/experiment_log.txt \
    ~/stp-t-mqar-p2/ 2>/dev/null

# Transfer to local
scp ubuntu@<ip>:~/stp_mqar_p2_*.tar.gz .           # cloud
scp username@anvil.rcac.purdue.edu:~/stp_*.tar.gz . # Purdue
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WORKSPACE` | parent of repo | Override workspace location |
| `PURDUE` | `0` | Set `1` to force cluster mode |
| `STP_MODELS` | *(all)* | Comma-separated model filter |
| `RUN_CONFIG` | `configs/run_configs.csv` | Path to CSV |
| `WANDB_MODE` | *(unset)* | Set `offline` on pod/cluster |
| `STP_RESULTS_DIR` | `${WORKSPACE}/results` | Per-run JSON output |
| `STP_CHECKPOINT_DIR` | `${WORKSPACE}/checkpoints` | Checkpoint .pt files |
| `STP_SAVE_CHECKPOINTS` | `best` | `best` / `all` / `none` |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| All accuracies ~25% | `grep return_embeddings zoology/zoology/model.py` must show `False`. Rerun setup. |
| `No module named zoology` | `cd zoology && pip3 install -e . --no-deps` |
| `No module named fla` | `pip3 install flash-linear-attention fla-core --no-deps` |
| `No module named stp_v3` | Rerun setup or `export PYTHONPATH=~/stp-t-mqar-p2/mixers:$PYTHONPATH` |
| `pip: command not found` | setup.sh auto-detects pip3/pip/python3 -m pip — rerun setup |
| `apt-get: permission denied` | Expected on Purdue — setup.sh skips apt gracefully |
| `tmux not found` | `sudo apt-get install tmux` (cloud) or use `screen` (Purdue) |
| OOM at d=256 | Reduce batch_size to 32 for d=256 rows (command above) |
| `head_first` kwarg crash | Rerun setup — patches Zoology fla wrappers |
| pydantic validation error | Rerun setup — patches zoology/config.py |
| Purdue: site-packages write error | setup.sh falls back to user site-packages automatically |
| Purdue: module not found | `module load anaconda cuda` then `conda activate your_env` first |