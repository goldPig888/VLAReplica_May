# MolmoAct2 VLAReplica setup

MolmoAct2 is maintained as a separate Git repository. Do not commit a Conda
environment, model weights, Hugging Face caches, datasets, or checkpoints to
this repository.

## One-time push from the original machine

Create a fork of `allenai/molmoact2` in the GitHub web interface. Then publish
the prepared branch (replace the URL if the fork has a different name):

```bash
cd ~/Desktop/Github/VLAReplica/molmoact2
git remote rename origin upstream
git remote add origin https://github.com/goldPig888/molmoact2.git
git push -u origin vlareplica-finetune
```

The prepared MolmoAct2 commit is `dd5fbb3`. The outer VLAReplica repository
intentionally ignores `molmoact2/`, preventing Git from recording a broken
embedded repository.

## 1. Check the new server

```bash
nvidia-smi
df -h
```

Choose a large storage location and set it once. For example:

```bash
export VLA_STORAGE=/metadisk/$USER
mkdir -p "$VLA_STORAGE"/{conda-envs,conda-pkgs,tmp,pip-cache,huggingface,lerobot-data,molmo-data,molmoact2-checkpoints}
```

## 2. Clone both repositories

```bash
git clone https://github.com/goldPig888/VLAReplica_May.git VLAReplica
cd VLAReplica
git clone --branch vlareplica-finetune --recurse-submodules \
  https://github.com/goldPig888/molmoact2.git molmoact2
git -C molmoact2 submodule update --init --recursive
```

## 3. Create the training environment on large storage

The environment is installed from MolmoAct2's own dependency specification,
not from the top-level VLAReplica `environment.yml`.

```bash
CONDA_PKGS_DIRS="$VLA_STORAGE/conda-pkgs" \
  conda create --prefix "$VLA_STORAGE/conda-envs/molmoact2" python=3.12 -y

conda activate "$VLA_STORAGE/conda-envs/molmoact2"

cd molmoact2/experiments
TMPDIR="$VLA_STORAGE/tmp" PIP_CACHE_DIR="$VLA_STORAGE/pip-cache" \
  python -m pip install -e '.[all]'
```

If PyTorch reports that the NVIDIA driver is too old, select a PyTorch CUDA
build supported by the driver shown by `nvidia-smi`; do not update the whole
environment blindly.

## 4. Configure caches

```bash
conda env config vars set \
  HF_HOME="$VLA_STORAGE/huggingface" \
  HF_HUB_CACHE="$VLA_STORAGE/huggingface/hub" \
  LEROBOT_DATA_ROOT="$VLA_STORAGE/lerobot-data" \
  MOLMO_DATA_DIR="$VLA_STORAGE/molmo-data" \
  TMPDIR="$VLA_STORAGE/tmp" \
  PIP_CACHE_DIR="$VLA_STORAGE/pip-cache" \
  LEROBOT_VIDEO_BACKEND=pyav

conda deactivate
conda activate "$VLA_STORAGE/conda-envs/molmoact2"
```

## 5. Verify before training

```bash
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))
PY
```

Then follow the benchmark training and monitoring commands in `cmds.md`.

## Repository responsibilities

- `VLAReplica_May`: benchmark/inference code, documentation, and commands.
- MolmoAct2 fork, branch `vlareplica-finetune`: dataset mixture, validation
  support, training launchers, dashboard, conversion, and upload scripts.
- Hugging Face: dataset and final trained checkpoint.
- Large server disk: Conda environment, caches, downloaded weights, logs, and
  intermediate checkpoints.
