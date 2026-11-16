# Source for colab/run_graph_colab.ipynb (cells separated by "# %%"). Regenerate the notebook with
#   python tools/py_to_ipynb.py colab/run_graph_colab.py colab/run_graph_colab.ipynb
# %% [markdown]
# # EcoPE-GAT graph models on a Colab GPU
#
# Runs `src/04_graph.py` (GCN, GraphSAGE, GAT, GAT + context, EcoPE-GAT) on a GPU.
# **Everything is checkpointed on Google Drive**:
# * finished jobs: `results/graph/<scheme>_k10/preds/*.npz` (skipped on restart)
# * Optuna studies: `results/graph/<scheme>_k10/checkpoints/<job>/optuna.db` (saved after every trial)
# * refits: `results/graph/<scheme>_k10/checkpoints/<job>/N<h>_seed<s>.npz` (saved when each finishes)
#
# If the session disconnects, reconnect and run all cells again: work resumes where it stopped.
#
# **Before the first run**, copy these files from the local repository to
# `MyDrive/ecope/data/processed/` on Google Drive:
# `plots.parquet`, `splits.parquet`, `effort_layer.parquet`.
# Runtime > Change runtime type > GPU (T4 is enough).

# %%
from google.colab import drive
drive.mount("/content/drive")

import os
DRIVE = "/content/drive/MyDrive/ecope"
for f in ["plots.parquet", "splits.parquet", "effort_layer.parquet"]:
    path = f"{DRIVE}/data/processed/{f}"
    print(("OK      " if os.path.exists(path) else "MISSING ") + path)

# %%
# Code: the repository branch (public repo; for a private repo use a token).
!rm -rf /content/ecope && git clone -q -b rebuild-step1-3 https://github.com/Laib-Djounaid/ecope.git /content/ecope
!pip -q install torch_geometric==2.8.1 lightgbm==4.7.0 optuna==5.0.0 pyyaml pyarrow
%cd /content/ecope
!git log --oneline -1

# %%
import torch
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
assert torch.cuda.is_available(), "Select a GPU runtime"

# %%
# Results and checkpoints live on Drive; SQLite studies are written locally and copied to Drive
# after every trial (see src/checkpoint.py).
os.environ["ECOPE_PROCESSED_DIR"] = f"{DRIVE}/data/processed"
os.environ["ECOPE_RESULTS_DIR"] = f"{DRIVE}/results"
os.environ["ECOPE_LOCAL_TMP"] = "/content/ckpt_local"

# %%
# Main run: graph models, both settings, spatial CV. The hybrid model is CPU-only (LightGBM) and
# runs on the local machine. Re-run this cell after a disconnect: finished work is skipped.
!python -W ignore src/04_graph.py --scheme spatial \
    --models gcn sage gat gat_concat ecope_gat --settings transductive inductive \
    --n-jobs 2 --threads 2 2>&1 | grep -v Warning

# %%
# Progress: finished jobs and saved refits.
import glob
base = f"{DRIVE}/results/graph/spatial_k10"
print(len(glob.glob(f"{base}/preds/*.npz")), "jobs finished")
print(len(glob.glob(f"{base}/checkpoints/*/N*_seed*.json")), "refits saved")
