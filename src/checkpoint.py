"""Fine-grained checkpointing so long runs survive interruptions (e.g. Colab disconnects).

Levels:
  1. job    - finished jobs write preds/<tag>.npz and are skipped on restart (runners).
  2. trial  - each Optuna study is an SQLite database, saved after every trial.
  3. refit  - each (horizon, seed) refit is saved as soon as it finishes.

The SQLite file is written on a LOCAL disk ($ECOPE_LOCAL_TMP, default: the checkpoint dir itself)
and copied atomically to the checkpoint dir after each trial, because SQLite on network /
Google-Drive file systems can be corrupted by a disconnect in the middle of a write.

Note: on resume the TPE sampler is re-seeded, so the remaining trials are not the exact sequence
an uninterrupted run would have drawn; completed trials are kept as they are.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import optuna


def _atomic_copy(src: Path, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def run_study(name: str, ckpt_dir: Path, objective, n_trials: int, seed: int) -> optuna.Study:
    """Create or resume an Optuna study and run it until it has n_trials COMPLETE trials."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    local_dir = Path(os.environ.get("ECOPE_LOCAL_TMP", ckpt_dir))
    local_dir.mkdir(parents=True, exist_ok=True)
    ckpt_db, local_db = ckpt_dir / f"{name}.db", local_dir / f"{name}.db"
    if local_db != ckpt_db and ckpt_db.exists() and not local_db.exists():
        shutil.copy2(ckpt_db, local_db)                     # resume from the checkpoint copy
    study = optuna.create_study(study_name=name, storage=f"sqlite:///{local_db.as_posix()}",
                                load_if_exists=True, direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)

    def save(study_, trial_):
        if local_db != ckpt_db:
            _atomic_copy(local_db, ckpt_db)

    if done < n_trials:
        study.optimize(objective, n_trials=n_trials - done, callbacks=[save])
    return study


def cached_refit(path: Path, fn):
    """Return the saved result of a refit, or compute it with fn() and save it.
    fn must return (dict of numpy arrays, dict of JSON-serialisable info)."""
    path = Path(path)
    assert "." not in path.name, f"checkpoint name must not contain a dot (suffix clash): {path.name}"
    if path.with_suffix(".json").exists() and path.with_suffix(".npz").exists():
        arrs = dict(np.load(path.with_suffix(".npz")))
        return arrs, json.load(open(path.with_suffix(".json")))
    path.parent.mkdir(parents=True, exist_ok=True)
    arrs, info = fn()
    np.savez_compressed(path.with_suffix(".npz"), **arrs)
    tmp = path.with_suffix(".json.tmp")
    json.dump(info, open(tmp, "w"))
    os.replace(tmp, path.with_suffix(".json"))             # json written last = refit complete
    return arrs, info
