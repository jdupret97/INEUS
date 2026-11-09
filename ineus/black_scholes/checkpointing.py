"""Weight-only checkpoints and training-history serialization."""

import os
import json
import pickle
import numpy as np


def save_checkpoint(path, model_weights, history, epoch):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    data = {
        'model_weights': model_weights,
        'history':       history,
        'epoch':         epoch,
    }
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"  [checkpoint saved → {path}]")


def load_checkpoint(path):
    """Load a trusted pickle checkpoint; never load untrusted pickle files."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_history_json(history, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    clean = {}
    for k, v in history.items():
        if k == 'distill_coeffs':
            clean[k] = v
        else:
            clean[k] = [float(x) if not np.isnan(x) else None for x in v]
    with open(path, 'w') as f:
        json.dump(clean, f, indent=2)
