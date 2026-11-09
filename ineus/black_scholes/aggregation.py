"""Aggregate prediction and loss plots across Black--Scholes run directories.

Means and population standard deviations are computed across independent runs.
Only epochs present in every loaded run are plotted. Checkpoints use pickle;
load only trusted files. No files are read and no plots are made at import time.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import pickle
import re

import numpy as np


def _seed_directories(base_dir, seeds):
    """Resolve repeated-run directories, retaining the standalone seed layout."""
    base_dir = Path(base_dir)
    summary_path = base_dir / "run_summaries.csv"
    mapped = {}
    if summary_path.is_file():
        with summary_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"run", "seed"}.issubset(reader.fieldnames or ()):
                raise ValueError(f"{summary_path} must contain run and seed columns.")
            for row in reader:
                try:
                    run_index, seed = int(row["run"]), int(row["seed"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{summary_path} contains an invalid run/seed pair.") from exc
                if run_index < 1 or not 0 <= seed < 2**32 or seed in mapped:
                    raise ValueError(f"{summary_path} contains invalid or duplicate run/seed metadata.")
                mapped[seed] = base_dir / f"run_{run_index:02d}"
    directories = {}
    for seed in seeds:
        run_dir = mapped.get(seed)
        directories[seed] = (run_dir if run_dir is not None and run_dir.is_dir()
                             else base_dir / f"seed_{seed}")
    return directories


def load_histories(base_dir, seeds, checkpoint=None):
    """Load final (or latest epoch) history for each available seed."""
    histories = {}
    for seed, seed_dir in _seed_directories(base_dir, seeds).items():
        if checkpoint is not None:
            candidates = [seed_dir / checkpoint]
        else:
            epochs = sorted(seed_dir.glob("epoch_*.pkl"), reverse=True)
            candidates = [seed_dir / "final.pkl", *epochs]
        for path in candidates:
            if path.is_file():
                with path.open("rb") as stream:
                    histories[seed] = pickle.load(stream)["history"]
                print(f"Loaded seed {seed} from {path.name}")
                break
        else:
            print(f"WARNING: no checkpoint found for seed {seed}")
    if not histories:
        raise ValueError(f"No histories found under {base_dir}.")
    return histories


def _align_series(series):
    """Align epoch-value mappings and return epochs plus run-by-epoch values."""
    if not series:
        raise ValueError("At least one run is required.")
    common = set.intersection(*(set(values) for values in series))
    if not common:
        raise ValueError("The runs have no common recorded epochs.")
    epochs = np.asarray(sorted(common), dtype=int)
    values = np.asarray([[run[epoch] for epoch in epochs] for run in series])
    return epochs, values


def align_histories(histories, key="student_pred"):
    """Align checkpoint histories by actual distillation epoch."""
    series = []
    for history in histories.values():
        epochs = history["distill_epoch"]
        values = history[key]
        if len(epochs) != len(values):
            raise ValueError(f"History field {key!r} has inconsistent epoch count.")
        series.append(dict(zip(epochs, values)))
    return _align_series(series)


def load_logged_losses(base_dir, seeds):
    """Read teacher/student logs, using the last record for repeated epochs."""
    train_runs, test_runs = [], []
    for seed, seed_dir in _seed_directories(base_dir, seeds).items():
        paths = sorted(seed_dir.glob("*.out"))
        if not paths:
            raise ValueError(f"No .out logs in {seed_dir}; use --prediction-only.")
        train, test = {}, {}
        for path in paths:
            with path.open(encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    if "[Teacher @" in line:
                        match = re.search(r"epoch (\d+).*train_loss=([\d.eE+\-]+)", line)
                        if match:
                            train[int(match.group(1))] = float(match.group(2))
                    elif "[Student @" in line:
                        match = re.search(r"epoch (\d+).*test_loss=([\d.eE+\-]+)", line)
                        if match:
                            test[int(match.group(1))] = float(match.group(2))
        train_runs.append(train)
        test_runs.append(test)
    train_epochs, train_values = _align_series(train_runs)
    test_epochs, test_values = _align_series(test_runs)
    return train_epochs, train_values, test_epochs, test_values


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "mathtext.fontset": "dejavusans", "font.size": 9,
        "axes.labelsize": 10, "axes.titlesize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
        "figure.dpi": 150, "savefig.dpi": 300,
        "axes.linewidth": 0.8, "lines.linewidth": 1.5,
    })
    return plt


def plot_mean_std(ax, x, y, color, label, ls="-", alpha_fill=0.15):
    """Plot the arithmetic mean and one population-standard-deviation band."""
    mu = y.mean(axis=0)
    sd = y.std(axis=0)
    ax.semilogy(x, mu, ls, color=color, lw=1.5, label=label)
    ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=alpha_fill)


def plot_losses(train_epochs, train_loss, test_epochs, test_loss, save_path):
    """Plot logged training loss and student diagnostic, mean plus/minus std."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    plot_mean_std(ax, train_epochs, train_loss, "tab:blue", "Train loss")
    plot_mean_std(ax, test_epochs, test_loss, "tab:orange", "Test loss")
    ax.set_xlabel("Epoch k")
    ax.set_ylabel("Loss")
    ax.legend(loc="best", framealpha=0.85, edgecolor="0.7")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {save_path}")


def plot_predictions(histories, save_path, u_ref=55.810):
    """Plot the across-run mean prediction and one-standard-deviation band."""
    epochs, preds = align_histories(histories)
    mu = preds.mean(axis=0)
    sd = preds.std(axis=0)
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(epochs, mu, "o-", color="navy", ms=3, lw=1.2,
            label=f"Mean ({len(histories)} runs)")
    ax.fill_between(epochs, mu - sd, mu + sd, color="navy", alpha=0.15)
    if u_ref is not None:
        ax.axhline(u_ref, color="firebrick", ls="--", lw=1, alpha=0.7,
                   label=f"Reference = {u_ref:.3f}")
    ax.set_xlabel("Epoch k")
    ax.set_ylabel(r"$u(0,\,x_0)$")
    ax.legend(loc="best", framealpha=0.85, edgecolor="0.7")
    print(f"Final u(0,x₀): {mu[-1]:.4f} ± {sd[-1]:.4f}")
    if u_ref is not None:
        print(f"Absolute error of mean prediction: {abs(mu[-1] - u_ref):.4e}")
    print(f"Individual finals: {preds[:, -1]}")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {save_path}")
    return epochs, mu, sd


def main(argv=None):
    """Load mapped run or seed directories and save aggregate figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--checkpoint", help="Checkpoint filename within each seed directory")
    parser.add_argument("--output-dir", type=Path, default=Path("figures/black_scholes"))
    parser.add_argument("--config", type=Path, help="YAML used for training; provides u_ref")
    parser.add_argument("--u-ref", type=float, help="Override the reference (55.810 without YAML)")
    parser.add_argument("--prediction-only", action="store_true", help="Do not read .out loss logs")
    args = parser.parse_args(argv)
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must not contain duplicates.")
    if args.checkpoint is not None and Path(args.checkpoint).name != args.checkpoint:
        parser.error("--checkpoint must be a filename, not a directory path.")
    reference = 55.810
    if args.config is not None:
        from ..configuration import from_document, read_yaml, validate
        run = from_document(read_yaml(args.config), experiment="black_scholes")
        validate(run)
        reference = run.parameters.evaluation.u_ref
    if args.u_ref is not None:
        reference = args.u_ref
    histories = load_histories(args.base_dir, args.seeds, args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.prediction_only:
        losses = load_logged_losses(args.base_dir, list(histories))
        plot_losses(*losses, save_path=args.output_dir / "fig1a_losses_multirun.png")
    plot_predictions(histories, args.output_dir / "fig1b_prediction_multirun.png", reference)


if __name__ == "__main__":
    main()
