"""Output helpers for INEUS-only runs (no training side effects)."""

from pathlib import Path


def save_ineus_result(result, output_dir):
    """Save the INEUS history and summary independently of baseline results.

The summary intentionally uses last-logged MAE, as in the comparison scripts.
This does not reevaluate or change the returned network/checkpoint.
"""
    import matplotlib.pyplot as plt
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = pd.DataFrame(result["history"])
    if history.empty:
        raise ValueError("INEUS produced no logged evaluation points.")
    summary = pd.DataFrame([{
        "final_mae_ineus": float(history["mae"].iloc[-1]),
        "best_logged_mae_ineus": float(history["mae"].min()),
        "best_loss_ineus": float(result["best_loss"]),
        "runtime_ineus": float(history["time"].iloc[-1]),
        "final_logged_epoch": int(history["epoch"].iloc[-1]),
    }])
    history.to_csv(output_dir / "ineus_history.csv", index=False)
    summary.to_csv(output_dir / "ineus_summary.csv", index=False)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(history["time"], history["mae"], marker="o", markersize=3, label="INEUS")
    axis.set_xlabel("Training time (s)")
    axis.set_ylabel("Test MAE")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "mae_vs_time_ineus.png", dpi=200)
    plt.close(figure)
    return {"history": history, "summary": summary}
