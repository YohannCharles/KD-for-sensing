from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(metrics: dict[str, list[float]], save_path: str | Path) -> None:
    output = Path(save_path)
    output.mkdir(parents=True, exist_ok=True)
    epochs = len(metrics.get("train_loss", []))
    if epochs == 0:
        return
    x = np.arange(1, epochs + 1)
    if "learning_rates" in metrics:
        plt.figure()
        plt.plot(x, metrics["learning_rates"])
        plt.xlabel("Epoch")
        plt.ylabel("Learning rate")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output / "LR_schedule.png")
        plt.close()
    plt.figure()
    if "train_loss" in metrics:
        plt.plot(x, metrics["train_loss"], "-o", label="Train")
    if "val_loss" in metrics:
        plt.plot(x, metrics["val_loss"], "-o", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output / "Loss_curves.png")
    plt.close()
    if "train_acc" in metrics or "val_acc" in metrics:
        plt.figure()
        if "train_acc" in metrics:
            plt.plot(x, metrics["train_acc"], "-o", label="Train")
        if "val_acc" in metrics:
            plt.plot(x, metrics["val_acc"], "-o", label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output / "Accuracy_curves.png")
        plt.close()

