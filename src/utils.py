import random
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 0.0, mode: str = "max"):
        if mode not in ("min", "max"):
            raise ValueError("mode deve ser 'min' ou 'max'.")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float):
        if self.best_score is None:
            self.best_score = score
            return False

        if self._is_improvement(score):
            self.best_score = score
            self.counter = 0
            return False

        self.counter += 1
        self.should_stop = self.counter >= self.patience

        return self.should_stop

    def _is_improvement(self, score: float):
        if self.mode == "max":
            return score > self.best_score + self.min_delta

        return score < self.best_score - self.min_delta

def create_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_checkpoint(model, optimizer, epoch, best_metric, path):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
    }
    torch.save(checkpoint, path)

def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, checkpoint

def save_history_csv(history, output_path):
    df = pd.DataFrame(history)
    df.to_csv(output_path, index=False)

def plot_training_curves(history, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = history["epoch"]

    # Loss
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "loss_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Accuracy
    plt.figure()
    plt.plot(epochs, history["train_accuracy"], label="Train accuracy")
    plt.plot(epochs, history["val_accuracy"], label="Val accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "accuracy_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Macro-F1
    plt.figure()
    plt.plot(epochs, history["train_macro_f1"], label="Train macro-F1")
    plt.plot(epochs, history["val_macro_f1"], label="Val macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("Training and validation macro-F1")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "macro_f1_curve.png", dpi=300, bbox_inches="tight")
    plt.close()
