import sys
import json
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from data.dataset import get_test_dataset, get_test_dataloader
from model.vit import create_vit_model
from metrics import (
    calculate_metrics,
    get_predictions_and_targets,
    calculate_confusion_matrix,
)
from utils import load_config, get_device, create_dir, load_checkpoint

DEFAULT_CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml")


def evaluate_model(model, test_loader, criterion, device):
    model.eval()

    running_loss = 0.0
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Avaliando", leave=False)

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size

            all_outputs.append(outputs.detach().cpu())
            all_targets.append(labels.detach().cpu())

    test_loss = running_loss / len(test_loader.dataset)

    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = calculate_metrics(all_outputs, all_targets)

    y_pred, y_true = get_predictions_and_targets(
        outputs=all_outputs,
        targets=all_targets,
    )

    return test_loss, metrics, y_true, y_pred

def save_confusion_matrix_outputs(y_true, y_pred, output_dir, num_classes):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = list(range(num_classes))

    cm = calculate_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        normalize=None,
    )

    cm_normalized = calculate_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        normalize="true",
    )

    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        output_dir / "confusion_matrix.csv"
    )

    pd.DataFrame(cm_normalized, index=labels, columns=labels).to_csv(
        output_dir / "confusion_matrix_normalized.csv"
    )

    plt.figure(figsize=(16, 14))
    plt.imshow(cm_normalized, interpolation="nearest")
    plt.title("Normalized confusion matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.colorbar()

    tick_step = max(1, num_classes // 20)
    ticks = list(range(0, num_classes, tick_step))

    plt.xticks(ticks, ticks, rotation=90)
    plt.yticks(ticks, ticks)

    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix_normalized.png", dpi=300)
    plt.close()

def main(config_path):
    config = load_config(config_path)

    experiment_name = config["experiment"]["name"]

    data_dir = config["paths"]["data_dir"]
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    results_dir = Path(config["paths"]["results_dir"]) / "test"

    image_size = config["dataset"]["image_size"]
    num_classes = config["dataset"]["num_classes"]
    batch_size = config["dataset"]["batch_size"]
    num_workers = config["dataset"]["num_workers"]
    download = config["dataset"]["download"]

    model_name = config["model"]["name"]
    label_smoothing = config["training"].get("label_smoothing", 0.0)

    device = get_device()

    print(f"Experimento: {experiment_name}")
    print(f"Config usada: {config_path}")
    print(f"Dispositivo usado: {device}")

    create_dir(results_dir)

    checkpoint_path = checkpoint_dir / f"{experiment_name}_best.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint não encontrado: {checkpoint_path}\n"
            "Verifique se o train.py já foi executado e salvou o melhor modelo."
        )

    test_dataset = get_test_dataset(
        data_dir=data_dir,
        image_size=image_size,
        download=download,
    )

    test_loader = get_test_dataloader(
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    print(f"Tamanho do teste: {len(test_dataset)}")

    model = create_vit_model(
        num_classes=num_classes,
        pretrained=False,
        model_name=model_name,
    )

    model, checkpoint = load_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    test_loss, test_metrics, y_true, y_pred = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    results = {
        "experiment_name": experiment_name,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "best_validation_metric": float(checkpoint["best_metric"]),
        "label_smoothing": float(label_smoothing),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
    }

    print("\nResultados no teste:")
    print(f"Test Loss: {results['test_loss']:.4f}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Test Macro-F1: {results['test_macro_f1']:.4f}")

    with open(results_dir / "test_metrics.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    pd.DataFrame([results]).to_csv(
        results_dir / "test_metrics.csv",
        index=False,
    )

    save_confusion_matrix_outputs(
        y_true=y_true,
        y_pred=y_pred,
        output_dir=results_dir,
        num_classes=num_classes,
    )

    print(f"\nResultados salvos em: {results_dir}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = DEFAULT_CONFIG_PATH

    main(config_path)


