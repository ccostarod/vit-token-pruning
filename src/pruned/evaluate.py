import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.dataset import get_test_dataset, get_test_dataloader
from evaluate import evaluate_model, save_confusion_matrix_outputs
from model.pruned_vit import create_pruned_vit_model
from utils import load_config, get_device, create_dir, load_checkpoint

DEFAULT_CONFIG_PATH = str(Path(__file__).resolve().parents[2] / "configs" / "pruning" / "pruning_topk.yaml")


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

    pruning_config = config["pruning"]
    prune_layers = pruning_config["prune_layers"]
    keep_ratios = pruning_config["keep_ratios"]
    score_method = pruning_config.get("score_method", "token_norm")

    device = get_device()

    print(f"Experimento: {experiment_name}")
    print(f"Config usada: {config_path}")
    print(f"Dispositivo usado: {device}")

    create_dir(results_dir)

    checkpoint_path = checkpoint_dir / f"{experiment_name}_best.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint nao encontrado: {checkpoint_path}\n"
            "Verifique se src/pruned/train.py ja foi executado e salvou o melhor modelo."
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

    model = create_pruned_vit_model(
        num_classes=num_classes,
        pretrained=False,
        model_name=model_name,
        prune_layers=prune_layers,
        keep_ratios=keep_ratios,
        score_method=score_method,
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

    num_tokens_final = model.last_token_counts[-1] if model.last_token_counts else None

    results = {
        "experiment_name": experiment_name,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "best_validation_metric": float(checkpoint["best_metric"]),
        "label_smoothing": float(label_smoothing),
        "pruning_method": pruning_config["method"],
        "score_method": score_method,
        "prune_layers": prune_layers,
        "keep_ratios": keep_ratios,
        "num_tokens_final": num_tokens_final,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
    }

    print("\nResultados no teste:")
    print(f"Test Loss: {results['test_loss']:.4f}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Test Macro-F1: {results['test_macro_f1']:.4f}")
    print(f"Tokens finais: {num_tokens_final}")

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
