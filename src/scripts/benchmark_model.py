import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import timm

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.dataset import get_test_dataset, get_test_dataloader
from metrics import calculate_metrics
from model.pruned_vit import create_pruned_vit_model
from model.vit import create_vit_model
from utils import create_dir, get_device, load_checkpoint, load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mede latencia, throughput, memoria e tokens em inferencia."
    )
    parser.add_argument("config_path", help="Caminho para o arquivo YAML de config.")
    parser.add_argument(
        "--model-type",
        choices=["teacher", "pruned"],
        required=True,
        help="Tipo de modelo a carregar.",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=5,
        help="Numero de batches de aquecimento antes da medicao.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Limita a quantidade de batches medidos. Por padrao mede todo o teste.",
    )
    return parser.parse_args()


def synchronize_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def get_checkpoint_path(config):
    experiment_name = config["experiment"]["name"]
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])

    return checkpoint_dir / f"{experiment_name}_best.pth"


def create_model(config, model_type):
    dataset_config = config["dataset"]
    model_config = config["model"]

    if model_type == "teacher":
        return create_vit_model(
            num_classes=dataset_config["num_classes"],
            pretrained=False,
            model_name=model_config["name"],
        )

    pruning_config = config["pruning"]

    return create_pruned_vit_model(
        num_classes=dataset_config["num_classes"],
        pretrained=False,
        model_name=model_config["name"],
        prune_layers=pruning_config["prune_layers"],
        keep_ratios=pruning_config["keep_ratios"],
        score_method=pruning_config.get("score_method", "token_norm"),
        pruning_method=pruning_config.get("method", "topk"),
        history_config=pruning_config.get("history"),
        preserve_order=pruning_config.get("preserve_order", True),
    )


def estimate_state_dict_size_mb(model):
    total_bytes = 0

    for tensor in model.state_dict().values():
        total_bytes += tensor.numel() * tensor.element_size()

    return total_bytes / (1024**2)


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def get_teacher_token_count(model):
    patch_embed = getattr(model, "patch_embed", None)

    if patch_embed is None or not hasattr(patch_embed, "num_patches"):
        return None

    return int(patch_embed.num_patches + 1)


def benchmark_model(model, test_loader, criterion, device, warmup_batches, max_batches):
    model.eval()

    measured_images = 0
    measured_batches = 0
    evaluated_images = 0
    evaluated_batches = 0
    total_time_seconds = 0.0
    running_loss = 0.0
    all_outputs = []
    all_targets = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            synchronize_if_needed(device)
            start_time = time.perf_counter()
            outputs = model(images)
            synchronize_if_needed(device)
            elapsed = time.perf_counter() - start_time

            loss = criterion(outputs, labels)
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            evaluated_images += batch_size
            evaluated_batches += 1

            all_outputs.append(outputs.detach().cpu())
            all_targets.append(labels.detach().cpu())

            if batch_idx < warmup_batches:
                continue

            total_time_seconds += elapsed
            measured_images += batch_size
            measured_batches += 1

            if max_batches is not None and measured_batches >= max_batches:
                break

    if measured_batches == 0:
        raise ValueError(
            "Nenhum batch foi medido. Reduza --warmup-batches ou aumente o conjunto avaliado."
        )

    peak_memory_mb = None
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    prediction_metrics = calculate_metrics(all_outputs, all_targets)
    evaluation_loss = running_loss / evaluated_images

    return {
        "evaluated_batches": evaluated_batches,
        "evaluated_images": evaluated_images,
        "evaluation_loss": evaluation_loss,
        "evaluation_accuracy": float(prediction_metrics["accuracy"]),
        "evaluation_macro_f1": float(prediction_metrics["macro_f1"]),
        "measured_batches": measured_batches,
        "measured_images": measured_images,
        "total_inference_time_seconds": total_time_seconds,
        "latency_ms_per_image": (total_time_seconds / measured_images) * 1000,
        "latency_ms_per_batch": (total_time_seconds / measured_batches) * 1000,
        "throughput_images_per_second": measured_images / total_time_seconds,
        "peak_memory_mb": peak_memory_mb,
    }


def build_results(config, config_path, model_type, model, checkpoint_path, checkpoint, metrics, args):
    experiment_name = config["experiment"]["name"]

    token_counts = None
    num_tokens_final = None
    token_reduction_ratio = None

    if model_type == "teacher":
        num_tokens_final = get_teacher_token_count(model)
        if num_tokens_final is not None:
            token_counts = [num_tokens_final]
    elif hasattr(model, "last_token_counts") and model.last_token_counts:
        token_counts = [int(value) for value in model.last_token_counts]
        num_tokens_final = token_counts[-1]

        teacher_tokens = get_teacher_token_count(model.backbone)
        if teacher_tokens:
            token_reduction_ratio = 1 - (num_tokens_final / teacher_tokens)

    device = next(model.parameters()).device

    results = {
        "experiment_name": experiment_name,
        "model_type": model_type,
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "best_validation_metric": float(checkpoint["best_metric"]),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "timm_version": timm.__version__,
        "cuda_available": torch.cuda.is_available(),
        "batch_size": int(config["dataset"]["batch_size"]),
        "warmup_batches": int(args.warmup_batches),
        "max_batches": args.max_batches,
        "num_parameters": int(count_parameters(model)),
        "model_size_mb": float(estimate_state_dict_size_mb(model)),
        "token_counts": token_counts,
        "num_tokens_final": num_tokens_final,
        "token_reduction_ratio": token_reduction_ratio,
        **metrics,
    }

    if "pruning" in config:
        results["pruning_method"] = config["pruning"].get("method")
        results["score_method"] = config["pruning"].get("score_method")
        results["prune_layers"] = config["pruning"].get("prune_layers")
        results["keep_ratios"] = config["pruning"].get("keep_ratios")
        results["preserve_order"] = config["pruning"].get("preserve_order", True)
        results["history_config"] = config["pruning"].get("history")

    return results


def save_results(results, output_dir):
    output_dir = Path(output_dir)
    create_dir(output_dir)

    json_path = output_dir / "performance_metrics.json"
    csv_path = output_dir / "performance_metrics.csv"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)

    return json_path, csv_path


def main():
    args = parse_args()
    config = load_config(args.config_path)
    device = get_device()

    experiment_name = config["experiment"]["name"]
    results_dir = Path(config["paths"]["results_dir"]) / "benchmark"
    checkpoint_path = get_checkpoint_path(config)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint nao encontrado: {checkpoint_path}")

    print(f"Experimento: {experiment_name}")
    print(f"Modelo: {args.model_type}")
    print(f"Config usada: {args.config_path}")
    print(f"Dispositivo usado: {device}")

    test_dataset = get_test_dataset(
        data_dir=config["paths"]["data_dir"],
        image_size=config["dataset"]["image_size"],
        download=config["dataset"]["download"],
    )
    test_loader = get_test_dataloader(
        test_dataset=test_dataset,
        batch_size=config["dataset"]["batch_size"],
        num_workers=config["dataset"]["num_workers"],
    )

    model = create_model(config, args.model_type)
    model, checkpoint = load_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=config["training"].get("label_smoothing", 0.0)
    )

    metrics = benchmark_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        warmup_batches=args.warmup_batches,
        max_batches=args.max_batches,
    )

    results = build_results(
        config=config,
        config_path=args.config_path,
        model_type=args.model_type,
        model=model,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        metrics=metrics,
        args=args,
    )

    json_path, csv_path = save_results(results, results_dir)

    print("\nBenchmark finalizado.")
    print(f"Loss: {results['evaluation_loss']:.4f}")
    print(f"Accuracy: {results['evaluation_accuracy']:.4f}")
    print(f"Macro-F1: {results['evaluation_macro_f1']:.4f}")
    print(f"Latencia por imagem: {results['latency_ms_per_image']:.4f} ms")
    print(f"Throughput: {results['throughput_images_per_second']:.2f} imagens/s")
    if results["peak_memory_mb"] is not None:
        print(f"Pico de memoria GPU: {results['peak_memory_mb']:.2f} MB")
    if results["token_counts"] is not None:
        print(f"Tokens: {results['token_counts']}")
    print(f"Resultados salvos em: {json_path}")
    print(f"CSV salvo em: {csv_path}")


if __name__ == "__main__":
    main()
