from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.dataset import get_train_val_datasets, get_train_val_dataloaders
from losses.distillation import distillation_kl_loss
from metrics import calculate_metrics
from model.pruned_vit import create_pruned_vit_model
from model.vit import create_vit_model
from train import train_epoch, validate
from utils import (
    load_config,
    set_seed,
    get_device,
    create_dir,
    save_history_csv,
    save_checkpoint,
    plot_training_curves,
    EarlyStopping,
)

DEFAULT_CONFIG_PATH = str(Path(__file__).resolve().parents[2] / "configs" / "pruning" / "pruning_topk.yaml")


def load_full_vit_checkpoint_into_pruned_model(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.backbone.load_state_dict(checkpoint["model_state_dict"], strict=True)

    return checkpoint


def create_teacher_model(config, checkpoint_path, device):
    model = create_vit_model(
        num_classes=config["dataset"]["num_classes"],
        pretrained=False,
        model_name=config["model"]["name"],
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model, checkpoint


def train_epoch_with_distillation(
    model,
    teacher_model,
    train_loader,
    supervised_criterion,
    optimizer,
    device,
    temperature,
    supervised_weight,
    distillation_weight,
    mixup_fn=None,
):
    model.train()
    teacher_model.eval()

    running_loss = 0.0
    running_supervised_loss = 0.0
    running_distillation_loss = 0.0
    all_outputs = []
    all_targets = []

    progress_bar = tqdm(train_loader, desc="Treinando com KD", leave=False)

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        labels_for_metrics = labels

        optimizer.zero_grad()

        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        student_outputs = model(images)

        with torch.no_grad():
            teacher_outputs = teacher_model(images)

        supervised_loss = supervised_criterion(student_outputs, labels)
        kd_loss = distillation_kl_loss(
            student_logits=student_outputs,
            teacher_logits=teacher_outputs,
            temperature=temperature,
        )
        loss = supervised_weight * supervised_loss + distillation_weight * kd_loss

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_supervised_loss += supervised_loss.item() * batch_size
        running_distillation_loss += kd_loss.item() * batch_size

        all_outputs.append(student_outputs.detach().cpu())
        all_targets.append(labels_for_metrics.detach().cpu())

        progress_bar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "kd": f"{kd_loss.item():.4f}",
            }
        )

    epoch_loss = running_loss / len(train_loader.dataset)
    supervised_epoch_loss = running_supervised_loss / len(train_loader.dataset)
    distillation_epoch_loss = running_distillation_loss / len(train_loader.dataset)

    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)

    metrics = calculate_metrics(all_outputs, all_targets)

    return epoch_loss, supervised_epoch_loss, distillation_epoch_loss, metrics


def main(config: dict):
    experiment_name = config["experiment"]["name"]
    seed = config["experiment"]["seed"]

    data_dir = config["paths"]["data_dir"]
    checkpoint_dir = config["paths"]["checkpoint_dir"]
    results_dir = config["paths"]["results_dir"]

    image_size = config["dataset"]["image_size"]
    num_classes = config["dataset"]["num_classes"]
    batch_size = config["dataset"]["batch_size"]
    num_workers = config["dataset"]["num_workers"]
    download = config["dataset"]["download"]
    use_augmentation = config["dataset"].get("augmentation", False)
    random_erasing_p = config["dataset"].get("random_erasing_p", 0.0)

    model_name = config["model"]["name"]
    pretrained = config["model"]["pretrained"]
    init_checkpoint = config["model"].get("init_checkpoint")

    pruning_config = config["pruning"]
    pruning_method = pruning_config.get("method", "topk")
    prune_layers = pruning_config["prune_layers"]
    keep_ratios = pruning_config["keep_ratios"]
    score_method = pruning_config.get("score_method", "token_norm")
    preserve_order = pruning_config.get("preserve_order", True)
    history_config = pruning_config.get("history")

    epochs = config["training"]["epochs"]
    lr = config["training"]["lr"]
    weight_decay = config["training"]["weight_decay"]
    label_smoothing = config["training"].get("label_smoothing", 0.0)
    mixup_alpha = config["training"].get("mixup_alpha", 0.0)
    early_stopping_config = config["training"].get("early_stopping", {})
    early_stopping_enabled = early_stopping_config.get("enabled", False)
    early_stopping_monitor = early_stopping_config.get("monitor", "val_accuracy")
    distillation_config = config.get("distillation", {})
    distillation_enabled = distillation_config.get("enabled", False)
    distillation_temperature = distillation_config.get("temperature", 2.0)
    supervised_weight = distillation_config.get("supervised_weight", 0.5)
    distillation_weight = distillation_config.get("distillation_weight", 0.5)
    teacher_checkpoint_path = distillation_config.get("teacher_checkpoint", init_checkpoint)

    if distillation_enabled:
        if distillation_temperature <= 0:
            raise ValueError("distillation.temperature deve ser maior que zero.")
        if supervised_weight < 0 or distillation_weight < 0:
            raise ValueError(
                "distillation.supervised_weight e distillation.distillation_weight "
                "devem ser maiores ou iguais a zero."
            )
        if supervised_weight + distillation_weight == 0:
            raise ValueError(
                "A soma de supervised_weight e distillation_weight deve ser maior que zero."
            )

    set_seed(seed)

    device = get_device()
    print(f"Experimento: {experiment_name}")
    print(f"Dispositivo usado: {device}")
    print(f"Pruning: {pruning_method} | score: {score_method}")
    print(f"Prune layers: {prune_layers}")
    print(f"Keep ratios: {keep_ratios}")

    create_dir(checkpoint_dir)
    create_dir(results_dir)

    train_dataset, val_dataset = get_train_val_datasets(
        data_dir=data_dir,
        image_size=image_size,
        download=download,
        use_augmentation=use_augmentation,
        random_erasing_p=random_erasing_p,
    )

    train_loader, val_loader = get_train_val_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    print(f"Tamanho treino: {len(train_dataset)}")
    print(f"Tamanho validacao: {len(val_dataset)}")

    model = create_pruned_vit_model(
        num_classes=num_classes,
        pretrained=pretrained,
        model_name=model_name,
        prune_layers=prune_layers,
        keep_ratios=keep_ratios,
        score_method=score_method,
        pruning_method=pruning_method,
        history_config=history_config,
        preserve_order=preserve_order,
    )

    if init_checkpoint is None:
        raise ValueError("model.init_checkpoint deve apontar para o checkpoint do teacher.")

    teacher_checkpoint = load_full_vit_checkpoint_into_pruned_model(
        model=model,
        checkpoint_path=init_checkpoint,
        device=device,
    )

    print(f"Pesos iniciais carregados de: {init_checkpoint}")
    print(f"Teacher best_metric: {teacher_checkpoint.get('best_metric')}")

    model = model.to(device)
    teacher_model = None

    if distillation_enabled:
        if teacher_checkpoint_path is None:
            raise ValueError(
                "distillation.teacher_checkpoint deve apontar para o checkpoint do teacher."
            )

        teacher_model, distillation_teacher_checkpoint = create_teacher_model(
            config=config,
            checkpoint_path=teacher_checkpoint_path,
            device=device,
        )

        print(
            "Knowledge Distillation habilitado | "
            f"teacher: {teacher_checkpoint_path} | "
            f"temperature: {distillation_temperature} | "
            f"supervised_weight: {supervised_weight} | "
            f"distillation_weight: {distillation_weight}"
        )
        print(
            "KD teacher best_metric: "
            f"{distillation_teacher_checkpoint.get('best_metric')}"
        )

    eval_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    train_criterion = eval_criterion
    mixup_fn = None

    if mixup_alpha > 0.0:
        mixup_fn = Mixup(
            mixup_alpha=mixup_alpha,
            cutmix_alpha=0.0,
            prob=1.0,
            switch_prob=0.0,
            mode="batch",
            label_smoothing=label_smoothing,
            num_classes=num_classes,
        )
        train_criterion = SoftTargetCrossEntropy()

        print(
            "MixUp habilitado | "
            f"alpha: {mixup_alpha} | "
            "prob: 1.0 | "
            "mode: batch"
        )

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_macro_f1": [],
        "val_macro_f1": [],
        "num_tokens_final": [],
        "train_supervised_loss": [],
        "train_distillation_loss": [],
    }

    best_val_accuracy = 0.0
    checkpoint_path = Path(checkpoint_dir) / f"{experiment_name}_best.pth"
    early_stopping = None

    if early_stopping_enabled:
        early_stopping = EarlyStopping(
            patience=early_stopping_config.get("patience", 7),
            min_delta=early_stopping_config.get("min_delta", 0.0),
            mode=early_stopping_config.get("mode", "max"),
        )

        print(
            "Early stopping habilitado | "
            f"monitor: {early_stopping_monitor} | "
            f"patience: {early_stopping.patience}"
        )

    for epoch in range(1, epochs + 1):
        print(f"\nEpoca {epoch}/{epochs}")

        train_supervised_loss = None
        train_distillation_loss = None

        if distillation_enabled:
            train_loss, train_supervised_loss, train_distillation_loss, train_metrics = (
                train_epoch_with_distillation(
                    model=model,
                    teacher_model=teacher_model,
                    train_loader=train_loader,
                    supervised_criterion=train_criterion,
                    optimizer=optimizer,
                    device=device,
                    temperature=distillation_temperature,
                    supervised_weight=supervised_weight,
                    distillation_weight=distillation_weight,
                    mixup_fn=mixup_fn,
                )
            )
        else:
            train_loss, train_metrics = train_epoch(
                model=model,
                train_loader=train_loader,
                criterion=train_criterion,
                optimizer=optimizer,
                device=device,
                mixup_fn=mixup_fn,
            )

        val_loss, val_metrics = validate(
            model=model,
            val_loader=val_loader,
            criterion=eval_criterion,
            device=device,
        )

        scheduler.step()

        train_accuracy = train_metrics["accuracy"]
        val_accuracy = val_metrics["accuracy"]

        train_macro_f1 = train_metrics["macro_f1"]
        val_macro_f1 = val_metrics["macro_f1"]

        num_tokens_final = model.last_token_counts[-1] if model.last_token_counts else None

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Train Macro-F1: {train_macro_f1:.4f}"
        )

        if distillation_enabled:
            print(
                f"Train Supervised Loss: {train_supervised_loss:.4f} | "
                f"Train KD Loss: {train_distillation_loss:.4f}"
            )

        print(
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"Val Macro-F1: {val_macro_f1:.4f} | "
            f"Tokens finais: {num_tokens_final}"
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)
        history["train_macro_f1"].append(train_macro_f1)
        history["val_macro_f1"].append(val_macro_f1)
        history["num_tokens_final"].append(num_tokens_final)
        history["train_supervised_loss"].append(train_supervised_loss)
        history["train_distillation_loss"].append(train_distillation_loss)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_val_accuracy,
                path=checkpoint_path,
            )

            print(f"Novo melhor modelo salvo em: {checkpoint_path}")

        monitored_values = {
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_macro_f1,
            "val_loss": val_loss,
        }

        if early_stopping_monitor not in monitored_values:
            raise ValueError(
                "Monitor de early stopping invalido: "
                f"{early_stopping_monitor}. Use val_accuracy, val_macro_f1 ou val_loss."
            )

        if early_stopping is not None and early_stopping.step(monitored_values[early_stopping_monitor]):
            print(
                "Early stopping acionado: "
                f"{early_stopping.counter} epocas sem melhora em {early_stopping_monitor}."
            )
            break

    history_csv_path = Path(results_dir) / "training_history.csv"

    save_history_csv(history, history_csv_path)

    plot_training_curves(
        history=history,
        output_dir=results_dir,
    )

    print("\nTreinamento finalizado.")
    print(f"Melhor validation accuracy: {best_val_accuracy:.4f}")
    print(f"Historico salvo em: {history_csv_path}")
    print(f"Graficos salvos em: {results_dir}")

    return best_val_accuracy


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = DEFAULT_CONFIG_PATH

    config = load_config(config_path)

    main(config)
