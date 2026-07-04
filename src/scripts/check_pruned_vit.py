import sys
from pathlib import Path

import torch

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model.pruned_vit import create_pruned_vit_model


def run_check(pruning_method, prune_layers, keep_ratios, expected_token_counts):
    model = create_pruned_vit_model(
        num_classes=102,
        pretrained=False,
        prune_layers=prune_layers,
        keep_ratios=keep_ratios,
        score_method="token_norm",
        pruning_method=pruning_method,
        history_config={
            "min_long_history": 2,
            "beta_short": 0.5,
            "beta_long": 0.3,
            "alpha_ema": 0.3,
            "gamma": 0.7,
            "stability_weight": 0.2,
            "normalize_scores": True,
            "eps": 1e-6,
        },
    )

    model.eval()

    images = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        logits = model(images)

    print(f"Metodo: {pruning_method}")
    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Token counts: {model.last_token_counts}")

    assert tuple(logits.shape) == (2, 102)
    assert model.last_token_counts == expected_token_counts


def main():
    run_check(
        pruning_method="topk",
        prune_layers=[3, 6, 9],
        keep_ratios=[0.9, 0.8, 0.7],
        expected_token_counts=[197, 197, 197, 197, 178, 178, 178, 143, 143, 143, 101, 101, 101],
    )

    run_check(
        pruning_method="hybrid_history",
        prune_layers=[1, 3, 6],
        keep_ratios=[0.8, 0.65, 0.55],
        expected_token_counts=[197, 197, 158, 158, 104, 104, 104, 58, 58, 58, 58, 58, 58],
    )

    # Hybrid History fica como estrategia exploratoria; o foco deve seguir em Trend/Class-Aware.
    run_check(
        pruning_method="trend_adjusted",
        prune_layers=[1, 3, 6],
        keep_ratios=[0.8, 0.65, 0.55],
        expected_token_counts=[197, 197, 158, 158, 104, 104, 104, 58, 58, 58, 58, 58, 58],
    )

    run_check(
        pruning_method="class_aware_trend",
        prune_layers=[1, 3, 6],
        keep_ratios=[0.8, 0.65, 0.55],
        expected_token_counts=[197, 197, 158, 158, 104, 104, 104, 58, 58, 58, 58, 58, 58],
    )


if __name__ == "__main__":
    main()
