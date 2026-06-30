import sys
from pathlib import Path

import torch

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model.pruned_vit import create_pruned_vit_model


def main():
    model = create_pruned_vit_model(
        num_classes=102,
        pretrained=False,
        prune_layers=[3, 6, 9],
        keep_ratios=[0.9, 0.8, 0.7],
        score_method="token_norm",
    )

    model.eval()

    images = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        logits = model(images)

    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Token counts: {model.last_token_counts}")


if __name__ == "__main__":
    main()
