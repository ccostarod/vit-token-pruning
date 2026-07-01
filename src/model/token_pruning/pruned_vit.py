import timm
import torch
import torch.nn as nn

from model.token_pruning.strategies import create_pruning_strategy


class PrunedViT(nn.Module):
    def __init__(
        self,
        num_classes: int = 102,
        pretrained: bool = True,
        model_name: str = "vit_base_patch16_224.augreg_in21k",
        prune_layers: list[int] | None = None,
        keep_ratios: list[float] | None = None,
        score_method: str = "token_norm",
        pruning_method: str = "topk",
        history_config: dict | None = None,
        preserve_order: bool = True,
        track_token_indices: bool = False,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )

        self.prune_layers = prune_layers or []
        self.keep_ratios = keep_ratios or []
        self.score_method = score_method
        self.pruning_method = pruning_method
        self.preserve_order = preserve_order
        self.track_token_indices = track_token_indices
        self.last_token_counts = []
        self.last_keep_indices = []

        self.pruning_strategy = create_pruning_strategy(
            pruning_method=pruning_method,
            prune_layers=self.prune_layers,
            keep_ratios=self.keep_ratios,
            score_method=score_method,
            preserve_order=preserve_order,
            history_config=history_config,
        )

        self._validate_timm_vit()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self._prepare_tokens(images)

        patch_indices = None
        if self.track_token_indices:
            patch_indices = torch.arange(
                x.size(1) - 1,
                device=x.device,
            ).unsqueeze(0).expand(x.size(0), -1)

        self.last_token_counts = [x.size(1)]
        self.last_keep_indices = []

        if hasattr(self.pruning_strategy, "reset"):
            self.pruning_strategy.reset()

        for layer_idx, block in enumerate(self.backbone.blocks):
            x = block(x)
            x, keep_indices = self.pruning_strategy.step(layer_idx, x)

            if keep_indices is not None and self.track_token_indices:
                patch_indices = patch_indices.gather(dim=1, index=keep_indices)
                self.last_keep_indices.append(patch_indices.detach().cpu())

            self.last_token_counts.append(x.size(1))

        x = self.backbone.norm(x)

        return self.backbone.forward_head(x)

    def _prepare_tokens(self, images: torch.Tensor) -> torch.Tensor:
        x = self.backbone.patch_embed(images)

        if hasattr(self.backbone, "_pos_embed"):
            x = self.backbone._pos_embed(x)
        else:
            cls_token = self.backbone.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = x + self.backbone.pos_embed

        if hasattr(self.backbone, "patch_drop"):
            x = self.backbone.patch_drop(x)

        if hasattr(self.backbone, "norm_pre"):
            x = self.backbone.norm_pre(x)

        return x

    def _validate_timm_vit(self):
        required_attrs = [
            "patch_embed",
            "blocks",
            "norm",
            "forward_head",
        ]

        missing_attrs = [
            attr for attr in required_attrs
            if not hasattr(self.backbone, attr)
        ]

        if missing_attrs:
            raise ValueError(
                "O modelo timm escolhido nao parece ser um ViT compativel. "
                f"Atributos ausentes: {missing_attrs}"
            )


def create_pruned_vit_model(
    num_classes: int = 102,
    pretrained: bool = True,
    model_name: str = "vit_base_patch16_224.augreg_in21k",
    prune_layers: list[int] | None = None,
    keep_ratios: list[float] | None = None,
    score_method: str = "token_norm",
    pruning_method: str = "topk",
    history_config: dict | None = None,
    preserve_order: bool = True,
    track_token_indices: bool = False,
):
    return PrunedViT(
        num_classes=num_classes,
        pretrained=pretrained,
        model_name=model_name,
        prune_layers=prune_layers,
        keep_ratios=keep_ratios,
        score_method=score_method,
        pruning_method=pruning_method,
        history_config=history_config,
        preserve_order=preserve_order,
        track_token_indices=track_token_indices,
    )
