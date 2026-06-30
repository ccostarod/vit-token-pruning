import timm
import torch
import torch.nn as nn

from model.pruning import topk_prune_tokens
from model.token_scoring import cls_similarity_scores, token_norm_scores


SCORE_FUNCTIONS = {
    "token_norm": token_norm_scores,
    "cls_similarity": cls_similarity_scores,
}


class PrunedViT(nn.Module):
    def __init__(
        self,
        num_classes: int = 102,
        pretrained: bool = True,
        model_name: str = "vit_base_patch16_224.augreg_in21k",
        prune_layers: list[int] | None = None,
        keep_ratios: list[float] | None = None,
        score_method: str = "token_norm",
        track_token_indices: bool = False,
    ):
        super().__init__()

        if score_method not in SCORE_FUNCTIONS:
            raise ValueError(
                f"score_method invalido: {score_method}. "
                f"Use um de: {list(SCORE_FUNCTIONS)}."
            )

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )

        self.prune_layers = prune_layers or []
        self.keep_ratios = keep_ratios or []
        self.score_method = score_method
        self.score_fn = SCORE_FUNCTIONS[score_method]
        self.track_token_indices = track_token_indices
        self.last_token_counts = []
        self.last_keep_indices = []

        if len(self.prune_layers) != len(self.keep_ratios):
            raise ValueError("prune_layers e keep_ratios devem ter o mesmo tamanho.")

        self.prune_config = dict(zip(self.prune_layers, self.keep_ratios))

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

        for layer_idx, block in enumerate(self.backbone.blocks):
            x = block(x)

            if layer_idx in self.prune_config:
                scores = self.score_fn(x)
                if self.track_token_indices:
                    x, keep_indices = topk_prune_tokens(
                        x=x,
                        scores=scores,
                        keep_ratio=self.prune_config[layer_idx],
                        return_indices=True,
                    )
                    patch_indices = patch_indices.gather(dim=1, index=keep_indices)
                    self.last_keep_indices.append(patch_indices.detach().cpu())
                else:
                    x = topk_prune_tokens(
                        x=x,
                        scores=scores,
                        keep_ratio=self.prune_config[layer_idx],
                    )

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
    track_token_indices: bool = False,
):
    return PrunedViT(
        num_classes=num_classes,
        pretrained=pretrained,
        model_name=model_name,
        prune_layers=prune_layers,
        keep_ratios=keep_ratios,
        score_method=score_method,
        track_token_indices=track_token_indices,
    )
