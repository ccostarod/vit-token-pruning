import torch

from model.token_pruning.ops import gather_tokens, topk_prune_tokens


PRUNING_METHODS = {"topk", "hybrid_history"}

DEFAULT_HISTORY_CONFIG = {
    "min_long_history": 2,
    "beta_short": 0.5,
    "beta_long": 0.3,
    "alpha_ema": 0.3,
    "gamma": 0.7,
    "stability_weight": 0.2,
    "normalize_scores": True,
    "eps": 1e-6,
}


def token_norm_scores(x: torch.Tensor) -> torch.Tensor:
    """Calcula importancia dos patch tokens pela norma L2 do embedding."""
    if x.ndim != 3:
        raise ValueError("x deve ter shape [batch_size, num_tokens, hidden_dim].")

    patch_tokens = x[:, 1:, :]

    return patch_tokens.norm(dim=-1)


class TopKPruningStrategy:
    def __init__(
        self,
        prune_layers: list[int],
        keep_ratios: list[float],
        score_method: str = "token_norm",
        preserve_order: bool = True,
    ):
        if score_method != "token_norm":
            raise ValueError("A estrategia atual suporta apenas score_method='token_norm'.")

        if len(prune_layers) != len(keep_ratios):
            raise ValueError("prune_layers e keep_ratios devem ter o mesmo tamanho.")

        self.prune_config = dict(zip(prune_layers, keep_ratios))
        self.score_method = score_method
        self.preserve_order = preserve_order

    def step(self, layer_idx: int, x: torch.Tensor):
        if layer_idx not in self.prune_config:
            return x, None

        scores = token_norm_scores(x)

        return topk_prune_tokens(
            x=x,
            scores=scores,
            keep_ratio=self.prune_config[layer_idx],
            preserve_order=self.preserve_order,
            return_indices=True,
        )


class HybridHistoryPruningStrategy(TopKPruningStrategy):
    def __init__(
        self,
        prune_layers: list[int],
        keep_ratios: list[float],
        score_method: str = "token_norm",
        preserve_order: bool = True,
        history_config: dict | None = None,
    ):
        super().__init__(
            prune_layers=prune_layers,
            keep_ratios=keep_ratios,
            score_method=score_method,
            preserve_order=preserve_order,
        )

        self.history_config = {
            **DEFAULT_HISTORY_CONFIG,
            **(history_config or {}),
        }
        self.score_history = []
        self.ema_scores = None

    def reset(self):
        self.score_history = []
        self.ema_scores = None

    def step(self, layer_idx: int, x: torch.Tensor):
        current_scores = self._prepare_scores(token_norm_scores(x))
        next_ema_scores = self._update_ema_scores(current_scores)
        keep_indices = None

        if layer_idx in self.prune_config:
            pruning_scores = self._get_pruning_scores(
                current_scores=current_scores,
                ema_scores=next_ema_scores,
            )
            x, keep_indices = topk_prune_tokens(
                x=x,
                scores=pruning_scores,
                keep_ratio=self.prune_config[layer_idx],
                preserve_order=self.preserve_order,
                return_indices=True,
            )
            current_scores = self._gather_scores(current_scores, keep_indices)
            next_ema_scores = self._gather_scores(next_ema_scores, keep_indices)
            self.score_history = [
                self._gather_scores(previous_scores, keep_indices)
                for previous_scores in self.score_history
            ]

        self.score_history.append(current_scores)
        self.ema_scores = next_ema_scores

        return x, keep_indices

    def _prepare_scores(self, scores: torch.Tensor) -> torch.Tensor:
        if not self.history_config["normalize_scores"]:
            return scores

        eps = self.history_config["eps"]
        mean = scores.mean(dim=1, keepdim=True)
        std = scores.std(dim=1, keepdim=True, unbiased=False)

        return (scores - mean) / (std + eps)

    def _update_ema_scores(self, current_scores: torch.Tensor) -> torch.Tensor:
        if self.ema_scores is None:
            return current_scores

        gamma = self.history_config["gamma"]

        return gamma * self.ema_scores + (1.0 - gamma) * current_scores

    def _get_pruning_scores(
        self,
        current_scores: torch.Tensor,
        ema_scores: torch.Tensor,
    ) -> torch.Tensor:
        if not self.score_history:
            return current_scores

        previous_scores = self.score_history[-1]
        trend = current_scores - previous_scores

        if len(self.score_history) < self.history_config["min_long_history"]:
            return current_scores + self.history_config["beta_short"] * trend

        stacked_history = torch.stack([*self.score_history, current_scores], dim=0)
        stability = stacked_history.std(dim=0, unbiased=False)

        return (
            current_scores
            + self.history_config["beta_long"] * trend
            + self.history_config["alpha_ema"] * ema_scores
            - self.history_config["stability_weight"] * stability
        )

    def _gather_scores(
        self,
        scores: torch.Tensor,
        keep_indices: torch.Tensor,
    ) -> torch.Tensor:
        return gather_tokens(
            tokens=scores.unsqueeze(-1),
            indices=keep_indices,
        ).squeeze(-1)


def create_pruning_strategy(
    pruning_method: str,
    prune_layers: list[int],
    keep_ratios: list[float],
    score_method: str = "token_norm",
    preserve_order: bool = True,
    history_config: dict | None = None,
):
    if pruning_method == "topk":
        return TopKPruningStrategy(
            prune_layers=prune_layers,
            keep_ratios=keep_ratios,
            score_method=score_method,
            preserve_order=preserve_order,
        )

    if pruning_method == "hybrid_history":
        return HybridHistoryPruningStrategy(
            prune_layers=prune_layers,
            keep_ratios=keep_ratios,
            score_method=score_method,
            preserve_order=preserve_order,
            history_config=history_config,
        )

    raise ValueError(
        f"pruning_method invalido: {pruning_method}. "
        f"Use um de: {sorted(PRUNING_METHODS)}."
    )
