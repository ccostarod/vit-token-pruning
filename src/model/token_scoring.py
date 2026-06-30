import torch
import torch.nn.functional as F


def token_norm_scores(x: torch.Tensor) -> torch.Tensor:
    """Calcula importancia dos patch tokens pela norma L2 do embedding."""
    if x.ndim != 3:
        raise ValueError("x deve ter shape [batch_size, num_tokens, hidden_dim].")

    patch_tokens = x[:, 1:, :]

    return patch_tokens.norm(dim=-1)


def cls_similarity_scores(x: torch.Tensor) -> torch.Tensor:
    """Calcula importancia pela similaridade entre patch tokens e class token."""
    if x.ndim != 3:
        raise ValueError("x deve ter shape [batch_size, num_tokens, hidden_dim].")

    cls_token = x[:, :1, :]
    patch_tokens = x[:, 1:, :]

    cls_token = F.normalize(cls_token, dim=-1)
    patch_tokens = F.normalize(patch_tokens, dim=-1)

    return (patch_tokens * cls_token).sum(dim=-1)
