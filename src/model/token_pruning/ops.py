import math

import torch


def compute_num_keep(num_tokens: int, keep_ratio: float) -> int:
    """Converte uma proporcao de tokens mantidos em uma contagem inteira."""
    if num_tokens <= 0:
        raise ValueError("num_tokens deve ser maior que zero.")

    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("keep_ratio deve estar no intervalo (0, 1].")

    return max(1, min(num_tokens, math.ceil(num_tokens * keep_ratio)))


def gather_tokens(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Seleciona tokens por batch usando indices com shape [B, K]."""
    if tokens.ndim != 3:
        raise ValueError("tokens deve ter shape [batch_size, num_tokens, hidden_dim].")

    if indices.ndim != 2:
        raise ValueError("indices deve ter shape [batch_size, num_keep].")

    if tokens.size(0) != indices.size(0):
        raise ValueError("tokens e indices devem ter o mesmo batch_size.")

    gather_indices = indices.unsqueeze(-1).expand(-1, -1, tokens.size(-1))

    return tokens.gather(dim=1, index=gather_indices)


def topk_keep_indices(
    scores: torch.Tensor,
    keep_ratio: float,
    preserve_order: bool = True,
) -> torch.Tensor:
    """Retorna os indices Top-K dos patch tokens, sem incluir class token."""
    if scores.ndim != 2:
        raise ValueError("scores deve ter shape [batch_size, num_patch_tokens].")

    num_keep = compute_num_keep(
        num_tokens=scores.size(1),
        keep_ratio=keep_ratio,
    )

    keep_indices = torch.topk(
        scores,
        k=num_keep,
        dim=1,
    ).indices

    if preserve_order:
        keep_indices = keep_indices.sort(dim=1).values

    return keep_indices


def topk_prune_tokens(
    x: torch.Tensor,
    scores: torch.Tensor,
    keep_ratio: float,
    preserve_order: bool = True,
    return_indices: bool = False,
) -> torch.Tensor:
    """Remove patch tokens de menor score, preservando sempre o class token.

    `x` deve incluir o class token na posicao zero. `scores` deve conter
    apenas os scores dos patch tokens, portanto tem um token a menos que `x`.
    """
    if x.ndim != 3:
        raise ValueError("x deve ter shape [batch_size, num_tokens, hidden_dim].")

    if scores.ndim != 2:
        raise ValueError("scores deve ter shape [batch_size, num_patch_tokens].")

    if x.size(0) != scores.size(0):
        raise ValueError("x e scores devem ter o mesmo batch_size.")

    num_patch_tokens = x.size(1) - 1

    if scores.size(1) != num_patch_tokens:
        raise ValueError(
            "scores deve conter uma importancia para cada patch token, "
            "sem incluir o class token."
        )

    cls_token = x[:, :1, :]
    patch_tokens = x[:, 1:, :]

    keep_indices = topk_keep_indices(
        scores=scores,
        keep_ratio=keep_ratio,
        preserve_order=preserve_order,
    )

    kept_patch_tokens = gather_tokens(
        tokens=patch_tokens,
        indices=keep_indices,
    )

    pruned_tokens = torch.cat([cls_token, kept_patch_tokens], dim=1)

    if return_indices:
        return pruned_tokens, keep_indices

    return pruned_tokens
