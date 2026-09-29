import torch


def cross_entropy(logits, targets):
    log_sum_exp = torch.logsumexp(logits, dim=-1)
    target_logits = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    losses = log_sum_exp - target_logits
    return losses.mean()