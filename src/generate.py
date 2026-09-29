import torch

from src.attention import softmax


def generate(
    model,
    prompt_ids,
    max_new_tokens,
    context_length,
    *,
    temperature=1.0,
    top_p=1.0,
    eot_token_id=None,
    generator=None,
):
    if prompt_ids.numel() == 0:
        raise ValueError("prompt_ids must be non-empty")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not (0 < top_p <= 1):
        raise ValueError("top_p must be in (0, 1]")
    if context_length != model.context_length:
        raise ValueError("context_length must equal model.context_length")

    was_training = model.training
    model.eval()

    sequence = prompt_ids.clone()

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            cropped = sequence[-context_length:]
            logits = model(cropped.unsqueeze(0))
            next_logits = logits[0, -1]

            scaled_logits = next_logits / temperature
            probs = softmax(scaled_logits, dim=-1)

            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            cutoff = torch.searchsorted(cumulative_probs, top_p).item()
            sorted_probs[cutoff + 1:] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum()

            sampled_rank = torch.multinomial(sorted_probs, num_samples=1, generator=generator)
            next_token = sorted_indices[sampled_rank]

            sequence = torch.cat([sequence, next_token])

            if eot_token_id is not None and next_token.item() == eot_token_id:
                break

    model.train(was_training)
    return sequence