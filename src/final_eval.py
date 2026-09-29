import argparse
import math
import torch

from src.data import load_token_array, get_batch
from src.model import TransformerLM
from src.loss import cross_entropy
from src.optim import AdamW
from src.checkpoint import load_checkpoint


def build_model(device):
    return TransformerLM(
        vocab_size=8192, context_length=256, d_model=512, num_layers=4,
        n_q_heads=16, n_kv_heads=4, d_ff=1344, rope_theta=10_000.0, device=device,
    )


def evaluate(model, val_tokens, device):
    gen = torch.Generator().manual_seed(42)
    was_training = model.training
    model.eval()
    total = 0.0
    with torch.inference_mode():
        for _ in range(100):
            x, y = get_batch(val_tokens, 16, 256, device, gen)
            total += cross_entropy(model(x), y).item()
    model.train(was_training)
    mean = total / 100
    return mean, math.exp(mean)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--val_path", required=True)
    p.add_argument("--out", default="final_model.pt")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_tokens = load_token_array(args.val_path)

    model = build_model(device)
    opt = AdamW(model.parameters())
    load_checkpoint(args.checkpoint, model, opt,
                    torch.Generator().manual_seed(0), torch.Generator().manual_seed(0))

    ce, ppl = evaluate(model, val_tokens, device)
    print(f"[fp32 trained weights] val cross-entropy {ce:.4f}  perplexity {ppl:.4f}", flush=True)

    state = {
        name: t.detach().cpu().to(torch.float16) if t.is_floating_point() else t.detach().cpu()
        for name, t in model.state_dict().items()
    }
    torch.save(state, args.out)
    n = sum(t.numel() for t in state.values())
    print(f"saved {args.out}: {n} values (expected 19272192)", flush=True)

    model2 = build_model(device)
    model2.load_state_dict(torch.load(args.out))
    ce2, ppl2 = evaluate(model2, val_tokens, device)
    print(f"[fp16 exported weights] val cross-entropy {ce2:.4f}  perplexity {ppl2:.4f}", flush=True)


if __name__ == "__main__":
    main()