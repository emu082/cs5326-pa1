import argparse
import torch

from src.data import load_token_array, get_batch
from src.model import TransformerLM
from src.loss import cross_entropy
from src.optim import AdamW, get_lr_cosine_schedule, gradient_clipping
from src.checkpoint import save_checkpoint, load_checkpoint


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--val_path", required=True)
    parser.add_argument("--checkpoint_path", default="checkpoint.pt")
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--vocab_size", type=int, default=8192)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--n_q_heads", type=int, default=16)
    parser.add_argument("--n_kv_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10_000.0)

    parser.add_argument("--sequence_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--num_steps", type=int, default=10_000)

    parser.add_argument("--lr_max", type=float, default=3e-4)
    parser.add_argument("--lr_min", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--cosine_steps", type=int, default=9_999)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--checkpoint_interval", type=int, default=500)
    parser.add_argument("--num_val_batches", type=int, default=20)

    parser.add_argument("--train_seed", type=int, default=1)
    parser.add_argument("--val_seed", type=int, default=2)

    return parser.parse_args()


def evaluate(model, val_tokens, batch_size, seq_len, device, val_generator, num_batches):
    was_training = model.training
    model.eval()

    total_loss = 0.0
    with torch.inference_mode():
        for _ in range(num_batches):
            x, y = get_batch(val_tokens, batch_size, seq_len, device, val_generator)
            loss = cross_entropy(model(x), y)
            total_loss += loss.item()

    model.train(was_training)
    return total_loss / num_batches


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_tokens = load_token_array(args.train_path)
    val_tokens = load_token_array(args.val_path)

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        n_q_heads=args.n_q_heads,
        n_kv_heads=args.n_kv_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr_max,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    train_generator = torch.Generator().manual_seed(args.train_seed)
    val_generator = torch.Generator().manual_seed(args.val_seed)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.checkpoint_path, model, optimizer, train_generator, val_generator)

    for step in range(start_step, args.num_steps):
        model.train()

        lr = get_lr_cosine_schedule(step, args.lr_max, args.lr_min, args.warmup_steps, args.cosine_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad()
        train_loss = 0.0

        for _ in range(args.grad_accum_steps):
            x, y = get_batch(train_tokens, args.batch_size, args.sequence_length, device, train_generator)
            logits = model(x)
            loss = cross_entropy(logits, y)
            (loss / args.grad_accum_steps).backward()
            train_loss += loss.detach().item()

        train_loss /= args.grad_accum_steps
        grad_norm = gradient_clipping(model.parameters(), args.max_grad_norm)
        optimizer.step()

        completed = step + 1
        is_last = completed == args.num_steps
        do_eval = is_last or completed % args.eval_interval == 0
        do_log = do_eval or completed % args.log_interval == 0
        do_checkpoint = is_last or completed % args.checkpoint_interval == 0

        val_loss = None
        if do_eval:
            val_loss = evaluate(
                model, val_tokens, args.batch_size, args.sequence_length, device, val_generator, args.num_val_batches
            )

        if do_log:
            line = f"step {completed}/{args.num_steps}  lr {lr:.6f}  train_loss {train_loss:.4f}  grad_norm {grad_norm:.4f}"
            if val_loss is not None:
                line += f"  val_loss {val_loss:.4f}"
            print(line)

        if do_checkpoint:
            save_checkpoint(model, optimizer, completed, train_generator, val_generator, args.checkpoint_path)


if __name__ == "__main__":
    main()