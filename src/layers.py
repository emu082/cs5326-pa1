import math
import torch
import torch.nn as nn
from einops import einsum


class Linear(nn.Module):

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        weight = torch.empty(out_features, in_features, device=device, dtype=dtype)

        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

        self.weight = nn.Parameter(weight)

    def forward(self, x):
        return einsum(
            x, self.weight,
            "... in_features, out_features in_features -> ... out_features",
        )


class Embedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        weight = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        nn.init.trunc_normal_(weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
        self.weight = nn.Parameter(weight)

    def forward(self, token_ids):
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model, norm_eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.norm_eps = norm_eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x):
        in_dtype = x.dtype
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.to(torch.float32)

        mean_square = einsum(x, x, "... d_model, ... d_model -> ...") / self.d_model
        rms = torch.sqrt(mean_square + self.norm_eps)

        normed = x / rms.unsqueeze(-1)
        result = normed * self.weight

        return result.to(in_dtype)

def silu(x):
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w_gate = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w_up = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w_down = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x):
        gate = silu(self.w_gate(x))
        up = self.w_up(x)
        gated = gate * up
        return self.w_down(gated)