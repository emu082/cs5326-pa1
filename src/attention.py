import torch
import torch.nn as nn
import math
from einops import einsum, rearrange
from src.layers import Linear, Embedding, RMSNorm, SwiGLU 



class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, rope_theta, head_dim, context_length, device=None):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even")

        self.head_dim = head_dim
        self.context_length = context_length

        half_dim = head_dim // 2
        k = torch.arange(half_dim, device=device)
        inv_freq = rope_theta ** (-(2 * k) / head_dim)

        positions = torch.arange(context_length, device=device)
        angles = einsum(positions.float(), inv_freq, "seq, half -> seq half")

        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(self, x, token_positions):
        if x.shape[-1] != self.head_dim:
            raise ValueError("final dimension of x must equal head_dim")
        if token_positions.dtype not in (torch.int32, torch.int64):
            raise TypeError("token_positions must be integer")
        if token_positions.min() < 0 or token_positions.max() >= self.context_length:
            raise ValueError("token_positions outside valid range")

        extra_dims = x.dim() - token_positions.dim() - 1
        if extra_dims > 0:
            new_shape = token_positions.shape[:-1] + (1,) * extra_dims + (token_positions.shape[-1],)
            token_positions = token_positions.reshape(new_shape)

        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        rotated_1 = x1 * cos - x2 * sin
        rotated_2 = x1 * sin + x2 * cos

        out = torch.stack([rotated_1, rotated_2], dim=-1)
        out = out.flatten(-2)
        return out

def softmax(x, dim):
    x_max = x.max(dim=dim, keepdim=True).values
    x_shifted = x - x_max
    x_exp = torch.exp(x_shifted)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(query, key, value, mask=None):
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key must have the same feature dimension")
    if key.shape[-2] != value.shape[-2]:
        raise ValueError("key and value must have the same sequence length")
    d_k = query.shape[-1]
    scores = einsum(query,key,"... n_q d_k, ... n_kv d_k -> ... n_q n_kv") / math.sqrt(d_k)

    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError("mask must be a boolean tensor")
        if not mask.any(dim=-1).all():
            raise ValueError("mask must have at least one True value in each batch")
        scores = scores.masked_fill(~mask, float("-inf"))

    attn_weights = softmax(scores, dim=-1)
    return einsum(
        attn_weights,
        value,
        "... n_q n_kv, ... n_kv d_v -> ... n_q d_v"
    )

class GroupedQueryAttention(nn.Module): 
    def __init__(self, d_model, n_q_heads, n_kv_heads, context_length, rope_theta, device=None, dtype=None): 
        super().__init__() 
        if d_model % n_q_heads != 0: 
            raise ValueError("d_model must be divisible by n_q_heads") 
        if n_q_heads % n_kv_heads != 0: 
            raise ValueError("n_q_heads must be divisible by n_kv_heads") 
 
        self.n_q_heads = n_q_heads 
        self.n_kv_heads = n_kv_heads 
        self.group_size = n_q_heads // n_kv_heads 
        self.head_dim = d_model // n_q_heads 
 
        self.q_proj = Linear(d_model, n_q_heads * self.head_dim, device=device, dtype=dtype) 
        self.k_proj = Linear(d_model, n_kv_heads * self.head_dim, device=device, dtype=dtype) 
        self.v_proj = Linear(d_model, n_kv_heads * self.head_dim, device=device, dtype=dtype) 
        self.out_proj = Linear(n_q_heads * self.head_dim, d_model, device=device, dtype=dtype) 
 
        self.rope = RotaryPositionalEmbedding(rope_theta, self.head_dim, context_length, device=device) 
 
    def forward(self, x, token_positions=None): 
        batch, seq_len, _ = x.shape 
        if token_positions is None: 
            token_positions = torch.arange(seq_len, device=x.device) 
 
        q = self.q_proj(x) 
        k = self.k_proj(x) 
        v = self.v_proj(x) 
 
        q = rearrange(q, "batch seq (h_kv g d) -> batch h_kv g seq d", h_kv=self.n_kv_heads, g=self.group_size) 
        k = rearrange(k, "batch seq (h_kv d) -> batch h_kv seq d", h_kv=self.n_kv_heads) 
        v = rearrange(v, "batch seq (h_kv d) -> batch h_kv seq d", h_kv=self.n_kv_heads) 
 
        q = self.rope(q, token_positions) 
        k = self.rope(k, token_positions) 
 
        scores = einsum( 
            q, k, "batch h_kv g n_q d, batch h_kv n_kv d -> batch h_kv g n_q n_kv" 
        ) / math.sqrt(self.head_dim) 
 
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)) 
        scores = scores.masked_fill(~causal_mask, float("-inf")) 
 
        attn_weights = softmax(scores, dim=-1) 
 
        out = einsum(attn_weights, v, "batch h_kv g n_q n_kv, batch h_kv n_kv d -> batch h_kv g n_q d") 
        out = rearrange(out, "batch h_kv g n_q d -> batch n_q (h_kv g d)") 
 
        return self.out_proj(out)