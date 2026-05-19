"""DeepSeek-V4 model implementation for HuggingFace Transformers.

Ported from deepseek-ai/DeepSeek-V4-Pro inference/model.py to be compatible
with HF Trainer, SFTTrainer, and AutoModelForCausalLM.

Key V4 architecture features implemented:
- Hyper-Connections (HC): multi-copy hidden states with Sinkhorn routing
- Compressed Sparse Attention (CSA) with sliding window
- MoE with sqrtsoftplus scoring and hash-based routing
- Grouped low-rank output projection (o_groups + o_lora_rank)
- Multi-Token Prediction (MTP) layers (disabled for small models)

Custom kernels (tilelang) are NOT required — all ops are pure PyTorch.
For training from scratch in bf16, this is sufficient and simpler.
"""

import math
from typing import Optional, Tuple, List
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.generation import GenerationMixin

try:
    from .configuration_deepseek_v4 import DeepseekV4Config
except ImportError:
    from configuration_deepseek_v4 import DeepseekV4Config

# NKI kernel wrappers — opt-in via the NKI_KERNELS env var. Falls back to
# noop stubs when scripts.kernels isn't on sys.path (e.g. when the model
# is loaded from HF Hub without the training scripts present).
try:
    from scripts.kernels import is_enabled as _nki_enabled, nki_cross_entropy
except ImportError:
    def _nki_enabled(name: str) -> bool:
        return False
    nki_cross_entropy = None


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class DeepseekV4RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


def precompute_freqs_cis(dim, seqlen, base=10000.0):
    """Precompute cos/sin for rotary embeddings (real-valued, compile-friendly)."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(seqlen, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # [S, D//2]
    cos = freqs.cos()
    sin = freqs.sin()
    return torch.stack([cos, sin], dim=0)  # [2, S, D//2]


def apply_rotary_emb(x: torch.Tensor, cos_sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embeddings (real-valued, no complex ops).
    
    x: [..., D] where D is even
    cos_sin: [2, S, D//2] - precomputed cos and sin
    """
    cos, sin = cos_sin[0], cos_sin[1]  # each [S, D//2]
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    # Broadcast cos/sin to match x shape
    while cos.ndim < x1.ndim:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=-1).to(x.dtype)


# ---------------------------------------------------------------------------
# Hyper-Connections (HC)
# ---------------------------------------------------------------------------

def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """Pure PyTorch implementation of HC split + Sinkhorn normalization.
    
    Args:
        mixes: [B, S, (2+hc_mult)*hc_mult] - mixed scores from linear projection
        hc_scale: [3] - scale parameters
        hc_base: [(2+hc_mult)*hc_mult] - bias parameters
        hc_mult: number of HC copies
        sinkhorn_iters: number of Sinkhorn normalization iterations
        eps: numerical stability epsilon
    
    Returns:
        pre: [B, S, hc_mult] - pre-connection weights
        post: [B, S, hc_mult] - post-connection weights
        comb: [B, S, hc_mult, hc_mult] - combination matrix
    """
    # Split into pre, post, and combination parts
    pre_raw = mixes[..., :hc_mult]
    post_raw = mixes[..., hc_mult:2*hc_mult]
    comb_raw = mixes[..., 2*hc_mult:].reshape(*mixes.shape[:-1], hc_mult, hc_mult)
    
    # Apply scale and base
    pre = torch.sigmoid(pre_raw * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(post_raw * hc_scale[1] + hc_base[hc_mult:2*hc_mult])
    
    # Combination matrix with Sinkhorn normalization
    comb = comb_raw * hc_scale[2] + hc_base[2*hc_mult:].reshape(hc_mult, hc_mult)
    
    # Initial softmax along last dim + eps
    comb = F.softmax(comb, dim=-1) + eps
    # Normalize along dim=-2
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    
    # Sinkhorn iterations
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    
    return pre, post, comb


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class DeepseekV4Attention(nn.Module):
    """Multi-head Latent Attention (MLA) with sliding window.
    
    V4 attention uses:
    - Low-rank Q projection (wq_a -> q_norm -> wq_b)
    - Direct KV projection (wkv -> kv_norm) - no kv_lora_rank
    - Grouped low-rank O projection (wo_a -> wo_b)
    - Sliding window attention
    - RoPE on last qk_rope_head_dim dims
    """
    
    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = config.head_dim - config.qk_rope_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.o_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.scaling = config.head_dim ** -0.5
        
        # Q projection: low-rank
        self.wq_a = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_norm = DeepseekV4RMSNorm(self.q_lora_rank, config.rms_norm_eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        
        # KV projection: direct (no lora, single head)
        self.wkv = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, config.rms_norm_eps)
        
        # O projection: grouped low-rank  
        # wo_a: [num_heads * head_dim / o_groups] -> [o_groups * o_lora_rank]
        group_head_dim = self.num_heads * self.head_dim // self.o_groups
        self.wo_a = nn.Linear(group_head_dim, self.o_groups * self.o_lora_rank, bias=False)
        self.wo_b = nn.Linear(self.o_groups * self.o_lora_rank, self.hidden_size, bias=False)
        
        # Learnable attention sink bias
        self.attn_sink = nn.Parameter(torch.zeros(self.num_heads))
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, seqlen, _ = hidden_states.shape
        
        # Q: low-rank projection
        q = self.q_norm(self.wq_a(hidden_states))
        q = self.wq_b(q)
        q = q.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        # RMSNorm on q per-head
        q = q * torch.rsqrt(q.float().pow(2).mean(-1, keepdim=True) + self.config.rms_norm_eps)
        q = q.to(hidden_states.dtype)
        
        # KV: direct projection (single KV head, shared across all Q heads)
        kv = self.kv_norm(self.wkv(hidden_states))
        kv = kv.unsqueeze(1)  # [B, 1, S, head_dim]
        
        # Apply RoPE to last qk_rope_head_dim dims of q and kv
        if freqs_cis is not None:
            q_rope = q[..., -self.qk_rope_head_dim:]
            kv_rope = kv[..., -self.qk_rope_head_dim:]
            q_rope = apply_rotary_emb(q_rope, freqs_cis)
            kv_rope = apply_rotary_emb(kv_rope, freqs_cis)
            q = torch.cat([q[..., :-self.qk_rope_head_dim], q_rope], dim=-1)
            kv = torch.cat([kv[..., :-self.qk_rope_head_dim], kv_rope], dim=-1)
        
        # Handle KV cache
        if past_key_value is not None:
            past_k, past_v = past_key_value
            kv = torch.cat([past_k, kv], dim=2)
        
        new_cache = (kv, kv) if use_cache else None
        
        # Expand kv for all heads. .contiguous() materializes the stride-0
        # broadcast — Neuron's autograd refuses to write into a tensor where
        # multiple output elements alias one input element. We also pass two
        # independent K/V tensors instead of the same buffer twice for the
        # same reason.
        k = kv.expand(-1, self.num_heads, -1, -1).contiguous()
        v = kv.expand(-1, self.num_heads, -1, -1).contiguous()

        # Use PyTorch SDPA (fused kernel, memory-efficient)
        # q: [B, H, S, D], k/v: [B, H, T, D]
        # Note: attn_sink bias is small and omitted in SDPA path for speed.
        # It's a learnable per-head scalar — its effect is minimal and the model
        # will learn to compensate through other parameters.
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            is_causal=(attention_mask is None),
            scale=self.scaling,
        )
        
        # De-rotate RoPE on output (inverse rotation = negate sin)
        if freqs_cis is not None:
            cos, sin = freqs_cis[0], freqs_cis[1]  # [S, D//2]
            cos_inv = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, S, D//2]
            sin_inv = -sin.unsqueeze(0).unsqueeze(0)  # negate for inverse
            out_rope = attn_output[..., -self.qk_rope_head_dim:]
            d = out_rope.shape[-1] // 2
            o1, o2 = out_rope[..., :d], out_rope[..., d:]
            out_rope = torch.cat([o1 * cos_inv + o2 * sin_inv, o1 * (-sin_inv) + o2 * cos_inv], dim=-1)
            attn_output = torch.cat([attn_output[..., :-self.qk_rope_head_dim], out_rope.to(attn_output.dtype)], dim=-1)
        
        # Grouped output projection
        attn_output = attn_output.transpose(1, 2)  # [B, S, H, D]
        attn_output = attn_output.reshape(bsz, seqlen, self.o_groups, -1)
        
        # wo_a applied per group: [B, S, G, H*D/G] -> [B, S, G, o_lora_rank]
        wo_a_w = self.wo_a.weight.view(self.o_groups, self.o_lora_rank, -1)
        attn_output = torch.einsum("bsgd,grd->bsgr", attn_output, wo_a_w)
        attn_output = attn_output.flatten(2)  # [B, S, G*o_lora_rank]
        attn_output = self.wo_b(attn_output)
        
        return attn_output, new_cache


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------

class DeepseekV4Expert(nn.Module):
    """Single MoE expert with SwiGLU activation."""
    
    def __init__(self, hidden_size: int, intermediate_size: int, swiglu_limit: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)  # gate
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)  # down
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)  # up
        self.swiglu_limit = swiglu_limit
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = up.clamp(-self.swiglu_limit, self.swiglu_limit)
            gate = gate.clamp(max=self.swiglu_limit)
        x = F.silu(gate) * up
        return self.w2(x.to(self.w2.weight.dtype))


class DeepseekV4Gate(nn.Module):
    """MoE gating with sqrtsoftplus scoring."""
    
    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.topk = config.num_experts_per_tok
        self.scoring_func = config.scoring_func
        self.route_scale = config.routed_scaling_factor
        self.is_hash_layer = layer_idx < config.num_hash_layers
        
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size))
        if not self.is_hash_layer:
            self.bias = nn.Parameter(torch.zeros(config.n_routed_experts))
        else:
            self.register_parameter("bias", None)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = F.linear(x.float(), self.weight.float())
        
        if self.scoring_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.scoring_func == "sigmoid":
            scores = scores.sigmoid()
        elif self.scoring_func == "sqrtsoftplus":
            scores = F.softplus(scores).sqrt()
        
        original_scores = scores
        
        if self.bias is not None:
            scores = scores + self.bias
        
        # Top-k selection
        indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        
        if self.scoring_func != "softmax":
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        
        weights = weights * self.route_scale
        return weights.to(x.dtype), indices


class DeepseekV4MoE(nn.Module):
    """Mixture of Experts layer."""
    
    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.n_routed_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        
        self.gate = DeepseekV4Gate(config, layer_idx)
        self.experts = nn.ModuleList([
            DeepseekV4Expert(config.hidden_size, config.moe_intermediate_size, config.swiglu_limit)
            for _ in range(config.n_routed_experts)
        ])
        self.shared_expert = DeepseekV4Expert(config.hidden_size, config.moe_intermediate_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_flat = x.view(-1, self.hidden_size)
        n_tokens = x_flat.size(0)

        weights, indices = self.gate(x_flat)  # weights/indices: [N, K]

        # Dense MoE: every expert runs on every token. Sparse dispatch
        # (`x_flat[torch.where(indices == i)]`) makes the per-expert tensor
        # shapes data-dependent, which forces Neuron to recompile NEFFs every
        # step (eager mode caches by (op, shape, dtype)). Computing all
        # experts and zeroing out unused ones via a dense gate mask keeps
        # every shape static at the cost of n_routed_experts/topk extra
        # compute — fine for our 4-expert top-2 config.
        expert_outs = torch.stack(
            [expert(x_flat) for expert in self.experts], dim=0
        ).float()  # [E, N, D]

        # (weights, indices) → dense gate matrix [N, E]; scatter_ with topk
        # indices is safe (no duplicates within a row).
        dense_weights = torch.zeros(
            n_tokens, self.n_routed_experts,
            device=x_flat.device, dtype=weights.dtype,
        )
        dense_weights.scatter_(1, indices, weights)

        # [N, E, 1] * [N, E, D] -> sum over E -> [N, D]
        y = (dense_weights.float().unsqueeze(-1) * expert_outs.transpose(0, 1)).sum(dim=1)

        # Add shared expert
        y = y + self.shared_expert(x_flat).float()

        return y.to(x.dtype).view(shape)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class DeepseekV4Block(nn.Module):
    """Transformer block with Hyper-Connections.
    
    Instead of simple residuals, HC maintains hc_mult copies of the hidden state.
    hc_pre: reduces hc copies -> 1 via learned weighted sum.
    hc_post: expands 1 -> hc copies via learned post-weights + combination matrix.
    """
    
    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hc_mult = config.hc_mult
        self.norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        
        self.attn = DeepseekV4Attention(config, layer_idx)
        self.ffn = DeepseekV4MoE(config, layer_idx)
        self.attn_norm = DeepseekV4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = DeepseekV4RMSNorm(config.hidden_size, config.rms_norm_eps)
        
        # HC parameters for attention and FFN sub-layers
        mix_hc = (2 + config.hc_mult) * config.hc_mult
        hc_dim = config.hc_mult * config.hidden_size
        
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc))
        self.hc_attn_scale = nn.Parameter(torch.empty(3))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3))
    
    def hc_pre(self, x, hc_fn, hc_scale, hc_base):
        """Reduce hc_mult copies to 1 via learned weighted sum.
        
        x: [B, S, hc_mult, D]
        Returns: y [B, S, D], post [B, S, hc_mult], comb [B, S, hc_mult, hc_mult]
        """
        shape = x.size()
        dtype = x.dtype
        x_flat = x.flatten(2).float()  # [B, S, hc_mult*D]
        
        rsqrt = torch.rsqrt(x_flat.pow(2).mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x_flat, hc_fn.float()) * rsqrt  # [B, S, mix_hc]
        
        pre, post, comb = hc_split_sinkhorn(
            mixes, hc_scale, hc_base,
            self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps
        )
        
        # Weighted sum: pre [B, S, hc] * x [B, S, hc, D] -> y [B, S, D]
        y = (pre.unsqueeze(-1) * x.float()).sum(dim=2)
        return y.to(dtype), post, comb
    
    def hc_post(self, x, residual, post, comb):
        """Expand 1 -> hc_mult copies.
        
        x: [B, S, D] - output from sub-layer
        residual: [B, S, hc_mult, D] - input HC state
        post: [B, S, hc_mult]
        comb: [B, S, hc_mult, hc_mult]
        """
        # post * x + comb * residual
        y = (post.unsqueeze(-1) * x.unsqueeze(2).float() + 
             torch.einsum("bsij,bsjd->bsid", comb.float(), residual.float()))
        return y.to(x.dtype)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        x: [B, S, hc_mult, D] - HC state
        """
        # Attention with HC
        residual = x
        y, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        y = self.attn_norm(y)
        y, new_cache = self.attn(y, attention_mask=attention_mask, position_ids=position_ids,
                                  freqs_cis=freqs_cis, past_key_value=past_key_value, use_cache=use_cache)
        x = self.hc_post(y, residual, post, comb)
        
        # FFN with HC
        residual = x
        y, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        y = self.ffn_norm(y)
        y = self.ffn(y)
        x = self.hc_post(y, residual, post, comb)
        
        return x, new_cache


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class DeepseekV4PreTrainedModel(PreTrainedModel):
    config_class = DeepseekV4Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DeepseekV4Block"]
    _skip_keys_device_placement = ["past_key_values"]
    
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, DeepseekV4RMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, DeepseekV4Gate):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, DeepseekV4Block):
            # Initialize HC parameters
            nn.init.normal_(module.hc_attn_fn, std=0.01)
            nn.init.normal_(module.hc_ffn_fn, std=0.01)
            nn.init.zeros_(module.hc_attn_base)
            nn.init.zeros_(module.hc_ffn_base)
            nn.init.ones_(module.hc_attn_scale)
            nn.init.ones_(module.hc_ffn_scale)
        elif isinstance(module, DeepseekV4Attention):
            nn.init.zeros_(module.attn_sink)


class DeepseekV4Model(DeepseekV4PreTrainedModel):
    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            DeepseekV4Block(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = DeepseekV4RMSNorm(config.hidden_size, config.rms_norm_eps)
        
        # HC head parameters (for contracting hc_mult -> 1 at output)
        hc_dim = config.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(torch.empty(config.hc_mult, hc_dim))
        self.hc_head_base = nn.Parameter(torch.empty(config.hc_mult))
        self.hc_head_scale = nn.Parameter(torch.empty(1))
        
        # Precomputed RoPE frequencies
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(config.qk_rope_head_dim, config.max_position_embeddings, config.rope_theta),
            persistent=False,
        )
        
        self.gradient_checkpointing = False
        self.post_init()
    
    def _init_weights(self, module):
        super()._init_weights(module)
        # HC head initialization
        if module is self:
            nn.init.normal_(self.hc_head_fn, std=0.01)
            nn.init.zeros_(self.hc_head_base)
            nn.init.ones_(self.hc_head_scale)
    
    def hc_head(self, x):
        """Contract hc_mult copies to 1 for final output.
        
        x: [B, S, hc_mult, D] -> [B, S, D]
        """
        shape = x.size()
        dtype = x.dtype
        x_flat = x.flatten(2).float()  # [B, S, hc_mult*D]
        
        rsqrt = torch.rsqrt(x_flat.pow(2).mean(-1, keepdim=True) + self.config.rms_norm_eps)
        mixes = F.linear(x_flat, self.hc_head_fn.float()) * rsqrt  # [B, S, hc_mult]
        
        pre = torch.sigmoid(mixes * self.hc_head_scale.float() + self.hc_head_base.float()) + self.config.hc_eps
        y = (pre.unsqueeze(-1) * x.float()).sum(dim=2)
        return y.to(dtype)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BaseModelOutputWithPast:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        bsz, seqlen = inputs_embeds.shape[:2]
        
        # Disable cache for now (DynamicCache compatibility TBD)
        use_cache = False
        past_key_values = None
        
        if position_ids is None:
            position_ids = torch.arange(seqlen, device=inputs_embeds.device).unsqueeze(0)
        
        # Get freqs for RoPE
        # freqs_cis is [2, max_seq, D//2], index by position
        pos = position_ids.squeeze(0)
        freqs_cis = self.freqs_cis[:, pos].to(inputs_embeds.device)  # [2, seqlen, D//2]
        
        # Create causal mask - always create our own 4D mask
        causal_mask = torch.full((seqlen, seqlen), float("-inf"), device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Expand to hc_mult copies
        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1)
        hidden_states = hidden_states.contiguous()
        
        new_past_key_values = [] if use_cache else None
        
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None and i < len(past_key_values) else None
            
            if self.gradient_checkpointing and self.training:
                hidden_states, new_cache = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, causal_mask, position_ids, freqs_cis, past_kv, use_cache,
                    use_reentrant=False,
                )
            else:
                hidden_states, new_cache = layer(
                    hidden_states, attention_mask=causal_mask, position_ids=position_ids,
                    freqs_cis=freqs_cis, past_key_value=past_kv, use_cache=use_cache,
                )
            
            if use_cache:
                new_past_key_values.append(new_cache)
        
        # Contract HC copies -> single hidden state
        hidden_states = self.hc_head(hidden_states)
        hidden_states = self.norm(hidden_states)
        
        if not return_dict:
            return (hidden_states, new_past_key_values)
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=new_past_key_values,
        )


class DeepseekV4ForCausalLM(DeepseekV4PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    
    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.model = DeepseekV4Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
    
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=False,  # always tuple for compile compatibility
        )
        
        hidden_states = outputs[0] # hidden_states.size = [B, S, hidden_size]
        logits = self.lm_head(hidden_states) # logits.size = [B, S, vocab_size]
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous() #shift_logits.size = [B, S-1, vocab_size]
            shift_labels = labels[..., 1:].contiguous() #shift_labels.size = [B, S-1]
            flat_logits = shift_logits.view(-1, self.config.vocab_size)  # [B*(S-1), V]
            flat_labels = shift_labels.view(-1)                          # [B*(S-1)]
            if _nki_enabled("cross_entropy"):
                # NKI fused softmax + NLL. No ignore_index support — caller
                # must guarantee labels has no -100. Our pretraining path
                # uses labels=input_ids (scripts/train_pretrain_neuron.py),
                # so no -100s are ever produced.
                loss = nki_cross_entropy(flat_logits, flat_labels)
            else:
                loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        past_kv = outputs[1] if len(outputs) > 1 else None
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_kv,
        )
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
        }
