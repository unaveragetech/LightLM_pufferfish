import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import math
import tiktoken
import inspect
import os
from dataclasses import dataclass, field
from huggingface_hub import PyTorchModelHubMixin
from typing import Optional

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


@dataclass
class ModelConfig:
    vocab_size: int

    num_dims: int                       # number of dimensions
    num_heads: int                      # number of query heads
    num_kv_heads: int                   # number of key/value heads
    num_layers: int                     # total transformer layers
    ffn_hidden_dims: int                # hidden dimension for FFN/FFNwMoE

    context_len: int                    # maximum context length
    use_cache: bool                     # enable KV-caching
    use_flash: bool                     # use Flash Attention
    use_moe: bool                       # enable mixture-of-experts

    moe_num_experts: int                # total number of experts
    moe_active_experts: int             # number of experts per token (top_k)
    moe_eps: float = 1e-6               # epsilon for router stability
    moe_aux_loss_coef: float = 0.01     # coefficient for auxiliary loss
    moe_shared_experts: int = 0         # number of shared experts (DeepSeekMoE)
    use_lossfreebalance: bool = False   # use Auxiliary-loss-free load balancing strategy for mixture-of-experts from DeepSeek https://arxiv.org/pdf/2408.15664

    rmsnorm_eps: float = 1e-6
    rope_theta: float = 1e5

    # New configuration options
    activation_fn: str = "silu"         # Options: "silu", "gelu", "relu"
    norm_type: str = "rmsnorm"          # Options: "rmsnorm", "layernorm"

    ffn_dim_multiplier: Optional[int] = None    # optional multiplier to compute ffn_hidden_dims

    def __post_init__(self):
        """Run after initialization to set fast test mode"""
        self.is_fast_test = (
            self.num_layers <= 2 and
            self.num_dims <= 128 and
            self.context_len <= 128
        )

        if self.is_fast_test:
            # Override certain settings for fastest possible execution
            self.use_cache = False
            self.use_flash = False
            self.use_moe = False
            self.activation_fn = "relu"  # Simplest activation
            self.norm_type = "rmsnorm"   # Faster normalization
            self.rope_theta = 1e4        # Smaller positional encoding range


# Helper function for RoPE
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads to match number of query heads."""
    batch_size, num_kv_heads, seq_len, head_dim = x.shape
    return x.unsqueeze(2).expand(batch_size, num_kv_heads, n_rep, seq_len, head_dim).reshape(
        batch_size, num_kv_heads * n_rep, seq_len, head_dim
    )


class Rotary(nn.Module):
    def __init__(self, config):
        super(Rotary, self).__init__()

        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, config.num_dims // config.num_heads, 2).float() / (config.num_dims // config.num_heads)))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self.seq_len_saved = None
        self.cos_saved = None
        self.sin_saved = None

    def forward(self, x, seq_dim=1):
        seq_len = x.size(seq_dim)
        # Only recompute the cosine and sine matrices if the sequence length has changed.
        if seq_len != self.seq_len_saved:
            self.seq_len_saved = seq_len
            pos = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            # Compute the outer product between positions and inverse frequencies.
            freqs = torch.einsum("i,j->ij", pos, self.inv_freq) # (seq_len, inv_freq.shape[0])
            # Duplicate the freqs along the last dimension to create pairs.
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_saved = emb.cos()
            self.sin_saved = emb.sin()

        return self.cos_saved, self.sin_saved


class RMSNorm(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.g = nn.Parameter(torch.ones(config.num_dims))
        self.eps = config.rmsnorm_eps

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.g * self._norm(x.float()).type_as(x)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_cache = config.use_cache
        self.use_flash = config.use_flash

        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_heads if config.num_kv_heads is None else config.num_kv_heads

        self.num_rep = self.num_heads // self.num_kv_heads
        self.head_dim = config.num_dims // self.num_heads

        self.wq = nn.Linear(config.num_dims, config.num_dims, bias=False)
        nn.init.normal_(self.wq.weight, mean=0, std=1/math.sqrt(config.num_dims))
        self.wk = nn.Linear(config.num_dims, self.num_kv_heads * self.head_dim, bias=False)
        nn.init.normal_(self.wk.weight, mean=0, std=1/math.sqrt(config.num_dims))
        self.wv = nn.Linear(config.num_dims, self.num_kv_heads * self.head_dim, bias=False)
        nn.init.normal_(self.wv.weight, mean=0, std=1/math.sqrt(config.num_dims))

        self.wo = nn.Linear(config.num_dims, config.num_dims, bias=False)

        self.cache_k = None
        self.cache_v = None


    def rotate_half(self, x):
        half = x.shape[-1] // 2
        first_half, second_half  = x[..., :half], x[..., half:]
        return torch.cat([-second_half, first_half], dim=-1)


    def apply_rotary_pos(self, q, k, cos, sin):
        q_rot = q * cos + self.rotate_half(q) * sin
        k_rot = k * cos + self.rotate_half(k) * sin
        return q_rot, k_rot

    def update_kv_cache(self, batch_size, start_pos, context_len, keys, values, device):
        # Initialize cache if not exist
        if self.cache_k is None:
            self.cache_k = torch.zeros(
                (batch_size, self.config.context_len, self.num_kv_heads, self.head_dim),
                device=device
            )
            self.cache_v = torch.zeros(
                (batch_size, self.config.context_len, self.num_kv_heads, self.head_dim),
                device=device
            )

        # Update cache
        self.cache_k[:batch_size, start_pos:start_pos + context_len] = keys
        self.cache_v[:batch_size, start_pos:start_pos + context_len] = values

        return (self.cache_k[:batch_size, :start_pos + context_len],
                self.cache_v[:batch_size, :start_pos + context_len])


    def forward(self, x, cos, sin, start_pos):
        c_batch_size, c_context_len, c_dim = x.shape

        if self.use_cache and start_pos > 0:
            # Cache branch handling...
            pass
        else:
            # Non-cache branch (process the entire sequence normally)
            q = self.wq(x)
            k = self.wk(x)
            v = self.wv(x)

            # Reshape with correct dimensions
            q = q.view(c_batch_size, c_context_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(c_batch_size, c_context_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(c_batch_size, c_context_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

            queries, keys = self.apply_rotary_pos(q, k, cos, sin)

            # Ensure keys and values are repeated to match query heads regardless of attention method
            keys = repeat_kv(keys, self.num_rep)  # num_rep = num_heads // num_kv_heads
            values = repeat_kv(v, self.num_rep)

            if self.use_flash:
                # Now queries, keys, and values all have the same number of heads
                output = F.scaled_dot_product_attention(queries, keys, values, is_causal=True)
            else:
                # Calculate attention scores
                attention = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)

                # Apply causal mask
                mask = torch.triu(torch.ones(c_context_len, c_context_len, dtype=torch.bool), diagonal=1)
                attention.masked_fill_(mask.to(attention.device), float('-inf'))

                # Get attention weights and apply to values
                attention_weights = F.softmax(attention, dim=-1)
                output = torch.matmul(attention_weights, values)

            # Reshape output
            output = output.transpose(1, 2).contiguous().view(c_batch_size, c_context_len, c_dim)

            return output


class FeedForward(nn.Module):
    """
    Default Feed Forward Layer.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.ffn_hidden_dims

        self.w1 = nn.Linear(config.num_dims, self.hidden_dim, bias=False)
        self.w2 = nn.Linear(self.hidden_dim, config.num_dims, bias=False)
        self.w3 = nn.Linear(config.num_dims, self.hidden_dim, bias=False)

        self.activation_fn = {
            "silu": F.silu,
            "gelu": F.gelu,
            "relu": F.relu
        }[config.activation_fn]

    def forward(self, x):
        return self.w2(self.activation_fn(self.w1(x)) * self.w3(x)), None


class FFNwMoE(nn.Module):
    """
    Feed Forward with MoE with optional shared experts.
    Returns after forward:
        output: Combined outputs from experts
        aux_loss: Auxiliary loss tensor or routing metadata
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_dim = config.ffn_hidden_dims

        self.moe_active_experts = config.moe_active_experts # top_k
        self.moe_aux_loss_coef = config.moe_aux_loss_coef
        self.moe_eps = config.moe_eps
        self.moe_shared_experts = config.moe_shared_experts
        self.num_experts = config.moe_num_experts

        self.use_lossfreebalance = config.use_lossfreebalance


        self.router = nn.Linear(config.num_dims, self.num_experts, bias=False)
        self.experts = nn.ModuleList()
        for _ in range(self.num_experts):
            self.experts.append(
                nn.ModuleList([
                    nn.Linear(config.num_dims, self.hidden_dim, bias=False),
                    nn.Linear(self.hidden_dim, config.num_dims, bias=False),
                    nn.Linear(config.num_dims, self.hidden_dim, bias=False)
                ]))

        # shared experts (for DeepSeekMoE)
        self.shared_experts = nn.ModuleList()
        for _ in range(self.moe_shared_experts):
            self.shared_experts.append(
                nn.ModuleList([
                    nn.Linear(config.num_dims, self.hidden_dim, bias=False),
                    nn.Linear(self.hidden_dim, config.num_dims, bias=False),
                    nn.Linear(config.num_dims, self.hidden_dim, bias=False)
                ]))

        # Auxiliary-loss-free load balancing strategy for mixture-of-experts from DeepSeek https://arxiv.org/pdf/2408.15664
        if self.use_lossfreebalance:
            self.expert_biases = nn.Parameter(torch.zeros(self.num_experts))

    def forward(self, x: torch.Tensor):
        c_batch_size, c_context_len, c_dim = x.shape
        x_flat = x.view(-1, c_dim)          #c_batch_size * c_context_len, c_dim

        router_out = self.router(x_flat)
        router_probs = F.softmax(router_out, dim=-1)

        _, topk_indices = router_out.topk(self.moe_active_experts, dim=-1)

        aux_loss, topk_probs = self._compute_aux_loss(router_out, router_probs, topk_indices)

        output = self._compute_expert_outputs(x_flat, topk_indices, topk_probs, router_probs)

        return output.view(c_batch_size, c_context_len, c_dim), aux_loss

    def _compute_aux_loss(self, router_out, router_probs, topk_indices):
        """
        Computes the auxiliary loss based on whether loss-free balancing is used or not.
        """
        if not self.use_lossfreebalance:
            topk_probs, _ = router_probs.topk(self.moe_active_experts, dim=-1)
            expert_mask = F.one_hot(topk_indices[:, 0], self.num_experts).float()
            density = expert_mask.mean(dim=0)
            router_prob_mean = router_probs.mean(dim=0)
            aux_loss = self.moe_aux_loss_coef * torch.sum(density * router_prob_mean) * self.num_experts

        else: # if use_lossfreebalance
            router_out = router_out + self.expert_biases
            router_probs = torch.sigmoid(router_out) # from https://arxiv.org/pdf/2408.15664 paper
            topk_probs = router_probs.gather(-1, topk_indices)
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

            # In the case of Auxiliary-loss-free load balancing we pass router_probs, topk_indices as aux_loss for further calculations
            aux_loss = (router_probs, topk_indices)
        return aux_loss, topk_probs

    def _compute_expert_outputs(self, x_flat, topk_indices, topk_probs, router_probs):
        """
        Compute the output of the experts and shared experts if needed
        """
        output = torch.zeros_like(x_flat)

        for i in range(self.moe_active_experts):
            expert_index = topk_indices[:, i]
            expert_probs = topk_probs[:, i]

            for expert_id in range(self.num_experts):
                idx = (expert_id == expert_index).nonzero().squeeze()

                if idx.numel() == 0:
                    continue
                x_for_expert = x_flat[idx]
                w1, w2, w3 = self.experts[expert_id]

                expert_output = w2(F.silu(w1(x_for_expert)) * w3(x_for_expert))
                output[idx] += expert_output * expert_probs[idx].unsqueeze(-1)

        # shared experts(for DeepSeekMoE)
        for shared_expert_id in range(self.moe_shared_experts):
            w1, w2, w3 = self.shared_experts[shared_expert_id]
            expert_output = w2(F.silu(w1(x_flat)) * w3(x_flat))
            output = output + expert_output

        return output


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = GroupedQueryAttention(config)
        if config.use_moe:
            self.ffn = FFNwMoE(config)
        else:
            self.ffn = FeedForward(config)

        if config.norm_type == "rmsnorm":
            self.norm_attention = RMSNorm(config)
            self.norm_ffn = RMSNorm(config)
        else:
            self.norm_attention = nn.LayerNorm(config.num_dims, eps=config.rmsnorm_eps)
            self.norm_ffn = nn.LayerNorm(config.num_dims, eps=config.rmsnorm_eps)

    def forward(self, x, cos, sin, start_pos):
        x = x + self.attention(self.norm_attention(x), cos, sin, start_pos)
        ffn_out, aux_loss = self.ffn(self.norm_ffn(x))
        x = x + ffn_out
        return x, aux_loss


class Transformer(nn.Module, PyTorchModelHubMixin): # extending PyTorchModelHubMixin for save weights as safetensors
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.vocab_size = config.vocab_size
        self.num_dims = config.num_dims
        self.num_heads = config.num_heads
        self.context_len = config.context_len
        self.use_moe = config.use_moe
        self.use_lossfreebalance = config.use_lossfreebalance and self.use_moe

        self.num_layers = config.num_layers
        self.rotary_emb = Rotary(config)

        # Calculation of hidden_dim for FFN/FFNwMoE
        # multiple_of = 4
        # ffn_dim_multiplier = config.ffn_dim_multiplier
        hidden_dim = 4 * config.num_dims
        # hidden_dim = int(2 * config.num_dims / 3)

        # if ffn_dim_multiplier is not None:
        #     hidden_dim = int(ffn_dim_multiplier * hidden_dim)

        # config.ffn_hidden_dims = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.tokens_embedding = nn.Embedding(self.vocab_size, self.num_dims)

        self.blocks = nn.ModuleList()
        for _ in range(self.num_layers):
            self.blocks.append(Block(config))

        self.norm = torch.nn.modules.normalization.RMSNorm(config.num_dims, config.rmsnorm_eps) # you also can use RMSNorm(config)
        self.ll_head = nn.Linear(self.num_dims, self.vocab_size, bias=False)


        self.tokens_embedding.weight = self.ll_head.weight
        # torch.nn.init.normal_(self.ll_head.weight, mean=0.0, std=0.02)
        # torch.nn.init.normal_(self.tokens_embedding.weight, mean=0.0, std=0.02)

        # self.freqs_complex = None # precompute_theta_pos_frequencies(self.num_dims // self.num_heads, self.context_len * 2, device=config.device)




    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None, start_pos: int = 0):
        _, seq_len = x.shape

        x = self.tokens_embedding(x)
        cos, sin = self.rotary_emb(x, seq_dim=1)

        # if self.freqs_complex == None:
        #     self.freqs_complex = precompute_theta_pos_frequencies(self.num_dims // self.num_heads, self.context_len * 2, device=x.device)
        # freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]

        total_aux_loss = 0

        for block in self.blocks:
            x, aux_loss = block(x, cos, sin, start_pos=start_pos)
            if self.use_moe and not self.use_lossfreebalance:
                total_aux_loss += aux_loss

        x = self.norm(x)
        logits = self.ll_head(x)


        if targets is None:
            loss = None
            ce_loss = None
        else:
            c_batch_size, c_context_len, c_dim = logits.shape
            logits = logits.view(c_batch_size*c_context_len, c_dim)
            targets = targets.view(c_batch_size*c_context_len)
            ce_loss = F.cross_entropy(logits, targets)

            if self.use_moe and not self.use_lossfreebalance: loss = ce_loss + total_aux_loss    # in this case, ce_loss its loss w/o aux_loss
            else: # if we want to use Auxiliary-loss-free load balancing we pass router_probs, topk_indices as ce_loss
                # Also, work when moe is not used
                loss = ce_loss
                ce_loss = aux_loss

        return logits, loss, ce_loss

    @torch.no_grad()
    def generate(self, x: torch.Tensor, max_tokens: int, temperature: float = 1.0, top_k: int = 50,
                 use_cache: bool = False):
        """
        Generate text from x up to max_tokens
        """
        for c_tkn_pos in range(max_tokens):
            if use_cache:
                if c_tkn_pos == 0:
                    logits, _, ce_loss = self.forward(x, start_pos=c_tkn_pos)
                else:
                    logits, _, ce_loss = self.forward(x[:, -1:], start_pos=c_tkn_pos)
            else:
                logits, _, ce_loss = self.forward(x)

            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                tkl, idx = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < tkl[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token), dim=1)
        return x


def main():
    # config = ModelConfig(
    #     device = 'cuda' if torch.cuda.is_available() else 'cpu',
    #     vocab_size = 50304,

    #     num_dims = 1024,
    #     num_heads = 16,
    #     num_kv_heads = 4,
    #     num_layers = 16,
    #     ffn_hidden_dims = 1024 * 4,

    #     rmsnorm_eps = 1e-6,
    #     rope_theta = 1e5,

    #     context_len = 1024,

    #     use_cache = False,
    #     use_flash = False,
    #     use_moe = False,

    #     moe_num_experts = 6,
    #     moe_active_experts = 1,
    #     moe_eps = 1e-6,
    #     moe_aux_loss_coef = 0.01,
    #     moe_shared_experts = 0,
    #     use_lossfreebalance = False,

    # )


    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # SEED = 1337

    # torch.manual_seed(SEED)
    # if device == 'cuda':
    #     torch.cuda.manual_seed(SEED)

    # model = Transformer(config)
    # model = model.to(device)
    # model = torch.compile(model)

    # print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    pass


if __name__ == "__main__":
    main()
