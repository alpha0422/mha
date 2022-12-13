#!/usr/bin/env bash

import torch
import triton
from triton_mha import attention as triton_attention

from typing import Optional, Tuple

try:
    from flash_attn.flash_attn_interface import flash_attn_func
    HAS_FLASH = True
except BaseException:
    raise BaseException("Please install HazyResearch/flash-attention.") 
    HAS_FLASH = False

# Simplified version of https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py#L231
# Removed Linear layers in it.
class MultiHeadAttention(torch.nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, num_heads, head_dim, dropout=0.0, training=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.training = training
        self.scale = head_dim**-0.5

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = query_states.size()
        head_dim = embed_dim // self.num_heads

        # reshape from (bsz, seq_len, num_heads, head_dim) to (bsz, num_heads, seq_len, head_dim)
        query_states = query_states * self.scale
        key_states = self._shape(key_states, -1, bsz)
        value_states = self._shape(value_states, -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = torch.nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        return attn_output, attn_weights_reshaped

#BATCH, N_HEADS, N_CTX, D_HEAD = 4, 48, 4096, 64
BATCH, N_HEADS, N_CTX, D_HEAD = 16, 8, 1024, 64
# vary seq length for fixed head and batch=4
configs = [triton.testing.Benchmark(
    x_names=['N_CTX'],
    x_vals=[256 * i for i in range(1, 9)],
    line_arg='provider',
    line_vals=['torch', 'flash', 'triton'],
    line_names=['Torch', 'Flash', 'Triton'],
    styles=[('red', '-'), ('blue', '-'), ('black', '-')],
    ylabel='ms',
    plot_name=f'fused-attention-batch{BATCH}-head{N_HEADS}-d{D_HEAD}-{mode}',
    args={'H': N_HEADS, 'BATCH': BATCH, 'D_HEAD': D_HEAD, 'dtype': torch.float16, 'mode': mode}
) for mode in ['fwd', 'bwd']]


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, D_HEAD, mode, provider, dtype=torch.float16, device="cuda"):
    assert mode in ['fwd', 'bwd']
    warmup = 25
    rep = 100
    if provider == "triton":
        q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        sm_scale = 1.3
        fn = lambda: triton_attention(q, k, v, sm_scale)
        if mode == 'bwd':
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, percentiles=None, warmup=warmup, rep=rep)
        return ms
    if provider == "flash":
        lengths = torch.full((BATCH,), fill_value=N_CTX, device=device)
        cu_seqlens = torch.zeros((BATCH + 1,), device=device, dtype=torch.int32)
        cu_seqlens[1:] = lengths.cumsum(0)
        qkv = torch.randn((BATCH * N_CTX, 3, H, D_HEAD), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: flash_attn_func(qkv, cu_seqlens, 0., N_CTX, causal=True)
        if mode == 'bwd':
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, percentiles=None, warmup=warmup, rep=rep)
        return ms
    if provider == "torch":
        q = torch.randn((BATCH, N_CTX, H * D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        k = torch.randn((BATCH, N_CTX, H * D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        v = torch.randn((BATCH, N_CTX, H * D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        mha = MultiHeadAttention(H, D_HEAD)
        fn = lambda: mha(q, k, v)
        if mode == 'bwd':
            o, _ = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, percentiles=None, warmup=warmup, rep=rep)
        return ms

# only works on A100 at the moment
bench_flash_attention.run(save_path='.', print_data=True)
