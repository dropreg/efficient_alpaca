# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple
import os
import math
import logging

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from fairseq import options, utils

from fairscale.nn.model_parallel import initialize as mpu
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from fairscale.nn.model_parallel.mappings import scatter_to_model_parallel_region, gather_from_model_parallel_region
import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from .lora_modules import LoRA

logger = logging.getLogger(__name__)


class LLaMAMegatronTransformer(nn.Module):

    def __init__(self, cfg, tgt_dict, embed_tokens, lora_tuning):
        super().__init__()
        
        self.lora_tuning = lora_tuning

        self.tgt_dict = tgt_dict
        self.embed_dim = cfg.decoder_embed_dim
        self.num_layers = cfg.decoder_layers
        self.num_heads = cfg.decoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.max_target_positions = cfg.max_target_positions

        self.pad = self.tgt_dict.pad()
        self.embed_tokens = embed_tokens

        self.layers = torch.nn.ModuleList()
        self.layers.extend(
            [
                self.build_decoder_layer(cfg, self.lora_tuning)
                for _ in range(self.num_layers)
            ]
        )

        self.layer_norm = RMSNorm(self.embed_dim)
        self.output_projection = ColumnParallelLinear(
            self.embed_dim, len(self.tgt_dict), bias=False, init_method=lambda x: x
        )

        self.freqs_cis = self.precompute_freqs_cis(
            self.embed_dim // self.num_heads, self.max_target_positions * 2
        )
        self._future_mask = torch.empty(0)

    def build_decoder_layer(self, cfg, lora_tuning):
        layer = LLaMATransformerLayer(cfg, lora_tuning)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        return layer

    def precompute_freqs_cis(self, dim: int, end: int, theta: float = 10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device)  # type: ignore
        freqs = torch.outer(t, freqs).float()  # type: ignore
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis

    def output_layer(self, x):
        return self.output_projection(x).float()

    def buffered_future_mask(self, tensor):
        dim = tensor.size(1)
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def forward_inf(
        self,
        prev_output_tokens,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        src_pos: Optional[Tensor] = None,
        tgt_pos: Optional[Tensor] = None,
        trunc_flg: bool = False,
    ):
        
        if incremental_state is not None and trunc_flg:
                prev_output_tokens = prev_output_tokens[:, -1:]
        
        bsz, target_len = prev_output_tokens.size()
        x = self.embed_tokens(prev_output_tokens)
        
        key_padding_mask = prev_output_tokens.eq(self.pad)
        if incremental_state is not None:
            key_padding_mask = torch.cat([incremental_state['padding_mask'], key_padding_mask], dim=-1)

        self.freqs_cis = self.freqs_cis.to(x.device)
        if incremental_state is not None:
            freqs_cis = self.freqs_cis[:key_padding_mask.size(1)]
        else:
            freqs_cis = self.freqs_cis[:target_len]

        if incremental_state is not None:
            tgt_attn_mask = self.buffered_future_mask(x)
            tgt_len = tgt_attn_mask.size(1)
            src_len = key_padding_mask.size(1)
            src_attn_mask = torch.torch.zeros([tgt_len, src_len - tgt_len]).to(tgt_attn_mask)
            self_attn_mask = torch.cat([src_attn_mask, tgt_attn_mask], dim=1)
        else:
            self_attn_mask = self.buffered_future_mask(x)
         
        hidden_state = [x]
        attn_state = None
        for layer_idx, layer in enumerate(self.layers):
            
            if incremental_state is not None:
                context = torch.cat([incremental_state[layer_idx]['key'], x], dim=1)
            else:
                context = x
            
            x, attn = layer(
                x,
                context,
                freqs_cis,
                key_padding_mask,
                self_attn_mask,
                src_pos,
                tgt_pos,
            )

            attn_state = attn
            hidden_state.append(x)
            
        attn_state = attn_state.mean(dim=1)
        x = self.layer_norm(x)
        return x, key_padding_mask, attn_state, hidden_state
         
    def forward(self, prev_output_tokens):
        bsz, target_len = prev_output_tokens.size()
        x = self.embed_tokens(prev_output_tokens)
        
        key_padding_mask = prev_output_tokens.eq(self.pad)
        freqs_cis = self.freqs_cis.to(x.device)[:target_len]
        self_attn_mask = self.buffered_future_mask(x)
        
        hidden_state = [x]
        attn_state = None
        for layer_idx, layer in enumerate(self.layers):
            
            x, attn = layer(
                x,
                x,
                freqs_cis,
                key_padding_mask,
                self_attn_mask,
            )

            attn_state = attn
            hidden_state.append(x)
            
        attn_state = attn_state.mean(dim=1)
        x = self.layer_norm(x)
        return x, key_padding_mask, attn_state, hidden_state

class LLaMATransformerLayer(nn.Module):

    def __init__(self, cfg, lora_tuning):
        super().__init__()

        self.lora_tuning = lora_tuning

        self.embed_dim = cfg.decoder_embed_dim
        self.num_heads = cfg.decoder_attention_heads
        self.ffn_embed_dim = cfg.decoder_ffn_embed_dim

        self.attention = LLaMAAttention(self.num_heads, self.embed_dim, lora_tuning)
        self.feed_forward = LLaMAFeedForward(self.embed_dim, self.ffn_embed_dim)

        self.attention_norm = RMSNorm(self.embed_dim)
        self.ffn_norm = RMSNorm(self.embed_dim)

    def forward(
        self, 
        query: Tensor, 
        key_value: Tensor,
        freqs_cis: Tensor, 
        key_padding_mask: Optional[Tensor],
        self_attn_mask: Optional[Tensor],
        src_pos: Optional[Tensor] = None,
        tgt_pos: Optional[Tensor] = None,
    ):

        x, attn = self.attention(
            self.attention_norm(query),
            self.attention_norm(key_value),
            freqs_cis, 
            key_padding_mask,
            self_attn_mask,
            src_pos,
            tgt_pos,
        )
        x = query + x
        x = x + self.feed_forward(self.ffn_norm(x))

        return x, attn

class RMSNorm(torch.nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
 
class LLaMAAttention(nn.Module):

    def __init__(self, num_heads, embed_dim, lora_tuning):
        super().__init__()

        self.lora_tuning = lora_tuning

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.local_num_heads = self.num_heads // fs_init.get_model_parallel_world_size()

        self.q_proj = ColumnParallelLinear(
            self.embed_dim,
            self.embed_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.k_proj = ColumnParallelLinear(
            self.embed_dim,
            self.embed_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.v_proj = ColumnParallelLinear(
            self.embed_dim,
            self.embed_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.out_proj = RowParallelLinear(
            self.embed_dim,
            self.embed_dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        if self.lora_tuning:
            self.q_lora = LoRA(self.embed_dim, self.embed_dim)
            self.k_lora = LoRA(self.embed_dim, self.embed_dim)
            self.v_lora = LoRA(self.embed_dim, self.embed_dim)

    def apply_rotary_emb(
        self,
        query: Tensor,
        key: Tensor,
        freqs_cis: Tensor,
        src_pos: Tensor,
        tgt_pos: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        def reshape_for_broadcast(freqs_cis: Tensor, x: Tensor):
            ndim = x.ndim
            assert 0 <= 1 < ndim
            assert freqs_cis.shape == (x.shape[1], x.shape[-1])
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
            return freqs_cis.view(*shape)

        q_ = torch.view_as_complex(query.float().reshape(*query.shape[:-1], -1, 2))
        k_ = torch.view_as_complex(key.float().reshape(*key.shape[:-1], -1, 2))

        if src_pos is not None and tgt_pos is not None:
            if freqs_cis.size(0) == q_.size(1):
                freqs_cis = reshape_for_broadcast(freqs_cis, q_)
                q_list = []
                k_list = []
                for idx, attn_p in enumerate(src_pos):
                    q_list.append(q_[idx] * freqs_cis.index_select(dim=1, index=attn_p))
                    k_list.append(k_[idx] * freqs_cis.index_select(dim=1, index=attn_p))
                q_out = torch.view_as_real(torch.cat(q_list, dim=0)).flatten(3)
                k_out = torch.view_as_real(torch.cat(k_list, dim=0)).flatten(3)
            else:
                freqs_cis = reshape_for_broadcast(freqs_cis, k_)
                q_list = []
                k_list = []
                idx = 0
                for q_pos, k_pos in zip(tgt_pos, torch.cat([src_pos, tgt_pos], dim=-1)):
                    q_list.append(q_[idx] * freqs_cis.index_select(dim=1, index=q_pos))
                    k_list.append(k_[idx] * freqs_cis.index_select(dim=1, index=k_pos))
                    idx += 1
                q_out = torch.view_as_real(torch.cat(q_list, dim=0)).flatten(3)
                k_out = torch.view_as_real(torch.cat(k_list, dim=0)).flatten(3)
        else:
            freqs_cis = reshape_for_broadcast(freqs_cis, q_)
            q_out = torch.view_as_real(q_ * freqs_cis).flatten(3)
            k_out = torch.view_as_real(k_ * freqs_cis).flatten(3)
        return q_out.type_as(query), k_out.type_as(key)

    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
        freqs_cis: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        src_pos: Optional[Tensor] = None,
        tgt_pos: Optional[Tensor] = None,
    ):
        
        bsz, tgt_len, embed_dim = query.size()
        bsz, src_len, embed_dim = key_value.size()
        
        q = self.q_proj(query)
        k = self.k_proj(key_value)
        v = self.v_proj(key_value)
        
        if self.lora_tuning:
            
            q = gather_from_model_parallel_region(q) + self.q_lora(query)
            k = gather_from_model_parallel_region(k) + self.k_lora(key_value)
            v = gather_from_model_parallel_region(v) + self.v_lora(key_value)

            q = scatter_to_model_parallel_region(q)
            k = scatter_to_model_parallel_region(k)
            v = scatter_to_model_parallel_region(v)

        q = q.view(bsz, tgt_len, self.local_num_heads, self.head_dim)
        k = k.view(bsz, src_len, self.local_num_heads, self.head_dim)
        v = v.view(bsz, src_len, self.local_num_heads, self.head_dim)

        q, k = self.apply_rotary_emb(q, k, freqs_cis=freqs_cis, src_pos=src_pos, tgt_pos=tgt_pos)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask.unsqueeze(0).unsqueeze(1)
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf")
            )

        attn_softmax_scores = F.softmax(attn_scores.float(), dim=-1).type_as(q)
        output = torch.matmul(attn_softmax_scores, v)
        output = output.transpose(1, 2).contiguous().view(bsz, tgt_len, -1)
        return self.out_proj(output), attn_softmax_scores

class LLaMAFeedForward(nn.Module):

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        multiple_of = 256
        self.hidden_dim = int(2 * self.hidden_dim / 3)
        self.hidden_dim = multiple_of * ((self.hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            self.embed_dim, self.hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            self.hidden_dim, self.embed_dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            self.embed_dim, self.hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
