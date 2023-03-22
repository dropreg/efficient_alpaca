# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import os
from omegaconf import II
import math
import logging

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from fairseq import options, utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.utils import safe_getattr, safe_hasattr
from fairseq.models import (
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)

from fairscale.nn.model_parallel import initialize as mpu
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from fairscale.nn.model_parallel.mappings import scatter_to_model_parallel_region, gather_from_model_parallel_region
import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear,
)
from hub_interface import LLaMAHubInterface

logger = logging.getLogger(__name__)


@dataclass
class LLaMAConfig(FairseqDataclass):

    dropout: float = field(default=0.1, metadata={"help": "dropout probability"})
    attention_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability for attention weights"}
    )
    decoder_embed_dim: int = field(
        default=512, metadata={"help": "decoder embedding dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=2048, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(default=6, metadata={"help": "num decoder layers"})
    decoder_attention_heads: int = field(
        default=8, metadata={"help": "num decoder attention heads"}
    )
    max_target_positions: Optional[int] = II("task.max_target_positions")


@register_model("llama", dataclass=LLaMAConfig)
class LLaMA(BaseFairseqModel):
    
    def __init__(self, decoder, lora_tuning):
        super().__init__()

        self.decoder = decoder
        self.lora_tuning = lora_tuning
        
        logger.info('model tuning method {}'.format(self.lora_tuning))
        if self.lora_tuning:
            self.mark_only_lora_as_trainable()

        self.lora_model_inf = None

    def set_lora_model_inf(self, lora_model_inf):
        self.lora_model_inf = lora_model_inf

    def mark_only_lora_as_trainable(self) -> None:
        for n, p in self.named_parameters():
            if 'lora' not in n:
                p.requires_grad = False
            else:
                p.requires_grad = True

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        
        cls.initialize_model_parallel()

        task.source_dictionary.pad_to_multiple_(torch.distributed.get_world_size() * 8)
        task.target_dictionary.pad_to_multiple_(torch.distributed.get_world_size() * 8)
        
        logger.info("rescale [src] dictionary: {} types and [tgt] dictionary: {} types".format(
            len(task.source_dictionary), len(task.target_dictionary)))

        embed_tokens = cls.build_embedding(
            args, task.target_dictionary, args.decoder_embed_dim
        )
        decoder = LLaMaTransformer(
            args, task.target_dictionary, embed_tokens, task.lora_tuning
        )
        return cls(decoder, task.lora_tuning) 
    
    @classmethod
    def initialize_model_parallel(cls):
        logger.info("llama model init process group")

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")

        if not mpu.model_parallel_is_initialized():
            ws = torch.distributed.get_world_size()
            mpu.initialize_model_parallel(ws)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file,
        **kwargs
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            **kwargs,
        )
        return LLaMAHubInterface(x["args"], x["task"], x["models"][0])
    
    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        
        embed_tokens = ParallelEmbedding(
            len(dictionary), embed_dim, init_method=lambda x: x
        )
        return embed_tokens

    def forward_encoder(self, encoder_inputs):

        src_x, src_padding, src_attn, src_hiddens = self.decoder(
            prev_output_tokens=encoder_inputs['src_tokens'],
            src_pos=encoder_inputs['src_pos'],
        )

        return {
            "encoder_out": [src_x],
            "encoder_padding_mask": [src_padding],
            "encoder_states": src_hiddens,
            "src_tokens": [encoder_inputs['src_tokens']],
            "src_pos": [encoder_inputs['src_pos']],
            "tgt_pos": [encoder_inputs['tgt_pos']] if encoder_inputs['tgt_pos'] is not None else [],
            "bos_token_pos": [encoder_inputs['bos_token_pos']],
        }

    def forward_decoder(self, prev_output_tokens, encoder_out, incremental_state=None):
        
        if len(incremental_state) == 0:
            incremental_state["padding_mask"] = encoder_out["encoder_padding_mask"][0]
            for layer_idx, layer_hidden_states in enumerate(encoder_out["encoder_states"]):
                
                incremental_state[layer_idx] = {}
                incremental_state[layer_idx]['key'] = layer_hidden_states
            incremental_state['src_pos'] = encoder_out['src_pos'][0]
            incremental_state['tgt_pos'] = encoder_out['bos_token_pos'][0]
        
        tgt_x, tgt_padding, tgt_attn, tgt_hiddens = self.decoder(
            prev_output_tokens=prev_output_tokens,
            incremental_state=incremental_state,
            src_pos=incremental_state['src_pos'],
            tgt_pos=incremental_state['tgt_pos'],
            trunc_flg=True,
        )

        tgt_out = self.decoder.output_layer(tgt_x)

        if len(incremental_state) > 0:
            incremental_state["padding_mask"] = tgt_padding
            for layer_idx, tgt_hid in enumerate(tgt_hiddens):
                
                incremental_state[layer_idx]['key'] = torch.cat(
                    [incremental_state[layer_idx]['key'], tgt_hid], dim=1
                )
            incremental_state['src_pos'] = torch.cat([
                incremental_state['src_pos'], incremental_state['tgt_pos']], dim=-1)
            incremental_state['tgt_pos'] += 1
        return tgt_out, {"attn": [tgt_attn], "inner_states": tgt_hiddens}, incremental_state
    
    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        logits = net_output[0]

        if log_probs:
            return utils.log_softmax(logits, dim=-1)
        else:
            return utils.softmax(logits, dim=-1)
        
    def forward(self, src_tokens, src_lengths, src_pos, tgt_pos, tgt_tokens):
        
        src_x, src_padding, src_attn, src_hiddens = self.decoder(
            prev_output_tokens=src_tokens,
            src_pos=src_pos,
        )
        
        incremental_state = {}
        incremental_state["padding_mask"] = src_padding
        for layer_idx, layer_hidden_states in enumerate(src_hiddens):
            incremental_state[layer_idx] = {}
            incremental_state[layer_idx]['key'] = layer_hidden_states

        tgt_x, tgt_padding, tgt_attn, tgt_hiddens = self.decoder(
            prev_output_tokens=tgt_tokens,
            incremental_state=incremental_state,
            src_pos=src_pos,
            tgt_pos=tgt_pos,
        )
        tgt_out = self.decoder.output_layer(tgt_x)
        return tgt_out

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        new_order: Tensor,
    ):
        for key, value in incremental_state.items():
            if "padding_mask" in str(key):
                incremental_state[key] = value.index_select(0, new_order)
            elif "pos" in str(key):
                incremental_state[key] = value.index_select(0, new_order)
            else:
                incremental_state[key]['key'] = value['key'].index_select(0, new_order)
        return incremental_state

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(0, new_order)]
        
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        
        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(0, new_order)

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_pos"]) == 0:
            src_pos = []
        else:
            src_pos = [(encoder_out["src_pos"][0]).index_select(0, new_order)]

        if len(encoder_out["tgt_pos"]) == 0:
            tgt_pos = []
        else:
            tgt_pos = [(encoder_out["tgt_pos"][0]).index_select(0, new_order)]
        
        if len(encoder_out["bos_token_pos"]) == 0:
            bos_token_pos = []
        else:
            bos_token_pos = [(encoder_out["bos_token_pos"][0]).index_select(0, new_order)] 

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "src_pos": src_pos,  # B x T
            "tgt_pos": tgt_pos,  # B x T
            "bos_token_pos": bos_token_pos,
        }

    def upgrade_state_dict_named(self, state_dict, name):
        
        if self.lora_tuning and self.lora_model_inf is not None:
            if os.path.exists(self.lora_model_inf):
                logger.info("load lora model from {}".format(self.lora_model_inf))
                with open(self.lora_model_inf, "rb") as f:
                    lora_state_dict = torch.load(f, map_location=torch.device("cuda"))['model']
                for k in list(lora_state_dict.keys()):
                    state_dict[k] = lora_state_dict[k]

        if "decoder.embed_tokens.weight" not in state_dict.keys():
            for k in list(state_dict.keys()):
                if "tok_embeddings.weight" in k:
                    state_dict["decoder.embed_tokens.weight"] = state_dict[k]
                    del state_dict[k]
                elif "output.weight" in k:
                    state_dict["decoder.output_projection.weight"] = state_dict[k]
                    del state_dict[k]
                
                elif "layers" in k:

                    if "inner_attention" in k:
                        del state_dict[k]
                        continue

                    if "wq" in k:
                        new_k = 'decoder.' + k.replace("wq", "q_proj")
                    elif "wk" in k:
                        new_k = 'decoder.' + k.replace("wk", "k_proj")
                    elif "wv" in k:
                        new_k = 'decoder.' + k.replace("wv", "v_proj")
                    elif "wo" in k:
                        new_k = 'decoder.' + k.replace("wo", "out_proj")
                    elif "feed_forward" in k:
                        new_k = 'decoder.' + k
                    elif "_norm" in k:
                        new_k = 'decoder.' + k
                    else:
                        continue
                    
                    state_dict[new_k] = state_dict[k]
                    del state_dict[k]

                elif "norm.weight" in k:
                    state_dict["decoder.layer_norm.weight"] = state_dict[k]
                    del state_dict[k]

                else:
                    raise NotImplementedError

        super().upgrade_state_dict_named(state_dict, name)

class LLaMaTransformer(nn.Module):

    def __init__(self, cfg, tgt_dict, embed_tokens, lora_tuning):
        super().__init__()
        
        self.tgt_dict = tgt_dict
        self.embed_dim = cfg.decoder_embed_dim
        self.num_layers = cfg.decoder_layers
        self.num_heads = cfg.decoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.max_target_positions = cfg.max_target_positions

        self.pad = self.tgt_dict.pad()

        self.lora_tuning = lora_tuning
        # init layers
        self.embed_tokens = embed_tokens

        self.layers = torch.nn.ModuleList()
        self.layers.extend(
            [
                LLaMATransformerLayer(cfg, self.lora_tuning)
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

    def forward(
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
         
class LLaMATransformerLayer(nn.Module):

    def __init__(self, cfg, lora_tuning):
        super().__init__()

        self.embed_dim = cfg.decoder_embed_dim
        self.num_heads = cfg.decoder_attention_heads
        self.ffn_embed_dim = cfg.decoder_ffn_embed_dim
        self.lora_tuning = lora_tuning

        self.attention = LLaMAAttention(self.num_heads, self.embed_dim, self.lora_tuning)
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

        x, attn = self.attention.forward(
            self.attention_norm(query),
            self.attention_norm(key_value),
            freqs_cis, 
            key_padding_mask,
            self_attn_mask,
            src_pos,
            tgt_pos,
        )
        x = query + x
        x = x + self.feed_forward.forward(self.ffn_norm(x))

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

class LLaMALoRA(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.lora_alpha = 32
        self.r = 4
        self.scaling = self.lora_alpha / self.r

        self.lora_A = nn.Parameter(torch.zeros((self.r, input_dim)))
        self.lora_B = nn.Parameter(torch.zeros((output_dim, self.r)))
        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

    def upgrade_state_dict_named(self, state_dict, name):
        
        prefix = name + '.lora_A'
        if prefix not in state_dict:
            state_dict[prefix] = self.lora_A

        prefix = name + '.lora_B'
        if prefix not in state_dict:
            state_dict[prefix] = self.lora_B
 
class LLaMAAttention(nn.Module):

    def __init__(self, num_heads, embed_dim, lora_tuning):
        super().__init__()

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.local_num_heads = self.num_heads // fs_init.get_model_parallel_world_size()
        self.lora_tuning = lora_tuning

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
            self.q_lora = LLaMALoRA(self.embed_dim, self.embed_dim)
            self.k_lora = LLaMALoRA(self.embed_dim, self.embed_dim)
            self.v_lora = LLaMALoRA(self.embed_dim, self.embed_dim)

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

def llama_base_architecture(args):

    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.0)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 4096)
    args.decoder_ffn_embed_dim = safe_getattr(args, "decoder_ffn_embed_dim", 4096 * 4)
    args.decoder_layers = safe_getattr(args, "decoder_layers", 32)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 32)
    args.max_target_positions = safe_getattr(args, "max_target_positions", 2048)

@register_model_architecture("llama", "llama_7b")
def llama_7b(args):
    llama_base_architecture(args)
