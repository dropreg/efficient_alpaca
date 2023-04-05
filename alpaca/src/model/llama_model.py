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
from fairseq.models import (
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import DEFAULT_MIN_PARAMS_TO_WRAP
from fairscale.nn.model_parallel import initialize as mpu
from .hub_interface import LLaMAHubInterface
from .llama_transformer import LLaMATransformer
from .llama_megatron_transformer import LLaMAMegatronTransformer
from fairscale.nn.model_parallel.layers import ParallelEmbedding
from fairseq.utils import safe_getattr, safe_hasattr


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
    checkpoint_activations: bool = field(
        default=False, metadata={"help": "checkpoint activations at each layer"}
    )
    offload_activations: bool = field(
        default=False,
        metadata={"help": "move checkpointed activations to CPU after they are used."},
    )
    min_params_to_wrap: int = field(
        default=DEFAULT_MIN_PARAMS_TO_WRAP,
        metadata={
            "help": ("minimum number of params for a layer to be wrapped with FSDP()")
        },
    )


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

@register_model("llama", dataclass=LLaMAConfig)
class LLaMA(BaseFairseqModel):
    
    def __init__(self, decoder, lora_tuning):
        super().__init__()
        self.decoder = decoder

        self.lora_tuning = lora_tuning
        logger.info('run efficient-tuning method {}'.format(self.lora_tuning))
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
        llama_base_architecture(args)
        
        logger.info("rescale [src] dictionary: {} types and [tgt] dictionary: {} types".format(
            len(task.source_dictionary), len(task.target_dictionary)))
        
        lora_tuning = safe_getattr(task, "lora_tuning", False)
        if safe_getattr(task, "megatron_model", False):
            cls.initialize_model_parallel()
            
            task.source_dictionary.pad_to_multiple_(torch.distributed.get_world_size() * 8)
            task.target_dictionary.pad_to_multiple_(torch.distributed.get_world_size() * 8)

            embed_tokens = cls.build_megatron_embedding(args, task.target_dictionary, args.decoder_embed_dim)
            decoder = LLaMAMegatronTransformer(args, task.target_dictionary, embed_tokens, lora_tuning)
        else:
            embed_tokens = cls.build_embedding(args, task.target_dictionary, args.decoder_embed_dim)
            decoder = LLaMATransformer(args, task.target_dictionary, embed_tokens, lora_tuning)
        
        return cls(decoder, lora_tuning)

    @classmethod
    def initialize_model_parallel(cls):
        logger.info("llama model init process group")

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")

        if not mpu.model_parallel_is_initialized():
            ws = torch.distributed.get_world_size()
            mpu.initialize_model_parallel(ws)
    
    @classmethod
    def build_megatron_embedding(cls, args, dictionary, embed_dim):
        embed_tokens = ParallelEmbedding(len(dictionary), embed_dim, init_method=lambda x: x)
        return embed_tokens

    @classmethod
    def build_embedding(cls, cfg, dictionary, embed_dim):
        emb = Embedding(len(dictionary), embed_dim, dictionary.pad())
        return emb
    
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

    def forward_encoder(self, encoder_inputs):

        src_x, src_padding, src_attn, src_hiddens = self.decoder.forward_inf(
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
        
        tgt_x, tgt_padding, tgt_attn, tgt_hiddens = self.decoder.forward_inf(
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
    
    def forward(self, prev_output_tokens):
        x, x_padding, x_attn, x_hiddens = self.decoder(prev_output_tokens)
        x_out = self.decoder.output_layer(x)
        return x_out

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
                print("load lora model from {}".format(self.lora_model_inf))
                with open(self.lora_model_inf, "rb") as f:
                    lora_state_dict = torch.load(f, map_location=torch.device("cuda"))['model']
                for k in list(lora_state_dict.keys()):
                    state_dict[k] = lora_state_dict[k]
            else:
                print("no lora model!")
        
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
