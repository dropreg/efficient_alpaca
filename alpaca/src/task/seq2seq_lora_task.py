import torch
import itertools
import os
import logging
from typing import Dict, Optional

from dataclasses import dataclass, field
from fairseq.tasks import FairseqTask, register_task
from fairseq.tasks.translation import TranslationConfig
from fairseq.data import iterators
from .seq2seq_ft_task import Seq2SeqFineTuningTask, FTTaskConfig
from fairseq.utils import safe_getattr, safe_hasattr


logger = logging.getLogger(__name__)


@dataclass
class LoRATaskConfig(FTTaskConfig):
    
    lora_model_inf: Optional[str] = field(
        default="", metadata={"help": "load lora model for inference"},
    )

    lora_tuning: bool = field(
        default=False, metadata={"help": "if using lora tuning"},
    )


@register_task("seq2seq_lora_task", dataclass=LoRATaskConfig)
class Seq2SeqLoRATask(Seq2SeqFineTuningTask):

    def __init__(self, cfg, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

        self.lora_model_inf = safe_getattr(cfg, "lora_model_inf", "")
        self.lora_tuning = safe_getattr(cfg, "lora_tuning", False)

    def build_model(self, cfg, from_checkpoint=False):
        model = super().build_model(cfg, from_checkpoint)
        if len(self.lora_model_inf) > 0:
            model.set_lora_model_inf(self.lora_model_inf)
            logging.info("Seq2SeqLoRATask load inference model checkpoint from {}".format(self.lora_model_inf))
        return model

