# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.data import encoders
from fairseq.hub_utils import GeneratorHubInterface
from omegaconf import open_dict


logger = logging.getLogger(__name__)


class LLaMAHubInterface(GeneratorHubInterface):

    def __init__(self, cfg, task, model):
        super().__init__(cfg, task, [model])
        self.model = self.models[0]

    def encode(
        self, sentence: str, *addl_sentences, no_separator=True
    ) -> torch.LongTensor:
        bpe_sentence = "<s> " + self.bpe.encode(sentence)
        tokens = self.task.target_dictionary.encode_line(bpe_sentence, append_eos=False)
        return tokens.long()

    def decode(self, tokens: torch.LongTensor):
        tokens = tokens.cpu().numpy()
        sentences = [self.bpe.sp.decode(tokens.tolist())]
        return sentences
        
    def sample(
        self, sentences: List[str], **kwargs
    ) -> List[str]:
        tokenized_sentences = [self.encode(sentence) for sentence in sentences]
        batched_hypos = self.generate(tokenized_sentences, **kwargs)
        return [self.decode(hypos[0]["tokens"]) for hypos in batched_hypos]

    def generate(
        self,
        tokenized_sentences: List[torch.LongTensor],
        **kwargs
    ) -> List[List[Dict[str, torch.Tensor]]]:

        generator = self.task.build_generator(
            self.models,
            kwargs,
        )

        results = []
        for batch in self._build_batches(tokenized_sentences, skip_invalid_size_inputs=False):
            batch = utils.apply_to_sample(lambda t: t.to(self.device), batch)
            translations = self.task.inference_step(
                generator, self.models, batch,
            )
            for id, hypos in zip(batch["id"].tolist(), translations):
                results.append((id, hypos))

        # sort output to match input order
        outputs = [hypos for _, hypos in sorted(results, key=lambda x: x[0])]
        return outputs
