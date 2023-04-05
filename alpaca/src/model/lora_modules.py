# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torch import nn
import math


class LoRA(nn.Module):

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
