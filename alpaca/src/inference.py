# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from model.llama_model import LLaMA
import argparse
import logging

logger = logging.getLogger(__name__)


@torch.no_grad()
def generate(alpaca):
    
    # load from txt
    # prompts = [
    #     "Give three tips for staying healthy.",
    #     "What are the three primary colors?",
    #     "Describe the structure of an atom.",
    #     "Describe a time when you had to make a difficult decision.",
    #     "Explain why the following fraction 4/16 is equivalent to 1/4",
    #     "Write a short story in third person narration about a protagonist who has to make an important career decision.",
    # ]
    
    # load from files
    prompts = open("alpaca/scripts/assert/test.src").readlines()

    eval_kwargs = dict(sampling=True, sampling_topp=0.95, temperature=0.8)
    for prompt in prompts:
        print("-----" * 20)
        prompt_text = "## Instruction:\n{}\n\n## Response:".format(prompt)
        print(prompt_text)
        output = alpaca.sample([prompt_text], **eval_kwargs)[0][0]
        print(output)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        required=True,
        type=str,
        default="",
        help="path containing model file",
    )
    parser.add_argument(
        "--model-file",
        default="",
        help="where in model_dir are weights saved",
    )
    parser.add_argument(
        "--lora-model-inf",
        default="",
        help="where in model_dir are weights saved",
    )
    parser.add_argument(
        "--lora-tuning",
        action="store_true",
        default=False,
        help="if true use XSUM_KWARGS else CNN_KWARGS",
    )

    parser.add_argument("--bpe",)
    parser.add_argument("--sentencepiece-model")
    args = parser.parse_args()
    
    kwargs = {
        "user_dir": "alpaca/src",
        "lora_model_inf": args.lora_model_inf,
        "bpe": args.bpe,
        "sentencepiece_model": args.sentencepiece_model,
        "source_lang": 'src',
        "target_lang": 'tgt',
        "lora_tuning": args.lora_tuning,
        "task": "seq2seq_lora_task",
    }
    alpaca = LLaMA.from_pretrained(
        model_name_or_path=args.model_dir,
        checkpoint_file=args.model_file,
        **kwargs,
    )
    alpaca = alpaca.eval()
    if torch.cuda.is_available():
        alpaca = alpaca.half().cuda()
    
    generate(alpaca)


if __name__ == "__main__":
    main()
