# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from llama_model import LLaMA
import argparse
import gradio as gr


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        required=True,
        type=str,
        default="alpaca_lora",
        help="path containing model file and src_dict.txt",
    )
    parser.add_argument(
        "--model-file",
        default="checkpoint_best.pt",
        help="where in model_dir are weights saved",
    )
    parser.add_argument(
        "--lora-model-inf",
        default="checkpoint_best.pt",
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
        "user_dir": "alpaca_lora/src", 
        "lora_model_inf": args.lora_model_inf,
        "bpe": args.bpe,
        "sentencepiece_model": args.sentencepiece_model,
        "source_lang": 'src',
        "target_lang": 'tgt',
        "lora_tuning": args.lora_tuning,
    }
    alpaca = LLaMA.from_pretrained(
        model_name_or_path=args.model_dir,
        checkpoint_file=args.model_file,
        **kwargs,
    )
    alpaca = alpaca.eval()
    if torch.cuda.is_available():
        alpaca = alpaca.cuda().half()
    
    @torch.no_grad()
    def process(prompt):
        print("Received:\n", prompt)
        eval_kwargs = dict(beam=1, sampling=True, sampling_topp=0.95, temperature=0.8)
        prompts = [prompt]
        results = alpaca.sample(prompts, **eval_kwargs)[0]
        print("Generated:\n", results[0])
        return str(results[0])

    demo = gr.Interface(
        title = "Efficient Alpaca",
        thumbnail = "https://github.com/dropreg/efficient_alpaca/blob/main/efficient_alpaca_logo.PNG",
        fn = process,
        inputs = gr.Textbox(lines=10, placeholder="Your prompt here..."),
        outputs = "text",
    )

    demo.launch(share=True)
