# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from model.llama_model import LLaMA
import argparse
import gradio as gr



def sample_demo(alpaca):

    @torch.no_grad()
    def process(prompt):
        prompt_text = "## Instruction:\n{}\n\n## Response:".format(prompt)
        print("Received:\n", prompt_text)
        eval_kwargs = dict(beam=1, sampling=True, sampling_topp=0.95, temperature=0.8, min_len=512)
        prompts = [prompt_text]
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

def demo(alpaca):

    @torch.no_grad()
    def process(prompt, temperature, topp):
        prompt_text = "## Instruction:\n{}\n\n## Response:".format(prompt)
        print("Received:\n", prompt_text)
        eval_kwargs = dict(sampling=True, sampling_topp=topp, temperature=temperature)
        prompts = [prompt_text]
        results = alpaca.sample(prompts, **eval_kwargs)[0]
        print("Generated:\n", results[0])
        return str(results[0])

    with gr.Blocks() as demo:
        gr.Markdown(
            """
            <p align="center" width="100%">
            <img src="https://github.com/dropreg/efficient_alpaca/raw/main/efficient_alpaca_logo.PNG" alt="Efficient-Alpaca" style="width: 40%; min-width: 200px; display: block; margin: auto;">
            </p>
            """)

        with gr.Row():
            with gr.Column():
                model_input = gr.Textbox(lines=15, placeholder='Input something', label='Input')
                with gr.Row():
                    gen = gr.Button("Generate")
                    clr = gr.Button("Clear")

            outputs = gr.Textbox(lines=15, label='Output')
                
        gr.Markdown(
            """
            Generation Parameter
            """)
        with gr.Row():
            with gr.Column():
                temperature = gr.Slider(maximum=1, value=0.8, minimum=0, label='Temperature')
                topp = gr.Slider(maximum=1, value=0.95, minimum=0, label='Top P')
        
        inputs = [model_input, temperature, topp]
        gen.click(fn=process, inputs=inputs, outputs=outputs)
        clr.click(fn=lambda value: gr.update(value=""), inputs=clr, outputs=model_input)
                
        gr.Markdown(
            """
            Our project can be found from [Efficient Alpaca](https://github.com/dropreg/efficient_alpaca)
            """)

    demo.launch(share=True)


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
    
    demo(alpaca)
