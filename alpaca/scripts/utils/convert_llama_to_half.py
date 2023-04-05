import torch
import json
import argparse

def convert_llama_half(llama_file):
    
    with open(llama_file, "rb") as f:
        llama_state = torch.load(f, map_location=torch.device("cpu"))

    for k in list(llama_state['model'].keys()):
        llama_state['model'][k] = llama_state['model'][k].half()
    
    dump_file = "checkpoint_half.pt"
    torch.save(llama_state, llama_file.replace("checkpoint_best.pt", dump_file))
    print("dump new model to {}".format(dump_file))

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llama-model-file",
        type=str,
        default="/opt/data/private/ckpt/alpaca/fsdp_belle/checkpoint_best.pt",
        help="path containing model file",
    )

    args = parser.parse_args()
    print("convert model {}".format(args.llama_model_file))
    convert_llama_half(args.llama_model_file)


if __name__ == "__main__":
    main()
