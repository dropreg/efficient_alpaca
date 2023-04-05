import torch
import json
import argparse


def build_default_state():

    state = {}
    
    state['args'] = {}
    state['args']['arch'] = "llama_7b"
    state['args']['task'] = "seq2seq_ft_task"
    state['args']['criterion'] = "lm_loss"

    state['args']['decoder_attention_heads'] = 32
    state['args']['decoder_embed_dim'] = 4096
    state['args']['decoder_ffn_embed_dim'] = 16384
    state['args']['decoder_layers'] = 32
    
    temp_parser = argparse.ArgumentParser()
    for key, value in state['args'].items():
        temp_parser.add_argument("--" + key, default=value)    
    args = temp_parser.parse_args([])

    state['args'] = args
    
    state['model'] = {}
    state['optimizer_history'] = [
        {
            'criterion_name': 'lm_loss',
            'optimizer_name': 'AdamOptimizer', 
            'lr_scheduler_state': {'best': None},
            'num_updates': 2000,
        }
    ]
    state['extra_state'] = {}
    print(state)
    return state

def build_llama_state_dict(llama_dir, llama_file):
    # please replace the llama_path with real path
    with open(llama_dir + llama_file, "rb") as f:
        llama_state = torch.load(f, map_location=torch.device("cpu"))
        
    # add pad to token weight and predicion weight
    dict_size, dict_dim = llama_state['tok_embeddings.weight'].size()
    pad = llama_state['tok_embeddings.weight'].new_zeros([1, dict_dim])
    llama_state['tok_embeddings.weight'] = torch.cat([llama_state['tok_embeddings.weight'], pad], dim=0)
    llama_state['output.weight'] = torch.cat([llama_state['output.weight'], pad], dim=0)

    state = build_default_state()
    state['model'] = llama_state
    dump_file = "model_no_pad.pt"
    torch.save(state, llama_dir + dump_file)
    print("dump new model to {}{}".format(llama_dir, dump_file))

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llama-model-dir",
        type=str,
        default="/opt/data/private/data/llama/7B/",
        help="path containing model file",
    )
    parser.add_argument(
        "--llama-model-file",
        type=str,
        default="consolidated.00.pth",
        help="where in model_dir are weights saved",
    )

    args = parser.parse_args()
    print("load model from {}{}".format(args.llama_model_dir, args.llama_model_file))
    build_llama_state_dict(args.llama_model_dir, args.llama_model_file)


if __name__ == "__main__":
    main()
