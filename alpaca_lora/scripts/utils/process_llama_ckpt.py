import torch
import json
import argparse


def build_default_state():

    state = {}
    
    state['args'] = {}
    state['args']['adam_betas'] = "(0.9, 0.98)"
    state['args']['adam_eps'] = 1e-06
    state['args']['clip_norm'] = 1.0

    state['args']['best_checkpoint_metric'] = "loss"
    state['args']['maximize_best_checkpoint_metric'] = False
    state['args']['memory_efficient_fp16'] = True
    state['args']['find_unused_parameters'] = False
    state['args']['cpu'] = False
    state['args']['device_id'] = 0
    state['args']['ddp_backend'] = "c10d"
    state['args']['distributed_backend'] = "nccl"
    state['args']['distributed_no_spawn'] = False
    state['args']['fp16'] = True
    state['args']['fp16_init_scale'] = 128
    state['args']['optimizer'] = "adam"
    state['args']['optimizer_overrides'] = '{}'
    state['args']['required_batch_size_multiple'] = 1
    state['args']['restore_file'] = "model.pt"
    state['args']['weight_decay'] = 0.01

    state['args']['arch'] = "llama_7b"
    state['args']['task'] = "llama_task"
    state['args']['criterion'] = "llama_loss"

    state['args']['dropout'] = 0.1
    state['args']['attention_dropout'] = 0.1
    state['args']['decoder_attention_heads'] = 32
    state['args']['decoder_embed_dim'] = 4096
    state['args']['decoder_ffn_embed_dim'] = 16384
    state['args']['decoder_layers'] = 32
    
    state['args']['max_target_positions'] = 2048
    state['args']['max_tokens'] = 2048
    
    temp_parser = argparse.ArgumentParser()
    for key, value in state['args'].items():
        temp_parser.add_argument("--" + key, default=value)    
    args = temp_parser.parse_args([])

    state['args'] = args

    state['model'] = {}
    state['optimizer_history'] = [
        {
            'criterion_name': 'llama_loss',
            'optimizer_name': 'MemoryEfficientFP16Optimizer', 
            'lr_scheduler_state': {'best': None},
            'num_updates': 5000,
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
    pad = llama_state['tok_embeddings.weight'].new_zeros([8, dict_dim])
    llama_state['tok_embeddings.weight'] = torch.cat([llama_state['tok_embeddings.weight'], pad], dim=0)
    llama_state['output.weight'] = torch.cat([llama_state['output.weight'], pad], dim=0)

    state = build_default_state()
    state['model'] = llama_state
    dump_file = "model.pt"
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
