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

DICT_MAP = {
    2: 32016,
    8: 32064,
}

def split_parameter(llama_state, parallel_size):

    parallel_state_list = []
    incr_dict_size = DICT_MAP[parallel_size]
    
    dict_size, dict_dim = llama_state['tok_embeddings.weight'].size()
    pad = llama_state['tok_embeddings.weight'].new_zeros([incr_dict_size - dict_size, dict_dim])
    llama_state['tok_embeddings.weight'] = torch.cat([llama_state['tok_embeddings.weight'], pad], dim=0)
    llama_state['output.weight'] = torch.cat([llama_state['output.weight'], pad], dim=0)

    embed_size = dict_dim // parallel_size
    ffn_embed_size = (256 * ((int(2 * dict_dim * 4 / 3) + 256 - 1) // 256)) // parallel_size
    parallel_dict_size = incr_dict_size // parallel_size

    for parallel_idx in range(parallel_size):
        parallel_state = {}
        start_embed_size = parallel_idx * embed_size
        end_embed_size = (parallel_idx + 1) * embed_size
        start_ffn_embed_size = parallel_idx * ffn_embed_size
        end_ffn_embed_size = (parallel_idx + 1) * ffn_embed_size
        start_parallel_dict_size = parallel_idx * parallel_dict_size
        end_parallel_dict_size = (parallel_idx + 1) * parallel_dict_size

        print("embed dim start={} end={}".format(start_embed_size, end_embed_size))
        print("ffn dim start={} end={}".format(start_ffn_embed_size, end_ffn_embed_size))

        for k in list(llama_state.keys()):
            if "inner_attention" in k:
                print("skip llama state key = {} size = {}".format(k, llama_state[k].size()))
                continue
            elif "norm.weight" in k or "_norm" in k:
                parallel_state[k] = llama_state[k].clone()
            elif "tok_embeddings.weight" in k:
                parallel_state[k] = llama_state[k][:, start_embed_size:end_embed_size].clone()
            elif "output.weight" in k:
                parallel_state[k] = llama_state[k][start_parallel_dict_size:end_parallel_dict_size, :].clone()
            elif "layers" in k:
                if "attention" in k and "wo" not in k:
                    # 2048, 4096
                    parallel_state[k] = llama_state[k][start_embed_size:end_embed_size, :].clone()
                elif "attention" in k and "wo" in k:
                    parallel_state[k] = llama_state[k][:, start_embed_size:end_embed_size].clone()
                elif "feed_forward.w1" in k:
                    parallel_state[k] = llama_state[k][start_ffn_embed_size:end_ffn_embed_size, :].clone()
                elif "feed_forward.w2" in k:
                    parallel_state[k] = llama_state[k][:, start_ffn_embed_size:end_ffn_embed_size].clone()
                elif "feed_forward.w3" in k:
                    parallel_state[k] = llama_state[k][start_ffn_embed_size:end_ffn_embed_size, :].clone()
                else:
                    print(llama_state[k].size())
                    print(k)
                    raise NotImplementedError
            else:
                print(state[k].size())
                print(k)
                raise NotImplementedError
            print("split llama state key = {} size = {}".format(k, llama_state[k].size()))
            print("parallel state size = {}".format(parallel_state[k].size()))
        # import pdb; pdb.set_trace()
        parallel_state_list.append(parallel_state)
    return parallel_state_list

def build_llama_state_dict(llama_dir, llama_file, parallel_size):
    # please replace the llama_path with real path
    with open(llama_dir + llama_file, "rb") as f:
        llama_state = torch.load(f, map_location=torch.device("cpu"))
    
    # add pad to token weight and predicion weight
    state = build_default_state()
    for parallel_idx, parallel_state in enumerate(split_parameter(llama_state, parallel_size)):
        state['model'] = parallel_state
        dump_file = "model-model_part-{}.pt".format(parallel_idx)
        torch.save(state, llama_dir + 'megatron_{}/'.format(parallel_size) + dump_file)
        print("dump new model to {}{}".format(llama_dir, dump_file))

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/opt/data/private/data/llama/7B/",
        help="path containing model file",
    )
    parser.add_argument(
        "--model-file",
        type=str,
        default="consolidated.00.pth",
        help="where in model_dir are weights saved",
    )
    parser.add_argument(
        "--parallel-size",
        type=int,
        default=2,
        help="model parallel size to split",
    )

    args = parser.parse_args()
    print("load model from {}{}.".format(args.model_dir, args.model_file))
    print("We will split the llama model into {} fragment.".format(args.parallel_size))
    build_llama_state_dict(args.model_dir, args.model_file, args.parallel_size)

if __name__ == "__main__":
    main()
