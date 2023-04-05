import torch
import json
import argparse

def build_default_state():

    state = {}
    
    state['args'] = {}
    state['args']['arch'] = "llama_7b"
    state['args']['task'] = "seq2seq_ft_task"
    state['args']['criterion'] = "ll_loss"
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
            'criterion_name': 'lm_loss',
            'optimizer_name': 'MemoryEfficientFP16Optimizer', 
            'lr_scheduler_state': {'best': None},
            'num_updates': 5000,
        }
    ]
    state['extra_state'] = {}
    print(state)
    return state

def build_llama_state_dict(llama_dir, parallel_size, prefix):
    
    llama_state = None
    for file_idx in range(parallel_size):
        print(file_idx)
        with open((llama_dir + prefix).format(file_idx), "rb") as f:
            sep_state = torch.load(f, map_location=torch.device("cpu"))['model']

        if llama_state is None:
            llama_state = sep_state
            continue

        for k in list(sep_state.keys()):
            
            print("{}: {} -> +{}".format(k, llama_state[k].size(), sep_state[k].size()))
            if "inner_attention" in k:
                print("skip llama state key = {} size = {}".format(k, llama_state[k].size()))
                continue
            elif "norm.weight" in k or "_norm" in k:
                continue
            elif "decoder.embed_tokens.weight" in k:
                llama_state[k] = torch.cat([llama_state[k].half(), sep_state[k].half()], dim=-1)
            elif "decoder.output_projection.weight" in k:
                llama_state[k] = torch.cat([llama_state[k].half(), sep_state[k].half()], dim=0)
            elif "layers" in k:
                if "attention" in k and "out_proj" not in k:
                    # 2048, 4096
                    llama_state[k] = torch.cat([llama_state[k].half(), sep_state[k].half()], dim=0)
                elif "attention" in k and "out_proj" in k:
                    llama_state[k] = torch.cat([llama_state[k].half(), sep_state[k].half()], dim=-1)
                elif "feed_forward.w1" in k:
                    llama_state[k] = torch.cat([llama_state[k].half(), sep_state[k].half()], dim=0)
                elif "feed_forward.w2" in k:
                    llama_state[k] = torch.cat([llama_state[k].half(), sep_state[k].half()], dim=-1)
                elif "feed_forward.w3" in k:
                    llama_state[k] = torch.cat([llama_state[k].half(), sep_state[k].half()], dim=0)
                else:
                    print(sep_state[k].size())
                    print(k)
                    raise NotImplementedError
            else:
                print(k)
                print(sep_state[k].size())
                raise NotImplementedError
    
    llama_state["decoder.embed_tokens.weight"] = llama_state["decoder.embed_tokens.weight"][:32008, :]
    llama_state["decoder.output_projection.weight"] = llama_state["decoder.output_projection.weight"][:32008, :]
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
        default="/opt/data/private/ckpt/alpaca/parallel_zh_new_ft/",
        help="path containing model file",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="checkpoint_1_15000-model_part-{}.pt",
        help="where in model_dir are weights saved",
    )
    parser.add_argument(
        "--parallel-size",
        type=int,
        default=8,
        help="model parallel size to split",
    )

    args = parser.parse_args()
    print("load model from {}{}".format(args.llama_model_dir, args.prefix))
    build_llama_state_dict(args.llama_model_dir, args.parallel_size, args.prefix)


if __name__ == "__main__":
    main()
