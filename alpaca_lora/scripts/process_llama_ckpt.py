import torch

def extract_state_dict():
    # download xml or bart ckpt, and recover the state dict
    xlm_path = "xlmr.base/model.pt"
    with open(xlm_path, "rb") as f:
        state = torch.load(f, map_location=torch.device("cpu"))

    # For args
    del state['args'].activation_dropout
    del state['args'].activation_fn
    del state['args'].encoder_attention_heads
    del state['args'].encoder_embed_dim
    del state['args'].encoder_ffn_embed_dim
    del state['args'].encoder_layers
    del state['args'].pooler_activation_fn
    del state['args'].pooler_dropout
    
    state['args'].dropout = 0.1
    state['args'].attention_dropout = 0.1
    state['args'].decoder_embed_dim = 4096
    state['args'].decoder_ffn_embed_dim = 4096 * 4
    state['args'].decoder_layers = 32
    state['args'].decoder_attention_heads = 32
    state['args'].max_target_positions = 512
    state['args'].share_all_embeddings = True

    state['args'].arch = "llama_7b"
    state['args'].criterion = 'llama_loss'
    state['args'].task ='llama_task'
    
    state['extra_state'] = {}

    state['model'] = {}

    llama_path = "llama/7B/consolidated.00.pth"
    with open(llama_path, "rb") as f:
        llama_state = torch.load(f, map_location=torch.device("cpu"))
    
    state['model'] = llama_state
    torch.save(state, "llama/7B/model.pt")

extract_state_dict()