import torch
from mnk_game.alphazero_net import MODELS

def load_net_from_path(cfg, ckpt_path, device):
    m, n, k = cfg["board_game"]["m"], cfg["board_game"]["n"], cfg["board_game"]["k"]
    net_type = cfg["bot"]["alphazero"]["net"]["type"]
    net = MODELS[net_type](m, n, k, **cfg["bot"]["alphazero"]["net"][net_type])
    state = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(state)
    net.to(device)
    return net

