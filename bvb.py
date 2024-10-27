import argparse
import yaml
import sys
from gui.mnk_gui import MnkGUI
from mnk_game.mcts_mnkgame import MonteCarloTreeSearchMnkGame
from mnk_game.alphazero_mnkgame import AlphaZeroMnkGame
from mnk_game.alphazero_net import MODELS


def main(cfg):
    gui = MnkGUI(**cfg["board_game"])
    # gui.player1 = MonteCarloTreeSearchMnkGame(**cfg["bot"]["mcts"])
    import torch
    m, n, k = cfg["board_game"]["m"], cfg["board_game"]["n"], cfg["board_game"]["k"]
    device = cfg["bot"]["alphazero"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Using CPU instead.")
        device = "cpu"
    cfg["bot"]["alphazero"]["device"] = device
    gui.player1 = AlphaZeroMnkGame(m, n, k, **cfg["bot"]["alphazero"])
    net_type = cfg["bot"]["alphazero"]["net"]["type"]
    net = MODELS[net_type](
        cfg["board_game"]["m"], cfg["board_game"]["n"], cfg["board_game"]["k"],
        **cfg["bot"]["alphazero"]["net"][net_type]
    )
    net.load_state_dict(torch.load(opt.weights, map_location=device))
    gui.player1.init_net(net)
    gui.player2 = MonteCarloTreeSearchMnkGame(**cfg["bot"]["mcts"])
    gui.player1.debug = True
    gui.player2.debug = True
    gui.main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',
                        type=str,
                        default='configs/tic_tac_toe.yaml',
                        help='path to config file')
    parser.add_argument('--weights',
                        type=str,
                        default=None,
                        help='alphazero weights path')
    opt = parser.parse_args()
    with open(opt.cfg, "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit()
    main(cfg)
