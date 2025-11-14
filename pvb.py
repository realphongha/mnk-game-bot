import argparse
import yaml
import sys
import logging
from gui.mnk_gui import MnkGUI
from mnk_game.mcts_mnkgame import MonteCarloTreeSearchMnkGame
from mnk_game.alphazero_mnkgame import AlphaZeroMnkGame
from mnk_game.alphazero_net import MODELS
from utils.logger import setup_logger
from utils.checkpoint import load_net_from_path


def main(cfg):
    setup_logger()
    gui = MnkGUI(**cfg["board_game"])
    bot = cfg["bot"]["algorithm"]
    if bot == "alphazero":
        import torch
        m, n, k = cfg["board_game"]["m"], cfg["board_game"]["n"], cfg["board_game"]["k"]
        device = cfg["bot"]["alphazero"]["device"]
        if device == "cuda" and not torch.cuda.is_available():
            logging.warning("CUDA is not available. Using CPU instead.")
            device = "cpu"
        cfg["bot"]["alphazero"]["device"] = device
        gui.player1 = AlphaZeroMnkGame(m, n, k, **cfg["bot"]["alphazero"])
        net = load_net_from_path(cfg, opt.weights, device)
        gui.player1.init_net(net)
    elif bot == "mcts":
        gui.player1 = MonteCarloTreeSearchMnkGame(**cfg["bot"]["mcts"])
    else:
        raise ValueError(f"Unknown bot type: {bot}")
    gui.player1.debug = True
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
            logging.error(exc)
            sys.exit()
    main(cfg)
