import argparse
import yaml
import sys
import time
import os
import traceback
import copy
import random
import hashlib
import shutil
import logging
from collections import Counter

import torch
import numpy as np
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)
mp.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader
from tqdm import tqdm
from mnk_game.alphazero_mnkgame import AlphaZeroMnkGame
from mnk_game.mcts_mnkgame import MonteCarloTreeSearchMnkGame
from mnk_game.random_agent_mnkgame import RandomAgentMnkGame
from mnk_game import get_agent
from mnk_game.alphazero_net import MODELS
from board_state.mnk_board import MnkBoard
from utils.logger import setup_logger


class MnkDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, data, m, n):
        self.cfg = cfg
        self.data = data
        self.draw_reward = cfg["bot"]["alphazero"]["draw_reward"]
        self.m, self.n = m, n
        self.hashes = set()
        # augmentation
        new_data = []
        for board, turn, policy, res in self.data:
            board = board.clone().numpy()
            policy = self.policy_to_2d(policy)
            new_data.append((board, turn, policy, res))
            new_data.append(self.flip_lr(board, turn, policy, res))
            new_data.append(self.flip_ud(board, turn, policy, res))
            new_data.append(
                self.flip_ud(*self.flip_lr(board, turn, policy, res))
            )
        self.data = new_data
        new_data = []
        for board, turn, policy, res in self.data:
            h = self.hash_board(board, turn)
            if h not in self.hashes:
                self.hashes.add(h)
                new_data.append((board, turn, policy, res))
        logging.info("Removed %d duplicates" % (len(self.data) - len(new_data)))
        self.data = new_data
        data_by_res = {0: [], -1: [], 1: []}
        for d in self.data:
            data_by_res[d[3]].append(d)
        count_by_res = {res: len(data) for res, data in data_by_res.items()}
        logging.info(f"Data by results: {count_by_res}")
        max_by_res = max(count_by_res.values())
        logging.info("Duplicating data to balance classes...")
        for res in [-1, 0, 1]:
            if count_by_res[res] == 0:
                logging.warning("No data for result %d" % res)
                continue
            while count_by_res[res] < max_by_res:
                self.data.append(random.choice(data_by_res[res]))
                count_by_res[res] += 1

    def __len__(self):
        return len(self.data)

    def hash_board(self, board, turn):
        board_bytes = board.tobytes()
        number_bytes = turn.to_bytes(8, byteorder='big', signed=True)
        h = hashlib.sha256(board_bytes + number_bytes).hexdigest()
        return h

    def policy_to_2d(self, policy):
        policy_2d = np.zeros((self.m, self.n))
        for i in range(self.m):
            for j in range(self.n):
                policy_2d[i, j] = policy[i * self.n + j]
        return policy_2d

    def policy_to_1d(self, policy):
        policy_1d = np.zeros((self.m * self.n,))
        for i in range(self.m):
            for j in range(self.n):
                policy_1d[i * self.n + j] = policy[i, j]
        return policy_1d

    def flip_lr(self, board, turn, policy, res):
        board = np.flip(board, axis=2)
        policy = np.flip(policy, axis=1)
        return board, turn, policy, res

    def flip_ud(self, board, turn, policy, res):
        board = np.flip(board, axis=1)
        policy = np.flip(policy, axis=0)
        return board, turn, policy, res

    def __getitem__(self, idx):
        board, turn, policy, res = self.data[idx]
        board = torch.tensor(board.copy(), dtype=torch.float)
        policy = self.policy_to_1d(policy)
        policy = torch.tensor(policy, dtype=torch.float)

        if res == 0:
            res = self.draw_reward
        return board, policy, res


def deep_copy_net(net, cfg):
    net_type = cfg["bot"]["alphazero"]["net"]["type"]
    new_net = MODELS[net_type](
        cfg["board_game"]["m"], cfg["board_game"]["n"], cfg["board_game"]["k"],
        **cfg["bot"]["alphazero"]["net"][net_type]
    )
    new_net.load_state_dict(copy.deepcopy(net.state_dict()))
    new_net.to(cfg["bot"]["alphazero"]["device"])
    return new_net


def temperature_decay(total_it, current_it, max_temp, min_temp):
    temp = max_temp - (max_temp - min_temp) * current_it / total_it
    logging.info(f"Current temperature: {temp}")
    return temp


def lr_decay(total_it, current_it, max_lr, min_lr):
    lr = max_lr - (max_lr - min_lr) * current_it / total_it
    logging.info(f"Current lr: {lr}")
    return lr


def self_play(cfg, temperature, bot1_type, bot2_type, num_games,
              workers, net=None, get_data_from_all=False):
    s = time.time()
    with mp.Pool(workers) as pool:
        args = [
            (cfg, temperature, bot1_type, bot2_type, net, get_data_from_all)
            for _ in range(num_games)
        ]
        res = pool.starmap(self_play_worker, args)
        res = [item for subres in res for item in subres]
    logging.info("Time taken: %.2f (s)" % (time.time() - s))
    return res


def self_play_worker(cfg, temperature, bot1_type, bot2_type,
                     net=None, get_data_from_all=False):
    bot1 = get_agent(bot1_type, cfg, net)
    bot2 = get_agent(bot2_type, cfg, net)
    bot1.temperature = temperature
    bot2.temperature = temperature
    bot1.debug = False
    bot2.debug = False
    training_data = []
    m, n, k = cfg["board_game"]["m"], cfg["board_game"]["n"], cfg["board_game"]["k"]
    try:
        board = MnkBoard(m, n, k)
        possible_pos = board.get_possible_pos()
        turn = random.choice((-1, 1))
        moves = []
        res = 0
        game_data = []
        while not res and possible_pos:
            if turn == -1:
                game = bot1
            else:
                game = bot2
            move, policy = game.solve(board, turn, moves)
            b = AlphaZeroMnkGame.bitboard_to_tensor(
                board.get_board(), m, n, "cpu", turn)[0]
            if isinstance(game, AlphaZeroMnkGame) or get_data_from_all:
                game_data.append((b, turn, policy))
            assert move in possible_pos, f"Invalid move: {move}, agent: {type(game)}"
            board.put(turn, move)
            moves.append(move)
            possible_pos = board.get_possible_pos()
            turn = -turn
            res = board.check_endgame()

        for board, turn, policy in game_data:
            training_data.append((board, turn, policy, res * turn))
    except Exception as e:
        traceback.print_exc()
    if net is not None:
        net.to("cpu")
        del net
    torch.cuda.empty_cache()
    return training_data


def train(data, cfg, lr, eps, net=None):
    m, n, k = cfg["board_game"]["m"], cfg["board_game"]["n"], cfg["board_game"]["k"]
    device = cfg["bot"]["alphazero"]["device"]
    dataset = MnkDataset(cfg, data, m, n)
    logging.info(f"Training the network for {eps} epochs and {len(dataset)} data points...")

    loader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=cfg["bot"]["alphazero"]["batch"],
        num_workers=cfg["bot"]["alphazero"]["workers"]
    )

    if net is None:
        net_type = cfg["bot"]["alphazero"]["net"]["type"]
        net = MODELS[net_type](
            m, n, k, **cfg["bot"]["alphazero"]["net"][net_type]
        )
        net.init_weights()
        net.to(device)
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    policy_loss = torch.nn.CrossEntropyLoss()
    value_loss = torch.nn.MSELoss()

    loss = None
    for epoch in tqdm(range(eps)):
        for board, policy, value in loader:
            optimizer.zero_grad()
            board = board.to(device).float()
            policy = policy.to(device).float()
            value = value.to(device).float()
            policy_pred, value_pred = net(board)
            policy_loss_ = policy_loss(policy_pred, policy)
            value_loss_ = value_loss(value_pred.view(-1), value)
            loss = policy_loss_ + value_loss_
            loss.backward()
            optimizer.step()

    logging.info(f"Training loss: {loss.item()}")
    return net, loss.item()


def play_worker(cfg, bot1_type, bot2_type, net1=None, net2=None):
    bot1 = get_agent(bot1_type, cfg, net1)
    bot2 = get_agent(bot2_type, cfg, net2)
    bot1.debug = False
    bot2.debug = False
    m, n, k = cfg["board_game"]["m"], cfg["board_game"]["n"], cfg["board_game"]["k"]
    board = MnkBoard(m, n, k)
    possible_pos = board.get_possible_pos()
    moves = []
    res = 0
    turn = random.choice((-1, 1))
    while not res and possible_pos:
        if turn == -1:
            game = bot1
        else:
            game = bot2

        move = game.predict(board, turn, moves)

        assert move in possible_pos
        board.put(turn, move)
        moves.append(move)
        res = board.check_endgame()
        possible_pos = board.get_possible_pos()
        turn = -turn
    if net1 is not None:
        net1.to("cpu")
        del net1
    if net2 is not None:
        net2.to("cpu")
        del net2
    torch.cuda.empty_cache()
    return res


def play(cfg, bot1_type, bot2_type, num_games,
              workers, net1=None, net2=None):
    s = time.time()
    with mp.Pool(workers) as pool:
        args = [
            (cfg, bot1_type, bot2_type, net1, net2)
            for _ in range(num_games)
        ]
        res = pool.starmap(play_worker, args)
    logging.info("Time taken: %.2f (s)" % (time.time() - s))
    res = np.array(res, dtype=float)
    res = (res + 1) / 2
    return float(np.mean(res))  # win rate of Bot 2


def arena(best_net, new_net, cfg, vs_mcts):
    m, n, k = cfg["board_game"]["m"], cfg["board_game"]["n"], cfg["board_game"]["k"]
    games = cfg["bot"]["alphazero"]["arena_games"]
    workers = cfg["bot"]["alphazero"]["workers"]
    eval = cfg["bot"]["alphazero"]["eval"]
    assert eval in ("vs_last", "vs_mcts")
    if eval == "vs_mcts" and vs_mcts[0] > 50.0:
        logging.info("AlphaZero agent is now stronger than MCTS agent. Switch to vs_last.")
        cfg["bot"]["alphazero"]["eval"] = "vs_last"
        eval = "vs_last"
    logging.info("Arena: ")
    evolved = True
    if best_net is not None:
        logging.info("Versus best iteration's net:")
        winrate = play(cfg, "alphazero", "alphazero", games,
            workers, net1=best_net, net2=new_net) * 100
        logging.info("Winrate against best iteration: %.2f%%" % winrate)
        if eval == "vs_last":
            evolved = winrate > 50.0

    logging.info("Versus pure MCTS:")
    winrate = play(cfg, "mcts", "alphazero", games, workers, net2=new_net) * 100
    if eval == "vs_mcts":
        evolved = winrate > vs_mcts[0]
    if winrate > vs_mcts[0]:
        vs_mcts[0] = winrate
    vs_mcts[1] = winrate
    logging.info("Winrate against pure MCTS: %.2f%%" % winrate)
    return evolved


def main(cfg, opt):
    net = None
    best_net = None
    device = cfg["bot"]["alphazero"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA is not available. Using CPU instead.")
        device = "cpu"
        cfg["bot"]["alphazero"]["device"] = device
    m, n, k = cfg["board_game"]["m"], cfg["board_game"]["n"], cfg["board_game"]["k"]
    date_time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join(cfg["bot"]["alphazero"]["exp_dir"],
                           cfg["board_game"]["name"] + "_" + date_time_str)
    os.makedirs(exp_dir)
    if not opt.no_wandb:
        import wandb
        wandb.init(
            project="mnk-alphazero",
            name=cfg["board_game"]["name"] + "_" + date_time_str,
            config={
                "name": cfg["board_game"]["name"], "m": m, "n": n, "k": k,
            }
        )
        wandb.save(opt.cfg)
    shutil.copy(opt.cfg, exp_dir)
    setup_logger(os.path.join(exp_dir, "train.log"))

    workers = cfg["bot"]["alphazero"]["workers"]
    num_it = cfg["bot"]["alphazero"]["it"]
    temperature = temperature_decay(
        num_it, 0, *cfg["bot"]["alphazero"]["temperature"])
    vs_mcts = [0.0, 0.0]  # best, current

    # bootstrap with pure MCTS
    # disable multiprocessing inside pure MCTS
    # (we already uses multiprocessing for self-play and arena)
    cfg["bot"]["mcts"]["processes"] = 1
    cfg["bot"]["mcts"]["num_simulations"] = 1
    if cfg["bot"]["alphazero"]["mcts_warm_start"]["self_play_games"] > 0:
        logging.info("Bootstrapping the weights by self-playing pure MCTS...")
        training_data = self_play(cfg, temperature, "mcts", "mcts",
            cfg["bot"]["alphazero"]["mcts_warm_start"]["self_play_games"],
            workers, None, True)
        net, loss = train(training_data, cfg,
            cfg["bot"]["alphazero"]["mcts_warm_start"]["lr"],
            cfg["bot"]["alphazero"]["mcts_warm_start"]["eps"],
            None)
        net.to("cpu")  # for multiprocessing
        net.share_memory()
        _ = arena(best_net, net, cfg, vs_mcts)
        best_net = deep_copy_net(net, cfg)
        torch.save(net.state_dict(), os.path.join(exp_dir, f"it0.pth"))
        torch.save(net.state_dict(), os.path.join(exp_dir, f"best.pth"))
        if not opt.no_wandb:
            wandb.log({
                "winrate_vs_mcts": vs_mcts[1],
                "learning_rate": cfg["bot"]["alphazero"]["mcts_warm_start"]["lr"],
                "temperature": temperature,
                "train_loss": loss
            })

    # main training loop
    for it in range(num_it):
        logging.info("\n=================")
        logging.info(f"Iteration {it+1}:")
        temperature = temperature_decay(
            num_it, it+1, *cfg["bot"]["alphazero"]["temperature"])
        lr = lr_decay(
            num_it, it+1,
            cfg["bot"]["alphazero"]["lr_scheduler"]["init_lr"],
            cfg["bot"]["alphazero"]["lr_scheduler"]["min_lr"]
        )
        # AlphaZero vs AlphaZero
        games = cfg["bot"]["alphazero"]["self_play_games"]
        logging.info(f"Self-playing AlphaZero vs AlphaZero for {games} games...")
        if net is not None:
            net.to("cpu")  # for multiprocessing
            net.share_memory()
        training_data = self_play(cfg, temperature, "alphazero", "alphazero",
            games, workers, net, False)
        # AlphaZero vs MCTS
        games = cfg["bot"]["alphazero"]["self_play_games_vs_mcts"]
        logging.info(f"Self-playing AlphaZero vs MCTS for {games} games...")
        training_data.extend(
            self_play(cfg, temperature, "alphazero", "mcts",
                games, workers, net, False)
        )
        # AlphaZero vs Random
        games = cfg["bot"]["alphazero"]["self_play_games_vs_random"]
        logging.info(f"Self-playing AlphaZero vs Random agent for {games} games...")
        training_data.extend(
            self_play(cfg, temperature, "alphazero", "random",
                games, workers, net, False)
        )

        net, loss = train(training_data, cfg,
                    lr, cfg["bot"]["alphazero"]["eps"],
            deep_copy_net(best_net, cfg) if best_net is not None else None)
        net.to("cpu")  # for multiprocessing
        net.share_memory()
        if best_net is not None:
            best_net.to("cpu")  # for multiprocessing
            best_net.share_memory()
        evolved = arena(best_net, net, cfg, vs_mcts)

        # save net
        torch.save(net.state_dict(), os.path.join(exp_dir, f"it{it+1}.pth"))
        logging.info(f"Saved weights to it{it+1}.pth")
        if evolved or best_net is None:
            best_net = deep_copy_net(net, cfg)
            torch.save(net.state_dict(), os.path.join(exp_dir, f"best.pth"))
            logging.info(f"Saved weights to best.pth")
        if not opt.no_wandb:
            wandb.log({
                "winrate_vs_mcts": vs_mcts[1],
                "learning_rate": lr,
                "temperature": temperature,
                "train_loss": loss
            })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/tic_tac_toe.yaml',
                        help='path to config file')
    parser.add_argument('--no-wandb', default=False, action='store_true',
                        help='disable wandb')
    opt = parser.parse_args()
    with open(opt.cfg, "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.error(exc)
            sys.exit()
    main(cfg, opt)

