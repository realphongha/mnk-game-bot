import argparse
import yaml
import sys
import time
import os
import traceback
import copy
import random
import hashlib

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from mnk_game.alphazero_mnkgame import AlphaZeroMnkGame
from mnk_game.mcts_mnkgame import MonteCarloTreeSearchMnkGame
from mnk_game.random_agent_mnkgame import RandomAgentMnkGame
from mnk_game.alphazero_net import MODELS
from board_state.mnk_board import MnkBoard


def temperature_decay(total_it, current_it, max_temp, min_temp):
    temp = max_temp - (max_temp - min_temp) * current_it / total_it
    print(f"Current temperature: {temp}")
    return temp


def self_play(cfg, temperature, bot1, bot2, num_games, get_data_from_all=False):
    bot1.temperature = temperature
    bot2.temperature = temperature
    bot1.debug = False
    bot2.debug = False
    print("Self-playing %d games..." % num_games)
    training_data = []
    m, n, k = cfg["board_game"]["m"], cfg["board_game"]["n"], cfg["board_game"]["k"]
    for _ in tqdm(range(num_games)):
        # clear last trees
        bot1.root = None
        bot2.root = None
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
                bb = AlphaZeroMnkGame.bitboard_to_tensor(board.get_board(), m, n, "cpu")[0]
                if isinstance(game, AlphaZeroMnkGame) or get_data_from_all:
                    game_data.append((bb, turn, policy))
                assert move in possible_pos, f"Invalid move: {move}, agent: {type(game)}"
                board.put(turn, move)
                moves.append(move)
                possible_pos = board.get_possible_pos()
                turn = -turn
                res = board.check_endgame()

            for board, turn, policy in game_data:
                training_data.append((board, turn, policy, res))
        except Exception as e:
            traceback.print_exc()
    return training_data


class MnkDataset(torch.utils.data.Dataset):
    def __init__(self, data, m, n):
        self.data = data
        self.m, self.n = m, n
        self.hashes = set()
        new_data = []
        for board, turn, policy, res in self.data:
            np_board = board.clone().numpy()
            h = self.hash_board(np_board, turn)
            if h not in self.hashes:
                self.hashes.add(h)
                new_data.append((board, turn, policy, res))
        print("Removed %d duplicates" % (len(self.data) - len(new_data)))
        self.data = new_data
        # augmentation
        new_data = []
        for board, turn, policy, res in self.data:
            board = board.clone().numpy()
            policy = self.policy_to_2d(policy)
            new_data.append((board, turn, policy, res))
            new_data.append(self.flip_lr(board, turn, policy, res))
            new_data.append(self.flip_ud(board, turn, policy, res))
            new_data.append(self.flip_turn(board, turn, policy, res))
            new_data.append(
                self.flip_ud(*self.flip_lr(board, turn, policy, res))
            )
            new_data.append(
                self.flip_turn(*self.flip_lr(board, turn, policy, res))
            )
            new_data.append(
                self.flip_turn(*self.flip_ud(board, turn, policy, res))
            )
            new_data.append(
                self.flip_lr(*self.flip_turn(*self.flip_ud(board, turn, policy, res))),
            )
        self.data = new_data

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

    def flip_turn(self, board, turn, policy, res):
        board = np.flip(board, axis=0)
        return board, turn, np.copy(policy), -res

    def __getitem__(self, idx):
        board, turn, policy, res = self.data[idx]
        board = torch.tensor(board.copy(), dtype=torch.float)
        policy = self.policy_to_1d(policy)
        policy = torch.tensor(policy, dtype=torch.float)

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


def train(data, cfg, lr, eps, net=None):
    m, n, k = cfg["board_game"]["m"], cfg["board_game"]["n"], cfg["board_game"]["k"]
    device = cfg["bot"]["alphazero"]["device"]
    dataset = MnkDataset(data, m, n)
    print(f"Training the network for {eps} epochs and {len(dataset)} data points...")

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

    return net


def play(player1, player2, cfg):
    m, n, k = cfg["board_game"]["m"], cfg["board_game"]["n"], cfg["board_game"]["k"]
    board = MnkBoard(m, n, k)
    possible_pos = board.get_possible_pos()
    moves = []
    res = 0
    while possible_pos:
        move = player1.predict(board, 1, moves)

        assert move in possible_pos
        board.put(1, move)
        moves.append(move)
        res = board.check_endgame()
        if res:
            break
        possible_pos = board.get_possible_pos()
        if not possible_pos:
            break

        move = player2.predict(board, -1, moves)
        assert move in possible_pos
        board.put(-1, move)
        moves.append(move)
        res = board.check_endgame()
        if res:
            break
        possible_pos = board.get_possible_pos()

    return res


def arena(best_net, new_net, cfg, best_vs_mcts):
    m, n, k = cfg["board_game"]["m"], cfg["board_game"]["n"], cfg["board_game"]["k"]
    games = cfg["bot"]["alphazero"]["arena_games"]
    new_game = AlphaZeroMnkGame(m, n, k, **cfg["bot"]["alphazero"])
    new_game.init_net(new_net)
    eval = cfg["bot"]["alphazero"]["eval"]
    assert eval in ("vs_last", "vs_mcts")
    print("Arena: ")
    evolved = True
    if best_net is not None:
        print("Versus best iteration's net:")
        best_game = AlphaZeroMnkGame(m, n, k, **cfg["bot"]["alphazero"])
        best_game.init_net(best_net)
        total = []
        for _ in tqdm(range(games//2)):
            res = play(new_game, best_game, cfg)
            if res == 1: total.append(1)
            elif res == -1: total.append(0)
            else: total.append(0.5)
            res = play(best_game, new_game, cfg)
            if res == -1: total.append(1)
            elif res == 1: total.append(0)
            else: total.append(0.5)
        percent = sum(total) / len(total) * 100 if total else 50.0
        print("Winrate against best iteration: %.2f%%" % percent)
        if eval == "vs_last":
            evolved = percent > 50.0

    print("Versus pure MCTS:")
    mcts = MonteCarloTreeSearchMnkGame(**cfg["bot"]["mcts"])
    mcts.debug = False
    total = []
    for _ in tqdm(range(games//2)):
        res = play(new_game, mcts, cfg)
        if res == 1: total.append(1)
        elif res == -1: total.append(0)
        else: total.append(0.5)
        res = play(mcts, new_game, cfg)
        if res == -1: total.append(1)
        elif res == 1: total.append(0)
        else: total.append(0.5)
    percent = sum(total) / len(total) * 100
    if eval == "vs_mcts":
        evolved = percent > best_vs_mcts[0]
        if evolved:
            best_vs_mcts[0] = percent
    print("Winrate against pure MCTS: %.2f%%" % percent)
    return evolved


def main(cfg):
    net = None
    best_net = None
    device = cfg["bot"]["alphazero"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Using CPU instead.")
        device = "cpu"
        cfg["bot"]["alphazero"]["device"] = device
    m, n, k = cfg["board_game"]["m"], cfg["board_game"]["n"], cfg["board_game"]["k"]
    date_time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join(cfg["bot"]["alphazero"]["exp_dir"], date_time_str)
    os.makedirs(exp_dir)

    temperature = temperature_decay(
        cfg["bot"]["alphazero"]["it"], 0, *cfg["bot"]["alphazero"]["temperature"])
    # bootstrap with pure MCTS
    if cfg["bot"]["alphazero"]["mcts_warm_start"]["self_play_games"] > 0:
        print("Bootstrapping the weights by self-playing pure MCTS...")
        p1 = MonteCarloTreeSearchMnkGame(**cfg["bot"]["mcts"])
        p2 = MonteCarloTreeSearchMnkGame(**cfg["bot"]["mcts"])
        training_data = self_play(cfg, temperature, p1, p2,
            cfg["bot"]["alphazero"]["mcts_warm_start"]["self_play_games"], True)
        net = train(training_data, cfg,
            cfg["bot"]["alphazero"]["mcts_warm_start"]["lr"],
            cfg["bot"]["alphazero"]["mcts_warm_start"]["eps"],
            None)
        _ = arena(best_net, net, cfg, [0.0,])
        best_net = deep_copy_net(net, cfg)
        torch.save(net.state_dict(), os.path.join(exp_dir, f"it0.pth"))
        torch.save(net.state_dict(), os.path.join(exp_dir, f"best.pth"))

    # main training loop
    best_vs_mcts = [0.0,]
    for it in range(cfg["bot"]["alphazero"]["it"]):
        print("\n=================")
        print(f"Iteration {it+1}:")
        temperature = temperature_decay(
            cfg["bot"]["alphazero"]["it"], it+1,
            *cfg["bot"]["alphazero"]["temperature"])
        # agents for self-play
        alphazero_agent1 = AlphaZeroMnkGame(m, n, k, **cfg["bot"]["alphazero"])
        alphazero_agent1.init_net(net)
        alphazero_agent2 = AlphaZeroMnkGame(m, n, k, **cfg["bot"]["alphazero"])
        alphazero_agent2.init_net(net)
        mcts_agent = MonteCarloTreeSearchMnkGame(**cfg["bot"]["mcts"])
        random_agent = RandomAgentMnkGame()
        # AlphaZero vs AlphaZero
        print("Self-playing AlphaZero vs AlphaZero...")
        training_data = self_play(cfg, temperature,
                                  alphazero_agent1, alphazero_agent2,
            cfg["bot"]["alphazero"]["self_play_games"], False)
        # AlphaZero vs MCTS
        print("Self-playing AlphaZero vs MCTS...")
        training_data.extend(self_play(cfg, temperature, alphazero_agent1, mcts_agent,
            cfg["bot"]["alphazero"]["self_play_games_vs_mcts"], False))
        # AlphaZero vs Random
        print("Self-playing AlphaZero vs Random agent...")
        training_data.extend(self_play(cfg, temperature, alphazero_agent1, random_agent,
            cfg["bot"]["alphazero"]["self_play_games_vs_random"], False))

        net = train(training_data, cfg,
            cfg["bot"]["alphazero"]["lr"],
            cfg["bot"]["alphazero"]["eps"],
            deep_copy_net(best_net, cfg) if best_net is not None else None)
        evolved = arena(best_net, net, cfg, best_vs_mcts)

        # save net
        torch.save(net.state_dict(), os.path.join(exp_dir, f"it{it+1}.pth"))
        if evolved or best_net is None:
            best_net = deep_copy_net(net, cfg)
            torch.save(net.state_dict(), os.path.join(exp_dir, f"best.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/tic_tac_toe.yaml',
                        help='path to config file')
    opt = parser.parse_args()
    with open(opt.cfg, "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit()
    main(cfg)

