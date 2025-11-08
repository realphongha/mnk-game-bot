import time
import math
import logging

import torch
import torch.nn.functional as F
import numpy as np

from .mnk_bot_base import MnkGameBotBase
from .alphazero_net import MODELS
from board_state.board import to_board
from board_state.mnk_state import MnkState


class AlphaZeroMnkGame(MnkGameBotBase):
    def __init__(self, m, n, k, max_thinking_time, batch, exploration_const,
                 dirichlet_alpha, dirichlet_eps, exp_dir, device, debug, net,
                 **kwargs):
        self.max_thinking_time = max_thinking_time
        self.batch = batch
        self.c = exploration_const
        self.m, self.n, self.k = m, n, k
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.net_cfg = net
        self.device = device
        self.debug = debug

    def init_net(self, net=None):
        if net is not None:
            self.net = net
            self.net.to(self.device)
        else:
            net_type = self.net_cfg["type"]
            self.net = MODELS[net_type](
                self.m, self.n, self.k,
                **self.net_cfg[net_type]
            )
            self.net.init_weights()
            self.net.to(self.device)
        self.net.eval()
        return self.net

    def update_tree(self, two_last_moves):
        try:
            if self.debug:
                logging.info("Inheriting previous tree root...")
            m1, m2 = two_last_moves
            self.root = self.root.children[m1].children[m2]
            self.root.parent = None
            return True
        except KeyError:
            if self.debug:
                logging.info("Moves not found in previous tree. Initializing new tree...")
            return False

    @staticmethod
    def score(node, c):
        # puct
        v = node.r / node.n if node.n > 0 else 0.0
        e = c * node.prior * math.sqrt(node.parent.n) / (1 + node.n)
        return v + e

    @staticmethod
    def bitboard_to_tensor(bb, m, n, device, turn):
        board = to_board(bb, m, n)
        # put current player in channel 0
        if turn == -1:
            board = np.flip(board, axis=0)
        tensor = torch.tensor(board.copy()).float().to(device)
        return tensor.unsqueeze(0)

    @torch.no_grad()
    def solve(self, board, turn, moves):
        self.net.eval()
        start = time.time()
        if len(moves) < 2 or self.root is None:
            if self.debug:
                logging.info("Initializing new tree...")
            self.root = MnkState(board, turn, "blah", None, None)
        else:
            if not self.update_tree(moves[-2:]):
                self.root = MnkState(board, turn, "blah", None, None)
        policy, value = self.net(
            self.bitboard_to_tensor(self.root.board.get_board(), self.m, self.n,
                                    self.device, self.root.turn)
        )
        policy = F.softmax(policy, dim=-1)
        policy = policy[0]
        value = value[0]
        dirichlet_noise = np.random.dirichlet(
            [self.dirichlet_alpha] * self.m * self.n
        )
        dirichlet_noise = torch.tensor(dirichlet_noise).to(self.device)
        policy = (1 - self.dirichlet_eps) * policy + self.dirichlet_eps * dirichlet_noise
        moves = set()
        states = self.root.get_next_states()
        for state in states:
            i, j = state.last_move
            moves.add((i, j))
        for i in range(self.m):
            for j in range(self.n):
                if (i, j) not in moves:
                    policy[i*self.n + j] = 0
        if policy.sum() > 0:
            policy = policy / policy.sum()
        for state in states:
            i, j = state.last_move
            state.prior = policy[i*self.n + j].item()
        while time.time()-start < self.max_thinking_time:
            self.loop()
        return self.get_results()

    def get_results(self):
        policy = np.zeros((self.m * self.n,), dtype=np.float32)
        policy_ = []
        nodes = list(self.root.children.values())
        for node in nodes:
            i, j = node.last_move
            policy[i*self.n + j] = node.n
            policy_.append(node.n)
        assert policy.sum() > 0, "No backpropagation found. Please increase max_thinking_time."
        policy /= policy.sum()
        i = None
        j = None
        if self.temperature <= 0.05:
            move = np.argmax(policy)
            i, j = move // self.n, move % self.n
        else:
            policy_ = np.power(policy_, 1.0/self.temperature)
            assert policy_.sum() > 0, "No backpropagation found. Please increase max_thinking_time."
            policy_ /= policy_.sum()
            move_i = np.random.choice([i for i in range(len(policy_))], p=policy_)
            i, j = nodes[move_i].last_move
        if self.debug:
            children = []
            for child in nodes:
                children.append((child, self.score(child, 0)))
            children.sort(key=lambda child: -child[1])
            k = 5
            children = children[:k] if len(children) >= k else children
            logging.info(f"\nTop {k} moves:")
            for child, score in children:
                logging.info(f"Move: {child.last_move} - score: %.4f - w: %i - n: %i" %
                    (score, child.r, child.n)
                )
        return (i, j), policy

    @torch.no_grad()
    def predict(self, board, turn, _moves):
        self.net.eval()
        b = self.bitboard_to_tensor(
            board.get_board(), self.m, self.n, self.device, turn)
        policy, value = self.net(b)
        policy = F.softmax(policy, dim=-1)
        policy = policy[0]
        self.last_predict_value = value[0]
        possible_move = set(board.get_possible_pos())
        best_move = None
        max_prob = -1.0
        if self.debug:
            moves = []
        for i in range(self.m):
            for j in range(self.n):
                if (i, j) not in possible_move:
                    continue
                move = i * self.n + j
                if policy[move] > max_prob:
                    max_prob = policy[move]
                    best_move = (i, j)
                if self.debug:
                    moves.append((i, j, policy[move]))
        if self.debug:
            moves.sort(key=lambda move: -move[2])
            k = 5
            moves = moves[:k] if len(moves) >= k else moves
            logging.info(f"\nTop {k} moves:")
            for i, j, p in moves:
                logging.info(f"Move: ({i}, {j}), prob: %.2f" % p)
            wr = (self.last_predict_value + 1) * 50
            logging.info("Winrate = %.2f%%" % wr)

        return best_move

    def selection(self):
        node = self.root
        while not node.is_leaf():
            selected_node = max(
                node.children.values(), key=lambda child: self.score(child, self.c))
            node = selected_node
        return node

    def expansion(self, node):
        res = node.board.check_endgame()
        if res:
            return -1.0  # node.turn player lost
        states = node.get_next_states()
        if not states:
            return 0.0
        policy, value = self.net(
            self.bitboard_to_tensor(node.board.get_board(), self.m, self.n,
                                    self.device, node.turn)
        )
        policy = F.softmax(policy, dim=-1)
        policy = policy[0]
        value = value[0]
        moves = {state.last_move for state in states}
        for i in range(self.m):
            for j in range(self.n):
                if (i, j) not in moves:
                    policy[i*self.n + j] = 0.0
        if policy.sum() > 0:
            policy = policy / policy.sum()
        for state in states:
            i, j = state.last_move
            state.prior = policy[i*self.n + j].item()
        return value.item()

    def backpropagation(self, node, value):
        reward = -value
        while node is not None:
            node.n += 1
            node.r += reward
            reward = -reward
            node = node.parent

        current_node = node
        while current_node is not None:
            current_node.n += 1
            current_node.r += reward
            reward = -reward

            # Fix the memleak: Call the weak reference to get the parent
            if current_node.parent is None:
                current_node = None # At the root
            else:
                current_node = current_node.parent() # Call the weakref

    def loop(self):
        node = self.selection()
        value = self.expansion(node)
        self.backpropagation(node, value)

