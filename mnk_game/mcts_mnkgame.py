import time
import random
import multiprocessing
from typing import Tuple

import numpy as np

from mcts.mcts import MonteCarloTreeSearchMixin
from .mnk_bot_base import MnkGameBotBase
from board_state.mnk_state import MnkState, rollout
from board_state.mnk_board import MnkBoard


class MonteCarloTreeSearchMnkGame(MonteCarloTreeSearchMixin, MnkGameBotBase):
    def __init__(self, max_thinking_time, max_rollout, processes, policy,
                 exploration_const, num_simulations) -> None:
        super().__init__(max_thinking_time)
        self.max_rollout = max_rollout
        self.processes = processes
        self.pool = multiprocessing.Pool(self.processes)
        self.policy = policy
        self.c = exploration_const
        self.num_simulations = num_simulations
        self.root = None
        self.debug = True

    def update_tree(self, two_last_moves):
        try:
            if self.debug:
                print("Inheriting previous tree root...")
            m1, m2 = two_last_moves
            self.root = self.root.children[m1].children[m2]
            return True
        except KeyError:
            if self.debug:
                print("Moves not found in previous tree. Initializing new tree...")
            return False

    def predict(self, board, turn, moves):
        return self.solve(board, turn, moves)[0]

    def solve(self, board: MnkBoard, turn: int, moves) -> Tuple[int, int]:
        start = time.time()
        if len(moves) < 2 or self.root is None:
            if self.debug:
                print("Initializing new tree...")
            self.root = MnkState(board, turn, self.policy, None, None)
        else:
            if not self.update_tree(moves[-2:]):
                self.root = MnkState(board, turn, self.policy, None, None)

        while time.time()-start < self.max_thinking_time and \
                self.total_rollout < self.max_rollout:
            self.loop()
        return self.get_results()

    def get_move_winrate(self, move):
        child = self.root.children.get(move, None)
        return self.score(child, 0) if child.n != 0 else None

    def get_p(self, temperature=1.0):
        if self.root is None:
            return None
        p = np.zeros((self.root.board.m * self.root.board.n,), dtype=np.float32)
        for child in self.root.children.values():
            i, j = child.last_move
            p[i * self.root.board.n + j] = child.n
        sum_p = np.sum(p)
        if temperature == 0.0:
            p = np.zeros_like(p)
            p[np.argmax(p)] = 1.0
        else:
            p = np.power(p, 1.0 / temperature)
            p /= np.sum(p)
        return p

    def get_results(self):
        best_child = None
        if self.total_rollout > 0 and len(self.root.children.values()) > 0:
            children = []
            for child in self.root.children.values():
                children.append((child, self.score(child, 0)))
            if self.temperature == 0.0:
                best_child = max(children, key=lambda child: child[1])[0]
            else:
                p = []
                for child, _ in children:
                    p.append(child.n)
                p = np.power(p, 1.0 / self.temperature)
                p /= np.sum(p)
                best_child_i = np.random.choice([i for i in range(len(children))], p=p)
                best_child = children[best_child_i][0]
            if self.debug:
                top_k = 5 if len(children) >= 5 else len(children)
                children.sort(key=lambda child: -child[1])
                print("\nTop %i moves:" % top_k)
                for child, score in children[:5]:
                    print("Move:", child.last_move, "- score: %.4f - w: %i - n: %i" %
                        (score, child.r, child.n)
                    )
                print("Played %i rollouts!" % self.rollout_count)
                print("Total: %i rollouts (inherited from previous trees)!" %
                    self.total_rollout)
        return best_child.last_move if (best_child and self.total_rollout > 0) else None, \
            self.get_p()

    def soft_selection(self, node):
        if self.temperature == 0.0:
            return max(node.children.values(), key=lambda child: self.score(child, 0))
        p = [n.n for n in node.children.values()]
        if np.sum(p) == 0.0:
            p = [1.0/len(p)] * len(p)
        else:
            p = np.power(p, 1.0 / self.temperature)
            p /= np.sum(p)
        return np.random.choice(list(node.children.values()), p=p)


    def selection(self):
        node = self.root
        while not node.is_leaf():
            selected_node = max(
                node.children.values(), key=lambda child: self.score(child, self.c))
            node = selected_node
        return node

    def choosing_policy(self, states):
        return random.choice(states)

    def expansion(self, node):
        if node.board.check_endgame():
            return node
        states = node.get_next_states()
        if not states:
            return node
        return self.choosing_policy(states)

    def simulation(self, node):
        self.rollout_count += self.num_simulations
        self.total_rollout += self.num_simulations
        if self.num_simulations == 1:
            return [node.rollout()]
        # node.board is deep-copied due to multiprocessing by default
        args = [(node.board, node.turn) for _ in range(self.num_simulations)]
        return self.pool.starmap(rollout, args)

    def backpropagation(self, node, winner, times=1):
        # node.turn means the next player
        if winner == node.turn:
            reward = 0
        elif winner == 0:  # a draw
            reward = 0.5
        else:
            reward = 1
        while node is not None:
            node.n += times
            node.r += reward * times
            node = node.parent
            reward = 1-reward

