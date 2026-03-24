import time
import random
import multiprocessing
import logging
import os
from typing import Tuple

import numpy as np

from mcts.mcts import MonteCarloTreeSearchMixin
from .mnk_bot_base import MnkGameBotBase
from board_state.mnk_state import MnkState, rollout
from board_state.mnk_board import MnkBoard


def init_worker():
    """Sets a unique random seed for each worker process."""
    seed = (os.getpid() + int(time.time() * 1000)) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    from board_state.mnk_state import seed_c_rand
    seed_c_rand(seed)


class MonteCarloTreeSearchMnkGame(MonteCarloTreeSearchMixin, MnkGameBotBase):
    def __init__(self, max_thinking_time, max_rollout, processes, policy,
                 exploration_const, num_simulations) -> None:
        super().__init__(max_thinking_time)
        self.max_rollout = max_rollout
        self.processes = processes
        if self.processes != 1:
            self.pool = multiprocessing.Pool(self.processes, initializer=init_worker)
        self.policy = policy
        self.c = exploration_const
        self.num_simulations = num_simulations
        self.root = None
        self.debug = True

    def __del__(self):
        # Avoid the multiprocessing pool leak
        if self.processes != 1 and hasattr(self, 'pool'):
            self.pool.close()
            self.pool.join()
            logging.info("Closed MCTS multiprocessing pool.")

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

    def predict(self, board, turn, moves):
        return self.solve(board, turn, moves)[0]

    def solve(self, board: MnkBoard, turn: int, moves) -> Tuple[int, int]:
        if len(moves) < 2 or self.root is None:
            if self.debug:
                logging.info("Initializing new tree...")
            self.root = MnkState(board, turn, self.policy, None, None)
        else:
            if not self.update_tree(moves[-2:]):
                self.root = MnkState(board, turn, self.policy, None, None)

        start = time.time()
        num_sim = 0
        while time.time()-start < self.max_thinking_time and \
                self.total_rollout < self.max_rollout:
            self.loop()
            num_sim += self.num_simulations
        res = self.get_results()
        logging.debug(f"MCTS simulations: {num_sim}")
        return res

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
        if temperature <= 0.05:
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
                children.append((child, child.n))
            if self.temperature <= 0.05:
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
                logging.info("\nTop %i moves:" % top_k)
                for child, _ in children[:5]:
                    logging.info(f"Move: {child.last_move} - winrate: %.4f - w: %i - n: %i" %
                        (child.r/child.n, child.r, child.n)
                    )
                logging.info("Played %i rollouts!" % self.rollout_count)
                logging.info("Total: %i rollouts (inherited from previous trees)!" %
                    self.total_rollout)
        return best_child.last_move if (best_child and self.total_rollout > 0) else None, \
            self.get_p()

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
        args = [(node.board.duplicate(), node.turn) for _ in range(self.num_simulations)]
        if self.processes != 1:
            return self.pool.starmap(rollout, args)
        else:
            return [rollout(*arg) for arg in args]

    def backpropagation(self, node, winner, times=1):
        # node.turn means the next player
        if winner == node.turn:
            reward = 0
        elif winner == 0:  # a draw
            reward = 0.5
        else:
            reward = 1
        current_node = node
        while current_node is not None:
            current_node.n += times
            current_node.r += reward * times

            # Fix the memleak: Call the weak reference to get the parent
            if current_node.parent is None:
                current_node = None # At the root
            else:
                current_node = current_node.parent() # Call the weakref

            reward = 1-reward

