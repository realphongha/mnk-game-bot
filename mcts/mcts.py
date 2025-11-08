import math
import cython
from abc import ABC, abstractmethod
from collections import Counter
from utils.perf_monitor import PerfMonitorMixin


class MonteCarloTreeSearchMixin(ABC, PerfMonitorMixin):

    root = None
    rollout_count = 0
    total_rollout = 0

    def loop(self) -> None:
        node = self.selection()
        node = self.expansion(node)
        winners = self.simulation(node)
        winners = Counter(winners)
        for winner in winners:
            self.backpropagation(node, winner, times=winners[winner])

    @abstractmethod
    def selection(self):
        pass

    @abstractmethod
    def expansion(self):
        pass

    @abstractmethod
    def simulation(self):
        pass

    @abstractmethod
    def backpropagation(self):
        pass

    @staticmethod
    def score(node, c):
        # ucb1
        if node.n == 0:
            return float("inf")

        # Fix the memleak: Call the weak reference to get the parent
        parent_node = node.parent()
        if parent_node is None:
            # This can happen if the node is the root
            return float("inf")

        return node.r/node.n + c * math.sqrt(math.log(parent_node.n)/node.n)

