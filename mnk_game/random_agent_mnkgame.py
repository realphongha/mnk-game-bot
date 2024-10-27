import random

from .mnk_bot_base import MnkGameBotBase


class RandomAgentMnkGame(MnkGameBotBase):
    def __init__(self):
        pass

    def solve(self, board, turn, moves):
        return random.choice(board.get_possible_pos()), None

