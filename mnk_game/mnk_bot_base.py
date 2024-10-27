from typing import Tuple
import numpy as np
from abc import ABC, abstractmethod
from board_state.mnk_board import MnkBoard


class MnkGameBotBase(ABC):
    def __init__(self, max_thinking_time) -> None:
        super().__init__()
        self.max_thinking_time = max_thinking_time
        self.temperature = 0.0

    @abstractmethod
    def solve(self, board: MnkBoard, turn: int, moves) -> Tuple[int, int]:
        pass
