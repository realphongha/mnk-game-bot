import numpy as np
from .mnk_board import MnkBoard


def to_bitboard(board):
    bitboard = {}
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            turn = board[i, j]
            if turn == 0:
                continue
            if turn not in bitboard:
                bitboard[turn] = 0
            j = board.shape[1] - 1 - j
            bitmask = 1 << (j + i * (board.shape[1] + 1))
            bitboard[turn] += bitmask
    return bitboard


def to_board(bitboard, m, n):
    board = np.zeros((2, m, n), dtype=np.int8)
    for turn in bitboard:
        c = 1 if turn == -1 else 0
        for i in range(m):
            for j in range(n):
                j_ = n - 1 - j
                if bitboard[turn] & (1 << (j_ + i * (n + 1))):
                    board[c, i, j] = 1
    return board


if __name__ == "__main__":
    import time
    from utils.perf_monitor import PerfMonitorMixin

    k = 3
    board = [[-1, 1 , -1, 0 ],
             [0 , -1, 0 , -1],
             [-1, 1 , 0 , 0 ],
             [1 , -1, 0 , 0 ]]
    board = np.array(board).astype(np.int8)
    m = board.shape[0]
    n = board.shape[1]
    print(to_board(to_bitboard(board), m, n))
    # board = np.zeros((15, 15))
    mnk_board = MnkBoard(board.shape[1], board.shape[0], k, to_bitboard(board))
    for _ in range(10):
        start = time.time()
        res = mnk_board.check_endgame()
        PerfMonitorMixin.update_perf_static("check_endgame", time.time() - start)
    print("Result:", res)
    PerfMonitorMixin.get_all_perfs()

