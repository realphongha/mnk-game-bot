# cython: infer_types=True
import numpy as np
cimport numpy as np
cimport cython
from gmpy2 cimport *
import_gmpy2()   # needed to initialize the C-API


DTYPE = np.uint8


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef bint check_board_cdef(mpz bb, int n, int k):
    cdef mpz res
    cdef int i
    res = bb

    # vertical
    for i in range(1, k):
        res &= (bb >> i)
    if res:
        return True
    
    # horizontal
    res = bb
    for i in range(1, k):
        res &= (bb >> ((n+1)*i))
    if res:
        return True
    
    # diagonal \
    res = bb
    for i in range(1, k):
        res &= (bb >> (n*i))
    if res:
        return True
    
    # diagonal /
    res = bb
    for i in range(1, k):
        res &= (bb >> ((n+2)*i))
    if res:
        return True
    return False


cdef class MnkBoard:
    cdef:
        public int m
        public int n
        int k
        dict board

    def __init__(self, int m, int n, int k, board_copy=None):
        # MAGIC NUMBERS: 1 and -1 are player symbols, 0 is empty cell
        self.m = m  # board width
        self.n = n  # board height
        self.k = k  # k-in-a-row for final win
        self.reset_board()
        if board_copy is not None:
            for turn in board_copy:
                self.board[turn] = mpz(board_copy[turn])

    def get_board(self):
        # for debug only
        return self.board

    def duplicate(self):
        return MnkBoard(self.m, self.n, self.k, self.board)

    def reset_board(self):
        self.board = {}

    def get_possible_pos(self):
        cdef int i, j
        cdef list res = []
        for i in range(self.m):
            for j in range(self.n):
                if self.is_empty(i, j):
                    res.append((i, j))
        return res

    def put(self, int turn, position, display=False):
        cdef int i, j 
        cdef mpz bitmask
        # assert turn == 1 or turn == -1, "Invalid player: %i" % turn
        if display:
            print("%s played (%i, %i)" % (
                    "Player 1" if turn == 1 else "Player 2", position[0], position[1]
                ))
        i, j = position
        j = self.n - 1 - j 
        bitmask = mpz(1) << (j + i*(self.n+1))
        # assert bitmask >= 0, f"Int overflow: 1 << {j + i*(self.n+1)} = {bitmask}"
        if turn not in self.board:
            self.board[turn] = mpz(0)
        self.board[turn] += bitmask
    
    cdef bint get(self, int turn, int i, int j):
        cdef mpz bitmask
        bitmask = mpz(1) << (self.n - 1 - j + i*(self.n+1))
        # assert bitmask >= 0, f"Int overflow: 1 << {self.n - 1 - j + i*(self.n+1)} = {bitmask}"
        return self.board.get(turn, mpz(0)) & bitmask

    def index(self, int i, int j):
        for turn in self.board.keys():
            if self.get(turn, i, j):
                return turn
        return 0

    cdef bint is_empty(self, int i, int j):
        return not (self.get(1, i, j) or self.get(-1, i, j))

    def check_endgame(self, int last_i=-1, int last_j=-1):
        # check if there is a winner
        # 0 maybe draw or not endgame
        cdef bint p1, p2
        for turn in self.board.keys():
            res = check_board_cdef(self.board[turn], self.n, self.k)
            if res:
                return turn
        return 0

