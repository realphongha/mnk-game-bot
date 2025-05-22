# cython: infer_types=True
import random
import numpy as np
cimport numpy as np
from libc.math cimport INFINITY


cdef class MnkState:
    cdef:
        public board
        public int turn
        str policy
        public last_move
        public MnkState parent
        public dict children
        public int n
        public float r
        # for alphazero
        public float prior

    def __init__(self, board, int turn, str policy, last_move, 
            MnkState parent, children=None, int n=0, float r=0.0) -> None:
        self.board = board
        self.turn = turn
        self.policy = policy
        self.last_move = last_move
        self.parent = parent
        self.children = children if children else {}
        self.n = n
        self.r = r

    def is_leaf(self):
        return not self.children or self.n == 0

    def get_next_states(self):
        pos = self.board.get_possible_pos()
        states = []
        for p in pos:
            if p not in self.children:
                new_board = self.board.duplicate()
                new_board.put(self.turn, p, False)
                new_state = MnkState(new_board, -self.turn, self.policy, 
                    p, self)
                self.children[p] = new_state
            else:
                new_state = self.children[p]
            states.append(new_state)
        return states

    def rollout(self):
        test_board = self.board.duplicate()
        return rollout(test_board, self.turn)


def rollout(board, int turn):
    cdef int res = board.check_endgame()
    cdef int i, j, index
    cdef list pos = board.get_possible_pos()
    while res == 0 and len(pos) != 0:
        index = random.randrange(0, len(pos))
        i, j = pos.pop(index)
        board.put(turn, (i, j), False)
        turn = -turn
        res = board.check_endgame(i, j)
    return res

