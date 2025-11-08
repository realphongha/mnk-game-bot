# cython: infer_types=True
import random
import numpy as np
cimport numpy as np
from libc.math cimport INFINITY
import weakref

# C imports for speed
from libc.stdlib cimport rand, srand
from libc.time cimport time

# Seed the C random number generator ONCE when the module is loaded
srand(<unsigned int>time(NULL))


cdef class MnkState:
    cdef:
        object __weakref__
        public board
        public int turn
        str policy
        public last_move
        public object parent
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
        self.parent = weakref.ref(parent) if parent is not None else None
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
        # Call our new, fast C-function
        test_board = self.board.duplicate()
        return c_rollout(test_board, self.turn)


cdef int c_rollout(board, int turn):
    """
    Fast, C-optimized rollout function.
    Uses C-level rand() and an O(1) "swap-and-pop" trick.
    """
    cdef int res = board.check_endgame()
    cdef int i, j, index, n_pos
    
    # Get moves as a Python list
    cdef list pos = board.get_possible_pos()
    n_pos = len(pos) # Use a C int for the counter

    while res == 0 and n_pos > 0:
        # 1. Use fast C-level rand()
        index = rand() % n_pos 
        
        # 2. Get the move
        (i, j) = pos[index]
        
        # 3. Play the move
        board.put(turn, (i, j), False)
        turn = -turn
        res = board.check_endgame(i, j) # check_endgame is already fast C

        # 4. Perform O(1) "swap-and-pop"
        # Replace the used move with the last move in the list
        pos[index] = pos[n_pos - 1]
        # Decrement the "size" of the list
        n_pos -= 1

    return res


# This Python-callable function is just a wrapper for our fast C-function
def rollout(board, int turn):
    return c_rollout(board, turn)

