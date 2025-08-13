import sys
import os
import random
from typing import Tuple

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame
from pygame import Rect
from pygame.font import Font

from utils import exception
import gui.colors as colors
from board_state.mnk_board import MnkBoard


pygame.init()


class MnkGUI:
    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        cell_size: int,
        name: str,
        fps: int,
        menu_font_size: int,
        button_font_size: int,
        symbol_font_size: int,
    ) -> None:
        # MAGIC NUMBERS: 1 and -1 are player symbols, 0 is empty cell
        self.cell_size = cell_size
        assert m >= k and n >= k, "m and n must be larger than k"
        assert k >= 3, "k must be larger than 3"
        self.m, self.n, self.k = m, n, k
        self.w = m * cell_size
        self.h = n * cell_size * 8 // 7
        self.center_w = self.w / 2
        self.center_h = self.h / 2
        self.menu_font_size = menu_font_size
        self.button_font_size = button_font_size
        self.symbol_font_size = symbol_font_size
        # self.button_font_size = self.h // 15
        # self.button_font_size_small = self.h // 15 * 3 // 4
        # self.symbol_font_size = self.h // 15
        self.board = MnkBoard(m, n, k)
        self.title = name.title()
        self.fps = fps
        self.turn_symbols = {0: None, 1: "X", -1: "O"}
        self.turn_colors = {
            0: None,
            1: colors.PENCIL_COLOR,
            -1: colors.PENCIL_COLOR,
        }
        self.window = pygame.display.set_mode((self.w, self.h))
        self.clock = pygame.time.Clock()
        self.state = None
        pygame.display.set_caption(self.title)
        self.cursor = pygame.SYSTEM_CURSOR_ARROW
        self.rect_cache = {}
        # None means human, otherwise bot
        self.player1 = None
        self.player2 = None
        self.moves = []

    def add_rect_to_cache(
        self, rect: Rect, left: int, top: int, width: int, height: int
    ):
        self.rect_cache[(left, top, width, height)] = rect

    def get_rect_from_cache(self, left: int, top: int, width: int, height: int):
        coor = (left, top, width, height)
        return self.rect_cache.get(coor, None)

    def clear_rect_cache(self):
        self.rect_cache = {}

    def render_rect(
        self,
        position,
        color,
        rect_width=1,
        border_radius=0,
        border_top_left_radius=-1,
        border_top_right_radius=-1,
        border_bottom_left_radius=-1,
        border_bottom_right_radius=-1,
        text_str=None,
        text_size=25,
        text_color=colors.BLACK,
    ):
        left, top, width, height = position
        rect = self.get_rect_from_cache(left, top, width, height)
        if rect is None:
            rect = Rect(left, top, width, height)
            self.add_rect_to_cache(rect, left, top, width, height)
        pygame.draw.rect(
            self.window,
            color,
            rect,
            rect_width,
            border_radius,
            border_top_left_radius,
            border_top_right_radius,
            border_bottom_left_radius,
            border_bottom_right_radius,
        )
        if text_str:
            font = Font(None, text_size)
            text = font.render(text_str, True, text_color)
            text_rect = text.get_rect(center=(left + width // 2, top + height // 2))
            self.window.blit(text, text_rect)
        return rect

    def render_board(self):
        rects = []
        for i in range(self.m):
            rects.append([])
            for j in range(self.n):
                symbol = self.board.index(i, j)
                rects[i].append(
                    self.render_rect(
                        (
                            j * self.cell_size,
                            i * self.cell_size,
                            self.cell_size,
                            self.cell_size,
                        ),
                        colors.PENCIL_COLOR,
                        rect_width=1,
                        text_str=self.turn_symbols[symbol],
                        text_size=self.symbol_font_size,
                        text_color=self.turn_colors[symbol],
                    )
                )
        return rects

    def render_ingame_bottom_text(self, text, text_color):
        self.render_rect(
            (
                self.w // 4,
                self.n * self.cell_size,
                self.w // 2,
                self.n * self.cell_size // 7,
            ),
            colors.WHITE,
            rect_width=-1,
            text_str=text,
            text_size=self.button_font_size,
            text_color=text_color,
        )

    def render_endgame_noti(self, res):
        symbol = self.turn_symbols[res]
        text = "Draw!" if symbol is None else f"{symbol} won!"
        self.render_ingame_bottom_text(text, colors.BLACK)

    def move(self, pos):
        self.board.put(self.state, pos)
        self.moves.append(pos)

    def bot_move(self, possible_pos):
        pos = self.player.predict(self.board, self.state, self.moves)
        assert pos in possible_pos, f"Invalid move: {pos}"
        self.move(pos)

    def change_state(self):
        res = self.board.check_endgame()
        if res == 0:
            self.state = -self.state
        else:
            self.state = 0
        return res

    def main(self):
        res = 0
        self.state = random.choice((1, -1))
        while True:
            pygame.mouse.set_cursor(self.cursor)
            self.window.fill(colors.PAPER_WHITE_COLOR)
            possible_pos = self.board.get_possible_pos()
            if self.state != 0:
                if len(possible_pos) == 0:
                    self.state = 0
                    res = 0
            rects = self.render_board()
            try:
                events = []
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    events.append(event)

                if self.state == 1 or self.state == -1:
                    self.player = self.player1 if self.state == 1 else self.player2
                    if self.player is None:
                        # read input from human player
                        for event in events:
                            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:  # left mouse
                                for i, row in enumerate(rects):
                                    for j, rect in enumerate(row):
                                        if rect.collidepoint(event.pos):
                                            if self.board.index(i, j) != 0:
                                                raise exception.Break
                                            self.move((i, j))
                                            res = self.change_state()
                                            raise exception.Break

                    else:
                        # bot turn
                        status_text = " Bot thinking..."
                        self.render_ingame_bottom_text(status_text, colors.BLACK)
                        pygame.display.update()
                        self.bot_move(possible_pos)
                        res = self.change_state()
                        raise exception.Break
                elif self.state == 0:
                    self.render_endgame_noti(res)
                else:
                    raise NotImplementedError(
                        "State %i is not implemented!" % self.state
                    )
            except exception.Break:
                pass
            pygame.display.update()
            self.clock.tick(self.fps)


if __name__ == "__main__":
    pass
