"""
gui.py - Pygame interface for Caro AI (Human vs Computer) on a 30x30 board.

Layout (1200 x 700 window, fits a 1366 x 768 laptop display):

    +------------------------------------------+-------------------+
    |  board viewport (860 x 660)              |  CARO AI          |
    |  zoom with the mouse wheel,              |  status card      |
    |  drag to pan, click to play              |  difficulty       |
    |                                          |  restart / view   |
    +------------------------------------------+-------------------+

The viewport is a camera over the 30x30 board: ``cell`` is the current cell
size in pixels and (``ox``, ``oy``) is where the board's top-left corner sits
inside the viewport.  Every mouse position is mapped through that camera, so
clicks are accurate at any zoom level.

The AI runs in a background thread (see game.py), so the window keeps
redrawing and reacting while the computer thinks.
"""

from __future__ import annotations

import math
import time
from typing import Dict, Optional, Tuple

import pygame

from ai import DIFFICULTIES
from board import AI, BOARD_SIZE, EMPTY, HUMAN
from game import (
    STATUS_AI_THINKING,
    STATUS_AI_TURN,
    STATUS_AI_WON,
    STATUS_DRAW,
    STATUS_HUMAN_TURN,
    STATUS_HUMAN_WON,
    Game,
)

# ----------------------------------------------------------------- geometry
WINDOW_W, WINDOW_H = 1200, 700
VIEW_X, VIEW_Y, VIEW_W, VIEW_H = 20, 20, 860, 660
PANEL_X = VIEW_X + VIEW_W + 20            # 900
PANEL_W = WINDOW_W - PANEL_X              # 300
MIN_CELL, MAX_CELL = 12.0, 64.0
ZOOM_STEP = 1.15
DRAG_THRESHOLD = 4                        # px of motion before a click becomes a drag
FPS = 60

# ------------------------------------------------------------------- colours
BG = (236, 239, 241)
VIEW_BG = (207, 216, 220)
BOARD_BG = (241, 217, 167)
BOARD_BORDER = (121, 85, 61)
GRID = (165, 130, 92)
GRID_STRONG = (130, 98, 66)
STAR = (110, 80, 50)
PANEL_BG = (38, 50, 56)
PANEL_TEXT = (236, 239, 241)
PANEL_MUTED = (176, 190, 197)
PANEL_LINE = (69, 90, 100)
X_COLOR = (30, 90, 168)
O_COLOR = (200, 55, 45)
LAST_MOVE = (255, 224, 130)
LAST_RING = (120, 90, 40)
WIN_CELL = (185, 246, 202)
WIN_LINE = (46, 125, 50)
HOVER = (255, 255, 255, 110)
BTN = (55, 71, 79)
BTN_HOVER = (69, 90, 100)
BTN_ACTIVE = (0, 150, 136)
BTN_RESTART = (230, 81, 0)
BTN_RESTART_HOVER = (255, 109, 0)
BTN_TEXT = (255, 255, 255)

TITLE = "CARO AI — HUMAN VS COMPUTER"


def _font(size: int, bold: bool = False) -> pygame.font.Font:
    for name in ("segoeui", "arial", "dejavusans"):
        path = pygame.font.match_font(name, bold=bold)
        if path:
            return pygame.font.Font(path, size)
    return pygame.font.Font(None, size + 6)


class Button:
    def __init__(self, rect: pygame.Rect, label: str, value: str = "") -> None:
        self.rect = rect
        self.label = label
        self.value = value

    def hit(self, pos: Tuple[int, int]) -> bool:
        return self.rect.collidepoint(pos)


class Camera:
    """Maps board cells <-> viewport pixels for a zoomable, pannable board."""

    def __init__(self, size: int = BOARD_SIZE) -> None:
        self.size = size
        self.cell = 22.0
        self.ox = 0.0
        self.oy = 0.0
        self.fit()

    # -- state
    @property
    def board_px(self) -> float:
        return self.cell * self.size

    def fit(self) -> None:
        """Show the whole board, centred in the viewport."""
        self.cell = max(MIN_CELL, min(MAX_CELL, math.floor(min(VIEW_W, VIEW_H) / self.size)))
        self.clamp()

    def clamp(self) -> None:
        """Keep the board on screen: centred when smaller than the viewport,
        otherwise never leaving a gap at the viewport edges."""
        bp = self.board_px
        if bp <= VIEW_W:
            self.ox = (VIEW_W - bp) / 2
        else:
            self.ox = min(0.0, max(VIEW_W - bp, self.ox))
        if bp <= VIEW_H:
            self.oy = (VIEW_H - bp) / 2
        else:
            self.oy = min(0.0, max(VIEW_H - bp, self.oy))

    # -- transforms
    def cell_to_px(self, row: int, col: int) -> Tuple[float, float]:
        """Top-left window pixel of a cell (float precision)."""
        return (VIEW_X + self.ox + col * self.cell, VIEW_Y + self.oy + row * self.cell)

    def cell_rect(self, row: int, col: int) -> pygame.Rect:
        """Integer pixel rect of a cell, consistent with px_to_cell: a pixel
        belongs to column ``col`` iff ceil(x_col) <= px < ceil(x_col+1)."""
        x, y = self.cell_to_px(row, col)
        x2, y2 = self.cell_to_px(row + 1, col + 1)
        left, top = math.ceil(x), math.ceil(y)
        return pygame.Rect(left, top, math.ceil(x2) - left, math.ceil(y2) - top)

    def cell_center(self, row: int, col: int) -> Tuple[int, int]:
        x, y = self.cell_to_px(row, col)
        return (round(x + self.cell / 2), round(y + self.cell / 2))

    def px_to_cell(self, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Window pixel -> (row, col), or None if outside the board/viewport."""
        x, y = pos
        if not (VIEW_X <= x < VIEW_X + VIEW_W and VIEW_Y <= y < VIEW_Y + VIEW_H):
            return None
        col = math.floor((x - VIEW_X - self.ox) / self.cell)
        row = math.floor((y - VIEW_Y - self.oy) / self.cell)
        if 0 <= row < self.size and 0 <= col < self.size:
            return (row, col)
        return None

    # -- navigation
    def zoom(self, factor: float, anchor: Optional[Tuple[int, int]] = None) -> None:
        """Zoom by ``factor`` keeping the board point under ``anchor`` fixed."""
        if anchor is None:
            anchor = (VIEW_X + VIEW_W // 2, VIEW_Y + VIEW_H // 2)
        new_cell = max(MIN_CELL, min(MAX_CELL, self.cell * factor))
        if new_cell == self.cell:
            return
        ax, ay = anchor[0] - VIEW_X, anchor[1] - VIEW_Y
        bx = (ax - self.ox) / self.cell      # board coordinate under the anchor
        by = (ay - self.oy) / self.cell
        self.cell = new_cell
        self.ox = ax - bx * self.cell
        self.oy = ay - by * self.cell
        self.clamp()

    def pan(self, dx: float, dy: float) -> None:
        self.ox += dx
        self.oy += dy
        self.clamp()

    def visible_range(self) -> Tuple[int, int, int, int]:
        """(row0, row1, col0, col1) inclusive range of cells inside the viewport."""
        c0 = max(0, math.floor(-self.ox / self.cell))
        r0 = max(0, math.floor(-self.oy / self.cell))
        c1 = min(self.size - 1, math.floor((VIEW_W - self.ox) / self.cell))
        r1 = min(self.size - 1, math.floor((VIEW_H - self.oy) / self.cell))
        return r0, r1, c0, c1

    def is_visible(self, row: int, col: int) -> bool:
        r0, r1, c0, c1 = self.visible_range()
        return r0 <= row <= r1 and c0 <= col <= c1

    def ensure_visible(self, row: int, col: int) -> None:
        """Pan the minimum amount so that the cell is fully inside the viewport."""
        x, y = self.cell_to_px(row, col)
        x -= VIEW_X
        y -= VIEW_Y
        margin = self.cell
        if x < margin:
            self.ox += margin - x
        elif x + self.cell > VIEW_W - margin:
            self.ox -= x + self.cell - (VIEW_W - margin)
        if y < margin:
            self.oy += margin - y
        elif y + self.cell > VIEW_H - margin:
            self.oy -= y + self.cell - (VIEW_H - margin)
        self.clamp()


class GomokuGUI:
    def __init__(self, game: Optional[Game] = None) -> None:
        pygame.init()
        pygame.display.set_caption("Caro AI — Human vs Computer (30 x 30)")
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        self.clock = pygame.time.Clock()
        self.game = game or Game("medium")
        self.camera = Camera(self.game.board.size)
        self.fonts: Dict[str, pygame.font.Font] = {
            "title": _font(24, bold=True),
            "big": _font(19, bold=True),
            "normal": _font(15),
            "small": _font(13),
            "tiny": _font(11),
        }
        self.hover_cell: Optional[Tuple[int, int]] = None
        self.last_mouse: Tuple[int, int] = (0, 0)   # from motion events (wheel has no pos)
        self.running = True
        self.ai_delay_until = 0.0        # short pause so the human move is visible first
        self.frames = 0                  # frame counter (responsiveness tests)
        # dragging state
        self._drag_button: Optional[int] = None
        self._drag_start: Tuple[int, int] = (0, 0)
        self._drag_last: Tuple[int, int] = (0, 0)
        self._dragged = False

        # side panel widgets
        x = PANEL_X + 20
        w = PANEL_W - 40
        self.diff_buttons = [
            Button(pygame.Rect(x + i * ((w + 8) // 3), 302, (w - 16) // 3, 38), name.title(), name)
            for i, name in enumerate(DIFFICULTIES)
        ]
        self.restart_button = Button(pygame.Rect(x, 356, w, 44), "Restart  (R)")
        third = (w - 16) // 3
        self.fit_button = Button(pygame.Rect(x, 450, third, 32), "Fit  (F)", "fit")
        self.zoom_in_button = Button(pygame.Rect(x + third + 8, 450, third, 32), "Zoom +", "in")
        self.zoom_out_button = Button(pygame.Rect(x + 2 * (third + 8), 450, third, 32), "Zoom −", "out")

    # ------------------------------------------------------------- helpers
    def cell_at(self, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        return self.camera.px_to_cell(pos)

    def cell_rect(self, row: int, col: int) -> pygame.Rect:
        return self.camera.cell_rect(row, col)

    def cell_center(self, row: int, col: int) -> Tuple[int, int]:
        return self.camera.cell_center(row, col)

    # -------------------------------------------------------------- events
    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.KEYDOWN:
            self.on_key(event.key)
        elif event.type == pygame.MOUSEWHEEL:
            mouse = self.last_mouse
            if self._in_view(mouse):
                self.camera.zoom(ZOOM_STEP if event.y > 0 else 1 / ZOOM_STEP, mouse)
                self.hover_cell = self.cell_at(mouse)
        elif event.type == pygame.MOUSEMOTION:
            self.on_motion(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button in (1, 2, 3):
            self.on_mouse_down(event.pos, event.button)
        elif event.type == pygame.MOUSEBUTTONUP and event.button in (1, 2, 3):
            self.on_mouse_up(event.pos, event.button)

    def on_key(self, key: int) -> None:
        if key == pygame.K_r:
            self.restart()
        elif key == pygame.K_ESCAPE:
            self.running = False
        elif key in (pygame.K_1, pygame.K_2, pygame.K_3):
            self.game.set_difficulty(DIFFICULTIES[key - pygame.K_1])
        elif key == pygame.K_f:
            self.camera.fit()
        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.camera.zoom(ZOOM_STEP)
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.camera.zoom(1 / ZOOM_STEP)
        elif key in (pygame.K_LEFT, pygame.K_a):
            self.camera.pan(self.camera.cell * 2, 0)
        elif key in (pygame.K_RIGHT, pygame.K_d):
            self.camera.pan(-self.camera.cell * 2, 0)
        elif key in (pygame.K_UP, pygame.K_w):
            self.camera.pan(0, self.camera.cell * 2)
        elif key in (pygame.K_DOWN, pygame.K_s):
            self.camera.pan(0, -self.camera.cell * 2)

    def _in_view(self, pos: Tuple[int, int]) -> bool:
        return VIEW_X <= pos[0] < VIEW_X + VIEW_W and VIEW_Y <= pos[1] < VIEW_Y + VIEW_H

    def on_mouse_down(self, pos: Tuple[int, int], button: int) -> None:
        self.last_mouse = pos
        if self._in_view(pos):
            self._drag_button = button
            self._drag_start = pos
            self._drag_last = pos
            self._dragged = False
        elif button == 1:
            self.on_click(pos)

    def on_motion(self, pos: Tuple[int, int]) -> None:
        self.last_mouse = pos
        if self._drag_button is not None:
            dx = pos[0] - self._drag_last[0]
            dy = pos[1] - self._drag_last[1]
            if not self._dragged:
                sx = pos[0] - self._drag_start[0]
                sy = pos[1] - self._drag_start[1]
                if abs(sx) > DRAG_THRESHOLD or abs(sy) > DRAG_THRESHOLD:
                    self._dragged = True
            if self._dragged:
                self.camera.pan(dx, dy)
            self._drag_last = pos
        self.hover_cell = self.cell_at(pos)

    def on_mouse_up(self, pos: Tuple[int, int], button: int) -> None:
        if self._drag_button == button:
            was_drag = self._dragged
            self._drag_button = None
            self._dragged = False
            if button == 1 and not was_drag:
                self.on_click(pos)

    def on_click(self, pos: Tuple[int, int]) -> None:
        """A plain left click: side-panel buttons or placing a stone."""
        if self.restart_button.hit(pos):
            self.restart()
            return
        for btn in self.diff_buttons:
            if btn.hit(pos):
                self.game.set_difficulty(btn.value)
                return
        if self.fit_button.hit(pos):
            self.camera.fit()
            return
        if self.zoom_in_button.hit(pos):
            self.camera.zoom(ZOOM_STEP)
            return
        if self.zoom_out_button.hit(pos):
            self.camera.zoom(1 / ZOOM_STEP)
            return
        cell = self.cell_at(pos)
        if cell and self.game.human_move(*cell):
            self.ai_delay_until = time.perf_counter() + 0.15

    def restart(self) -> None:
        self.game.restart()
        self.camera.fit()

    # -------------------------------------------------------------- update
    def update(self) -> None:
        status = self.game.status
        if status == STATUS_AI_TURN and time.perf_counter() >= self.ai_delay_until:
            self.game.start_ai_move()
        elif status == STATUS_AI_THINKING:
            move = self.game.poll_ai()
            if move and not self.camera.is_visible(*move):
                self.camera.ensure_visible(*move)

    # --------------------------------------------------------------- draw
    def draw(self) -> None:
        self.frames += 1
        self.screen.fill(BG)
        self.draw_board()
        self.draw_panel()
        pygame.display.flip()

    def draw_board(self) -> None:
        board = self.game.board
        cam = self.camera
        view = pygame.Rect(VIEW_X, VIEW_Y, VIEW_W, VIEW_H)
        pygame.draw.rect(self.screen, VIEW_BG, view, border_radius=8)
        self.screen.set_clip(view)

        # board surface
        bx, by = cam.cell_to_px(0, 0)
        bp = cam.board_px
        pygame.draw.rect(self.screen, BOARD_BORDER, (bx - 5, by - 5, bp + 10, bp + 10), border_radius=5)
        pygame.draw.rect(self.screen, BOARD_BG, (bx, by, bp, bp))

        r0, r1, c0, c1 = cam.visible_range()

        # winning cells + last move highlight (under the grid lines)
        win_cells = set(board.winning_line)
        for r, c in win_cells:
            pygame.draw.rect(self.screen, WIN_CELL, cam.cell_rect(r, c))
        if board.last_move and board.last_move not in win_cells:
            pygame.draw.rect(self.screen, LAST_MOVE, cam.cell_rect(*board.last_move))

        # hover highlight for a legal human move
        if (self.hover_cell and self.game.status == STATUS_HUMAN_TURN
                and board.is_empty(*self.hover_cell) and self._drag_button is None):
            rect = cam.cell_rect(*self.hover_cell)
            overlay = pygame.Surface(rect.size, pygame.SRCALPHA)
            overlay.fill(HOVER)
            self.screen.blit(overlay, rect.topleft)

        # grid (every 5th line a little stronger to help orientation)
        for i in range(c0, c1 + 2):
            x = math.ceil(bx + i * cam.cell)
            colour = GRID_STRONG if i % 5 == 0 else GRID
            pygame.draw.line(self.screen, colour, (x, round(by)), (x, round(by + bp)), 1)
        for i in range(r0, r1 + 2):
            y = math.ceil(by + i * cam.cell)
            colour = GRID_STRONG if i % 5 == 0 else GRID
            pygame.draw.line(self.screen, colour, (round(bx), y), (round(bx + bp), y), 1)
        if cam.cell >= 14:
            for r in (7, 15, 22):
                for c in (7, 15, 22):
                    pygame.draw.circle(self.screen, STAR, cam.cell_rect(r, c).topleft, 3)

        # coordinates along the board edges
        if cam.cell >= 16:
            font = self.fonts["tiny"] if cam.cell < 24 else self.fonts["small"]
            for i in range(c0, c1 + 1):
                lbl = font.render(str(i), True, BOARD_BORDER)
                cx = cam.cell_center(0, i)[0]
                self.screen.blit(lbl, (cx - lbl.get_width() // 2, round(by) - lbl.get_height() - 6))
            for i in range(r0, r1 + 1):
                lbl = font.render(str(i), True, BOARD_BORDER)
                cy = cam.cell_center(i, 0)[1]
                self.screen.blit(lbl, (round(bx) - lbl.get_width() - 8, cy - lbl.get_height() // 2))

        # stones (only the visible ones)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                v = board.get(r, c)
                if v == HUMAN:
                    self.draw_x(r, c)
                elif v == AI:
                    self.draw_o(r, c)

        # last-move ring
        if board.last_move:
            pygame.draw.rect(self.screen, LAST_RING, cam.cell_rect(*board.last_move), 2)

        # winning line
        if len(board.winning_line) >= 2:
            start = cam.cell_center(*board.winning_line[0])
            end = cam.cell_center(*board.winning_line[-1])
            width = max(3, round(cam.cell / 7))
            pygame.draw.line(self.screen, WIN_LINE, start, end, width)
            pygame.draw.circle(self.screen, WIN_LINE, start, width)
            pygame.draw.circle(self.screen, WIN_LINE, end, width)

        self.screen.set_clip(None)
        pygame.draw.rect(self.screen, BOARD_BORDER, view, 2, border_radius=8)

    def draw_x(self, row: int, col: int) -> None:
        cx, cy = self.cell_center(row, col)
        d = max(3, round(self.camera.cell * 0.27))
        w = max(2, round(self.camera.cell / 10))
        pygame.draw.line(self.screen, X_COLOR, (cx - d, cy - d), (cx + d, cy + d), w)
        pygame.draw.line(self.screen, X_COLOR, (cx - d, cy + d), (cx + d, cy - d), w)

    def draw_o(self, row: int, col: int) -> None:
        cx, cy = self.cell_center(row, col)
        radius = max(3, round(self.camera.cell * 0.3))
        w = max(2, round(self.camera.cell / 10))
        pygame.draw.circle(self.screen, O_COLOR, (cx, cy), radius, w)

    # ---------------------------------------------------------------- panel
    def draw_panel(self) -> None:
        pygame.draw.rect(self.screen, PANEL_BG, (PANEL_X, 0, PANEL_W, WINDOW_H))
        x = PANEL_X + 20
        f = self.fonts
        mouse = pygame.mouse.get_pos()

        t1 = f["title"].render("CARO AI", True, PANEL_TEXT)
        t2 = f["big"].render("HUMAN VS COMPUTER", True, (128, 203, 196))
        self.screen.blit(t1, (x, 26))
        self.screen.blit(t2, (x, 58))
        sub = f["small"].render("30 x 30 board • five in a row wins", True, PANEL_MUTED)
        self.screen.blit(sub, (x, 84))
        pygame.draw.line(self.screen, PANEL_LINE, (x, 108), (WINDOW_W - 20, 108), 2)

        self.draw_status_card(x, 122)
        self.draw_legend(x, 226)

        lbl = f["normal"].render("Difficulty  (keys 1 / 2 / 3)", True, PANEL_MUTED)
        self.screen.blit(lbl, (x, 278))
        for btn in self.diff_buttons:
            active = btn.value == self.game.difficulty
            colour = BTN_ACTIVE if active else (BTN_HOVER if btn.hit(mouse) else BTN)
            pygame.draw.rect(self.screen, colour, btn.rect, border_radius=8)
            txt = f["normal"].render(btn.label, True, BTN_TEXT)
            self.screen.blit(txt, txt.get_rect(center=btn.rect.center))

        colour = BTN_RESTART_HOVER if self.restart_button.hit(mouse) else BTN_RESTART
        pygame.draw.rect(self.screen, colour, self.restart_button.rect, border_radius=8)
        txt = f["big"].render(self.restart_button.label, True, BTN_TEXT)
        self.screen.blit(txt, txt.get_rect(center=self.restart_button.rect.center))

        # view controls
        lbl = f["normal"].render(f"View   (zoom {self.camera.cell:.0f}px per cell)", True, PANEL_MUTED)
        self.screen.blit(lbl, (x, 424))
        for btn in (self.fit_button, self.zoom_in_button, self.zoom_out_button):
            colour = BTN_HOVER if btn.hit(mouse) else BTN
            pygame.draw.rect(self.screen, colour, btn.rect, border_radius=8)
            txt = f["small"].render(btn.label, True, BTN_TEXT)
            self.screen.blit(txt, txt.get_rect(center=btn.rect.center))

        y = 496
        lines = [
            ("How to play", True),
            ("• Click an empty cell to place your X.", False),
            ("• Mouse wheel: zoom at the cursor.", False),
            ("• Drag (any button) or arrow keys: pan.", False),
            ("• First to five in a row wins.", False),
            ("AI: Minimax + Alpha-Beta (no neural net).", False),
        ]
        for text, head in lines:
            font = f["normal"] if head else f["small"]
            colour = PANEL_TEXT if head else PANEL_MUTED
            self.screen.blit(font.render(text, True, colour), (x, y))
            y += 22 if head else 18

        info = self.game.last_ai_info
        if info:
            y = WINDOW_H - 58
            self.screen.blit(f["small"].render("Last AI move:", True, PANEL_MUTED), (x, y))
            self.screen.blit(f["tiny"].render(info, True, PANEL_MUTED), (x, y + 18))
        moves = f["small"].render(f"Moves played: {self.game.board.move_count}", True, PANEL_MUTED)
        self.screen.blit(moves, (x, WINDOW_H - 24))

    def draw_status_card(self, x: int, y: int) -> None:
        status = self.game.status
        rect = pygame.Rect(x, y, PANEL_W - 40, 92)
        if status == STATUS_HUMAN_WON:
            colour, head, sub = (46, 125, 50), "YOU WIN!", "Five in a row for X."
        elif status == STATUS_AI_WON:
            colour, head, sub = (183, 28, 28), "COMPUTER WINS", "O made five in a row."
        elif status == STATUS_DRAW:
            colour, head, sub = (93, 64, 55), "DRAW", "The board is full."
        elif status == STATUS_HUMAN_TURN:
            colour, head, sub = X_COLOR, "YOUR TURN  (X)", "Click a cell on the board."
        else:
            dots = "." * (1 + int(time.time() * 3) % 3)
            colour, head, sub = O_COLOR, f"COMPUTER THINKING{dots}", "Minimax search running"
        pygame.draw.rect(self.screen, colour, rect, border_radius=10)
        head_s = self.fonts["big"].render(head, True, BTN_TEXT)
        sub_s = self.fonts["small"].render(sub, True, (240, 240, 240))
        self.screen.blit(head_s, (x + 14, y + 14))
        self.screen.blit(sub_s, (x + 14, y + 48))
        if status in (STATUS_HUMAN_WON, STATUS_AI_WON, STATUS_DRAW):
            hint = self.fonts["small"].render("Press R or Restart to play again", True, (240, 240, 240))
            self.screen.blit(hint, (x + 14, y + 68))

    def draw_legend(self, x: int, y: int) -> None:
        f = self.fonts["normal"]
        d = 8
        cx, cy = x + 14, y + 12
        pygame.draw.line(self.screen, X_COLOR, (cx - d, cy - d), (cx + d, cy + d), 3)
        pygame.draw.line(self.screen, X_COLOR, (cx - d, cy + d), (cx + d, cy - d), 3)
        self.screen.blit(f.render("You  (X, moves first)", True, PANEL_TEXT), (x + 34, y + 2))
        cy2 = y + 36
        pygame.draw.circle(self.screen, O_COLOR, (cx, cy2), d, 3)
        self.screen.blit(f.render("Computer  (O)", True, PANEL_TEXT), (x + 34, cy2 - 10))

    # ---------------------------------------------------------------- loop
    def run(self) -> None:
        while self.running:
            for event in pygame.event.get():
                self.handle_event(event)
            self.update()
            self.draw()
            self.clock.tick(FPS)
        pygame.quit()


def run(difficulty: str = "medium") -> None:
    GomokuGUI(Game(difficulty)).run()
