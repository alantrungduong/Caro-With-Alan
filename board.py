"""
board.py - Core Gomoku (Caro) board logic.

Pure game rules, no graphics and no AI.  The GUI and the AI both build on
this module, and the automated tests exercise it directly.

Board cells hold one of three integer values:
    EMPTY (0), HUMAN (1, drawn as X), AI (2, drawn as O).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

EMPTY = 0
HUMAN = 1  # X
AI = 2     # O

BOARD_SIZE = 30
WIN_LENGTH = 5

# The four line directions that matter for five-in-a-row:
# horizontal, vertical, main diagonal, anti diagonal.
DIRECTIONS: Tuple[Tuple[int, int], ...] = ((0, 1), (1, 0), (1, 1), (1, -1))

Move = Tuple[int, int]


def other_player(player: int) -> int:
    """Return the opponent of ``player``."""
    return AI if player == HUMAN else HUMAN


def symbol(player: int) -> str:
    """Human readable symbol for a cell value."""
    return {HUMAN: "X", AI: "O"}.get(player, ".")


class Board:
    """A 30x30 (configurable) Gomoku board with move validation and win/draw detection."""

    def __init__(self, size: int = BOARD_SIZE) -> None:
        self.size = size
        self.grid = np.zeros((size, size), dtype=np.int8)
        self.history: List[Tuple[int, int, int]] = []  # (row, col, player)
        self.winner: int = EMPTY
        self.winning_line: List[Move] = []

    # ------------------------------------------------------------------ basics
    def reset(self) -> None:
        """Clear the board and all game state."""
        self.grid.fill(EMPTY)
        self.history.clear()
        self.winner = EMPTY
        self.winning_line = []

    def copy(self) -> "Board":
        clone = Board(self.size)
        clone.grid = self.grid.copy()
        clone.history = list(self.history)
        clone.winner = self.winner
        clone.winning_line = list(self.winning_line)
        return clone

    @property
    def move_count(self) -> int:
        return len(self.history)

    @property
    def last_move(self) -> Optional[Move]:
        if not self.history:
            return None
        row, col, _ = self.history[-1]
        return (row, col)

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.size and 0 <= col < self.size

    def get(self, row: int, col: int) -> int:
        return int(self.grid[row, col])

    def is_empty(self, row: int, col: int) -> bool:
        return self.in_bounds(row, col) and self.grid[row, col] == EMPTY

    def is_valid_move(self, row: int, col: int) -> bool:
        """A move is legal when the game is not over and the cell is empty."""
        return self.winner == EMPTY and self.is_empty(row, col)

    def is_full(self) -> bool:
        return self.move_count >= self.size * self.size

    def is_draw(self) -> bool:
        return self.winner == EMPTY and self.is_full()

    def is_game_over(self) -> bool:
        return self.winner != EMPTY or self.is_full()

    def empty_cells(self) -> List[Move]:
        rows, cols = np.nonzero(self.grid == EMPTY)
        return list(zip(rows.tolist(), cols.tolist()))

    # ------------------------------------------------------------------- moves
    def place(self, row: int, col: int, player: int) -> bool:
        """Place a stone.  Returns False (and changes nothing) if illegal."""
        if player not in (HUMAN, AI):
            raise ValueError(f"invalid player {player!r}")
        if not self.is_valid_move(row, col):
            return False
        self.grid[row, col] = player
        self.history.append((row, col, player))
        line = self.check_win(row, col)
        if line:
            self.winner = player
            self.winning_line = line
        return True

    def undo(self) -> Optional[Tuple[int, int, int]]:
        """Remove the most recent stone.  Returns the removed (row, col, player)."""
        if not self.history:
            return None
        row, col, player = self.history.pop()
        self.grid[row, col] = EMPTY
        self.winner = EMPTY
        self.winning_line = []
        return (row, col, player)

    # ------------------------------------------------------------- win checks
    def check_win(self, row: int, col: int) -> Optional[List[Move]]:
        """
        If the stone at (row, col) completes five or more in a row, return the
        list of cells that form the winning sequence; otherwise return None.
        """
        player = self.grid[row, col]
        if player == EMPTY:
            return None
        for dr, dc in DIRECTIONS:
            cells: List[Move] = [(row, col)]
            # extend forwards
            r, c = row + dr, col + dc
            while self.in_bounds(r, c) and self.grid[r, c] == player:
                cells.append((r, c))
                r, c = r + dr, c + dc
            # extend backwards
            r, c = row - dr, col - dc
            while self.in_bounds(r, c) and self.grid[r, c] == player:
                cells.insert(0, (r, c))
                r, c = r - dr, c - dc
            if len(cells) >= WIN_LENGTH:
                return cells
        return None

    def find_winner(self) -> Tuple[int, List[Move]]:
        """Scan the whole board (used by tests and for safety after bulk setup)."""
        for row in range(self.size):
            for col in range(self.size):
                if self.grid[row, col] != EMPTY:
                    line = self.check_win(row, col)
                    if line:
                        return int(self.grid[row, col]), line
        return EMPTY, []

    # --------------------------------------------------------------- display
    def __str__(self) -> str:
        header = "   " + " ".join(f"{c:2d}" for c in range(self.size))
        rows = [header]
        for r in range(self.size):
            rows.append(f"{r:2d} " + " ".join(f" {symbol(int(v))}" for v in self.grid[r]))
        return "\n".join(rows)
