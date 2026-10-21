"""
game.py - Game controller: turn handling, difficulty, restart and the
background AI thread.

Keeping this logic out of gui.py means the whole game flow can be tested
headlessly (see tests/test_game.py) and the GUI stays a thin view layer.
"""

from __future__ import annotations

import threading
from typing import Optional

from ai import DIFFICULTIES, GomokuAI
from board import AI, EMPTY, HUMAN, Board, Move

STATUS_HUMAN_TURN = "human_turn"
STATUS_AI_THINKING = "ai_thinking"
STATUS_AI_TURN = "ai_turn"        # AI to move but search not started yet
STATUS_HUMAN_WON = "human_won"
STATUS_AI_WON = "ai_won"
STATUS_DRAW = "draw"


class Game:
    """Owns the board and the AI, and drives the human/AI turn cycle."""

    def __init__(self, difficulty: str = "medium", human_starts: bool = True) -> None:
        self.board = Board()
        self.ai = GomokuAI(difficulty, player=AI)
        self.human_starts = human_starts
        self.current_player = HUMAN if human_starts else AI
        self._thread: Optional[threading.Thread] = None
        self._result: Optional[Move] = None
        self._generation = 0          # bumps on restart to discard stale AI results
        self.last_ai_move: Optional[Move] = None
        self.last_ai_info: str = ""

    # --------------------------------------------------------------- state
    @property
    def difficulty(self) -> str:
        return self.ai.difficulty

    def set_difficulty(self, difficulty: str) -> None:
        """Change the AI level; takes effect from the AI's next move."""
        if difficulty not in DIFFICULTIES:
            raise ValueError(difficulty)
        self.ai.set_difficulty(difficulty)

    @property
    def ai_thinking(self) -> bool:
        """True from start_ai_move() until poll_ai() has applied the result.

        This is deliberately not ``thread.is_alive()``: a fast search can
        finish between two frames, and the status must stay "thinking" until
        the result is consumed, otherwise the GUI would start a new search.
        """
        return self._thread is not None

    @property
    def status(self) -> str:
        if self.board.winner == HUMAN:
            return STATUS_HUMAN_WON
        if self.board.winner == AI:
            return STATUS_AI_WON
        if self.board.is_full():
            return STATUS_DRAW
        if self.current_player == HUMAN:
            return STATUS_HUMAN_TURN
        return STATUS_AI_THINKING if self.ai_thinking else STATUS_AI_TURN

    def is_over(self) -> bool:
        return self.board.is_game_over()

    def restart(self) -> None:
        """Reset the board and turn; any AI search still running is discarded."""
        self._generation += 1
        if self._thread is not None and self._thread.is_alive():
            # The old search keeps running on its own copy of the board until
            # its budget expires; give the new game a fresh engine so the two
            # never share search state.
            self.ai = GomokuAI(self.ai.difficulty, player=AI)
        self._thread = None
        self._result = None
        self.board.reset()
        self.current_player = HUMAN if self.human_starts else AI
        self.last_ai_move = None
        self.last_ai_info = ""

    # --------------------------------------------------------------- human
    def human_move(self, row: int, col: int) -> bool:
        """Apply a human move if it is the human's turn and the cell is legal."""
        if self.status != STATUS_HUMAN_TURN:
            return False
        if not self.board.place(row, col, HUMAN):
            return False
        if not self.board.is_game_over():
            self.current_player = AI
        return True

    # ------------------------------------------------------------------ AI
    def start_ai_move(self) -> bool:
        """Launch the AI search in a background thread (GUI stays responsive)."""
        if self.status != STATUS_AI_TURN:
            return False
        generation = self._generation
        board_copy = self.board.copy()
        holder = {}

        def worker() -> None:
            move = self.ai.choose_move(board_copy)
            holder["move"] = move
            holder["gen"] = generation

        self._result = None
        self._holder = holder
        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()
        return True

    def poll_ai(self) -> Optional[Move]:
        """
        Call every frame.  When the AI thread has finished, apply its move
        and return it; otherwise return None.
        """
        if self._thread is None or self._thread.is_alive():
            return None
        holder = getattr(self, "_holder", {})
        self._thread = None
        if holder.get("gen") != self._generation or "move" not in holder:
            return None  # result belongs to a game that was restarted
        return self._apply_ai_move(holder["move"])

    def ai_move_sync(self) -> Optional[Move]:
        """Compute and apply the AI move on the calling thread (used by tests)."""
        if self.status != STATUS_AI_TURN:
            return None
        move = self.ai.choose_move(self.board)
        return self._apply_ai_move(move)

    def _apply_ai_move(self, move: Move) -> Optional[Move]:
        row, col = move
        if not self.board.place(row, col, AI):
            # Should never happen: the AI only proposes empty cells.
            fallback = self.board.empty_cells()
            if not fallback:
                return None
            row, col = fallback[0]
            self.board.place(row, col, AI)
        self.last_ai_move = (row, col)
        self.last_ai_info = (
            f"{self.ai.difficulty.title()} | {self.ai.last_reason} | "
            f"{self.ai.last_nodes} nodes | {self.ai.last_time:.2f}s"
        )
        if not self.board.is_game_over():
            self.current_player = HUMAN
        return (row, col)
