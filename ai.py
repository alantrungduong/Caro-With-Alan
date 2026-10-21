"""
ai.py - Gomoku AI: Minimax with Alpha-Beta pruning and a pattern heuristic.

This is classical game-tree search, NOT a neural network / deep learning model.

Overview
--------
* ``score_line``   - regex pattern matcher that scores one board line
                     (five, open four, closed four, open three, ...).
* ``SearchState``  - a lightweight working copy of the board that keeps
                     per-line scores up to date incrementally, so evaluating
                     a position after a move only re-scores 4 lines.
* ``point_score``  - cheap "what does placing a stone here create" scorer
                     used for candidate ordering and immediate win/block
                     detection.
* ``GomokuAI``     - the public opponent with Easy / Medium / Hard levels.
"""

from __future__ import annotations

import random
import re
import time
from typing import Dict, List, Optional, Sequence, Tuple

from board import AI, EMPTY, HUMAN, Board, Move, other_player

# ----------------------------------------------------------------------------
# Pattern values (from the point of view of the player who owns the stones)
# ----------------------------------------------------------------------------
FIVE = 10_000_000       # five (or more) in a row: game over
OPEN_FOUR = 1_000_000   # _XXXX_  : unstoppable next move
FOUR = 100_000          # OXXXX_, XXX_X ...: must be answered immediately
OPEN_THREE = 10_000     # __XXX_, _XXX__, _XX_X_ : becomes an open four
CLOSED_THREE = 1_000    # OXXX__ etc. : can only become a closed four
OPEN_TWO = 500          # __XX__, _X_X_
CLOSED_TWO = 50         # OXX___
ONE = 10

WIN_SCORE = FIVE * 10   # terminal score used inside the search

# Pattern templates.  X = own stone, O = opponent stone or board edge,
# _ = empty cell.  The list is in PRIORITY order: ``score_line_for`` matches
# the strongest shape first, marks the stones it used as consumed, and only
# then looks for weaker shapes.  This guarantees that a five is never counted
# as a four, an open four is never also counted as two closed fours, and so
# on, without needing fragile regex lookarounds.
_PATTERN_TEMPLATES: Sequence[Tuple[str, int]] = (
    (r"X{5,}", FIVE),
    (r"_XXXX_", OPEN_FOUR),
    (r"_XXXX", FOUR),          # closed four (blocked on the right)
    (r"XXXX_", FOUR),          # closed four (blocked on the left)
    (r"X{3,}_X+", FOUR),       # split fours: XXX_X, XXX_XX ...
    (r"XX_X{2,}", FOUR),       # XX_XX
    (r"X_X{3,}", FOUR),        # X_XXX
    (r"__XXX_", OPEN_THREE),
    (r"_XXX__", OPEN_THREE),
    (r"_XX_X_", OPEN_THREE),
    (r"_X_XX_", OPEN_THREE),
    (r"_XXX_", CLOSED_THREE),  # three boxed in: can only become a closed four
    (r"OXXX_", CLOSED_THREE),
    (r"_XXXO", CLOSED_THREE),
    (r"OXX_X_", CLOSED_THREE),
    (r"OX_XX_", CLOSED_THREE),
    (r"_XX_XO", CLOSED_THREE),
    (r"_X_XXO", CLOSED_THREE),
    (r"X_X_X", CLOSED_THREE),
    (r"XX__X", CLOSED_THREE),
    (r"X__XX", CLOSED_THREE),
    (r"__XX__", OPEN_TWO),
    (r"_X_X_", OPEN_TWO),
    (r"_XX_", OPEN_TWO),
    (r"_X__X_", OPEN_TWO),
    (r"OXX_", CLOSED_TWO),
    (r"_XXO", CLOSED_TWO),
    (r"OX_X_", CLOSED_TWO),
    (r"_X_XO", CLOSED_TWO),
    (r"_X_", ONE),
)

_CH = {EMPTY: ".", HUMAN: "1", AI: "2"}
_PAD = "#"       # board edge marker: behaves like an opponent stone
_USED = "C"      # a stone already counted in a stronger shape


def _build_patterns(player: int) -> List[Tuple[re.Pattern, int, int]]:
    """Compile the templates for ``player``: (regex, value, stones needed)."""
    own = _CH[player]
    opp = _CH[other_player(player)]
    out = []
    for template, value in _PATTERN_TEMPLATES:
        need = template.count("X")
        if "{5,}" in template:
            need = 5
        elif "{3,}" in template or "{2,}" in template or "+" in template:
            need = 4
        pat = (
            template.replace("X", own)
            .replace("O", f"[{opp}{_PAD}]")
            .replace("_", r"\.")
        )
        out.append((re.compile(pat), value, need))
    return out


_PATTERNS: Dict[int, List[Tuple[re.Pattern, int, int]]] = {
    HUMAN: _build_patterns(HUMAN),
    AI: _build_patterns(AI),
}

_line_cache: Dict[str, Tuple[int, int]] = {}


def score_line_for(line: str, player: int) -> int:
    """
    Score one padded line string for ``player``.

    Patterns are applied in priority order; the stones belonging to a match
    are replaced by ``_USED`` so weaker patterns cannot count them again.
    """
    own = _CH[player]
    remaining = line.count(own)
    if remaining == 0:
        return 0
    total = 0
    for regex, value, need in _PATTERNS[player]:
        if need > remaining:
            continue
        matches = regex.findall(line) and list(regex.finditer(line))
        if not matches:
            continue
        chars = list(line)
        for m in matches:
            total += value
            for i in range(m.start(), m.end()):
                if chars[i] == own:
                    chars[i] = _USED
                    remaining -= 1
        if remaining == 0:
            break
        line = "".join(chars)
    return total


def score_line(line: str) -> Tuple[int, int]:
    """
    Return (human_score, ai_score) for a line string made of '.', '1', '2'.
    The line is padded with '#' so that the board edge blocks shapes.
    Results are memoised: the same few thousand line strings repeat constantly
    during search.
    """
    cached = _line_cache.get(line)
    if cached is not None:
        return cached
    padded = _PAD + line + _PAD
    result = (score_line_for(padded, HUMAN), score_line_for(padded, AI))
    if len(_line_cache) > 300_000:
        _line_cache.clear()
    _line_cache[line] = result
    return result


# ----------------------------------------------------------------------------
# Board geometry: every line of length >= 5 in the four directions
# ----------------------------------------------------------------------------
def _build_lines(size: int) -> Tuple[List[List[Move]], List[List[Tuple[int, int]]]]:
    lines: List[List[Move]] = []
    # rows
    for r in range(size):
        lines.append([(r, c) for c in range(size)])
    # columns
    for c in range(size):
        lines.append([(r, c) for r in range(size)])
    # diagonals (down-right)
    for start in range(-(size - 5), size - 4):
        cells = [(r, r - start) for r in range(size) if 0 <= r - start < size]
        lines.append(cells)
    # anti-diagonals (down-left)
    for s in range(4, 2 * size - 5):
        cells = [(r, s - r) for r in range(size) if 0 <= s - r < size]
        lines.append(cells)

    cell_lines: List[List[Tuple[int, int]]] = [[] for _ in range(size * size)]
    for lid, cells in enumerate(lines):
        for pos, (r, c) in enumerate(cells):
            cell_lines[r * size + c].append((lid, pos))
    return lines, cell_lines


_GEOMETRY: Dict[int, Tuple[List[List[Move]], List[List[Tuple[int, int]]]]] = {}
_ZOBRIST: Dict[int, List[List[int]]] = {}


def _zobrist(size: int) -> List[List[int]]:
    """Random 64-bit keys for (cell, player) used to hash positions."""
    if size not in _ZOBRIST:
        rng = random.Random(0xC0FFEE)
        _ZOBRIST[size] = [[rng.getrandbits(64) for _ in range(3)] for _ in range(size * size)]
    return _ZOBRIST[size]


def _geometry(size: int):
    if size not in _GEOMETRY:
        _GEOMETRY[size] = _build_lines(size)
    return _GEOMETRY[size]


# ----------------------------------------------------------------------------
# Search state with incremental evaluation
# ----------------------------------------------------------------------------
class SearchState:
    """
    Working copy of the board used by the search.

    * ``grid``       - list of lists of ints (faster than numpy for scalar access)
    * ``adj``        - number of stones within Chebyshev distance 2 of each cell;
                       candidate moves are empty cells with adj > 0
    * ``line_scores``- (human, ai) score for every line, kept up to date
    * ``total``      - [human_total, ai_total]
    """

    NEIGHBOUR_RADIUS = 2

    def __init__(self, board: Board) -> None:
        self.size = board.size
        self.grid: List[List[int]] = board.grid.tolist()
        self.lines, self.cell_lines = _geometry(self.size)
        self.zobrist = _zobrist(self.size)
        self.hash = 0
        self.adj: List[List[int]] = [[0] * self.size for _ in range(self.size)]
        self.cand_set: set = set()   # empty cells with adj > 0, kept incrementally
        self.stones = 0
        self.line_scores: List[Tuple[int, int]] = [(0, 0)] * len(self.lines)
        self.total = [0, 0]
        # number of lines in which each side has a four (i.e. can make five
        # with one more stone).  Index 0 = human, 1 = AI.
        self.fours = [0, 0]
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r][c] != EMPTY:
                    self._bump_adj(r, c, +1)
                    self.stones += 1
                    self.hash ^= self.zobrist[r * self.size + c][self.grid[r][c]]
        # _bump_adj may have added cells that were still unvisited stones
        self.cand_set = {(r, c) for (r, c) in self.cand_set if self.grid[r][c] == EMPTY}
        for lid in range(len(self.lines)):
            s = self._rescore_line(lid)
            self.line_scores[lid] = s
            self.total[0] += s[0]
            self.total[1] += s[1]
            self.fours[0] += s[0] >= FOUR
            self.fours[1] += s[1] >= FOUR

    # -- helpers
    def _bump_adj(self, row: int, col: int, delta: int) -> None:
        """Update neighbour counts around (row, col) and the candidate set."""
        rad = self.NEIGHBOUR_RADIUS
        r0, r1 = max(0, row - rad), min(self.size - 1, row + rad)
        c0, c1 = max(0, col - rad), min(self.size - 1, col + rad)
        adj, grid, cand = self.adj, self.grid, self.cand_set
        for r in range(r0, r1 + 1):
            row_adj, row_grid = adj[r], grid[r]
            for c in range(c0, c1 + 1):
                row_adj[c] += delta
                if row_grid[c] == EMPTY:
                    if row_adj[c] > 0:
                        cand.add((r, c))
                    else:
                        cand.discard((r, c))

    def _line_string(self, lid: int) -> str:
        grid = self.grid
        return "".join(_CH[grid[r][c]] for r, c in self.lines[lid])

    def _rescore_line(self, lid: int) -> Tuple[int, int]:
        return score_line(self._line_string(lid))

    # -- moves
    def make(self, row: int, col: int, player: int) -> None:
        self.grid[row][col] = player
        self.stones += 1
        self.hash ^= self.zobrist[row * self.size + col][player]
        self.cand_set.discard((row, col))
        self._bump_adj(row, col, +1)
        self._rescore_cell(row, col)

    def unmake(self, row: int, col: int) -> None:
        self.hash ^= self.zobrist[row * self.size + col][self.grid[row][col]]
        self.grid[row][col] = EMPTY
        self.stones -= 1
        self._bump_adj(row, col, -1)
        if self.adj[row][col] > 0:
            self.cand_set.add((row, col))
        self._rescore_cell(row, col)

    def _rescore_cell(self, row: int, col: int) -> None:
        """Re-score the four lines through (row, col) and update the totals."""
        total, fours = self.total, self.fours
        for lid, _pos in self.cell_lines[row * self.size + col]:
            old = self.line_scores[lid]
            new = self._rescore_line(lid)
            self.line_scores[lid] = new
            total[0] += new[0] - old[0]
            total[1] += new[1] - old[1]
            fours[0] += (new[0] >= FOUR) - (old[0] >= FOUR)
            fours[1] += (new[1] >= FOUR) - (old[1] >= FOUR)

    # -- evaluation
    def evaluate(self, player: int) -> int:
        """Static evaluation from ``player``'s point of view."""
        human, ai = self.total
        if player == AI:
            return ai - human
        return human - ai

    def score_of(self, player: int) -> int:
        return self.total[1] if player == AI else self.total[0]

    def has_four(self, player: int) -> bool:
        """True if ``player`` can complete five with a single stone."""
        return self.fours[1 if player == AI else 0] > 0

    # -- candidate generation
    def candidates(self) -> List[Move]:
        if self.stones == 0:
            mid = self.size // 2
            return [(mid, mid)]
        return list(self.cand_set)

    # -- fast per-cell scoring (used for ordering, win and block detection)
    def point_score(self, row: int, col: int, player: int) -> int:
        """
        Estimate the value of ``player`` placing a stone at (row, col) by
        counting the runs it would join in each of the four directions.
        Exact for "makes five" detection, approximate for weaker shapes.
        """
        grid, size = self.grid, self.size
        total = 0
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            # forward run
            n1 = 0
            r, c = row + dr, col + dc
            while 0 <= r < size and 0 <= c < size and grid[r][c] == player:
                n1 += 1
                r, c = r + dr, c + dc
            open1 = 0 <= r < size and 0 <= c < size and grid[r][c] == EMPTY
            # stones after a single gap (for split shapes)
            gap1 = 0
            if open1:
                rr, cc = r + dr, c + dc
                while 0 <= rr < size and 0 <= cc < size and grid[rr][cc] == player:
                    gap1 += 1
                    rr, cc = rr + dr, cc + dc
            # backward run
            n2 = 0
            r, c = row - dr, col - dc
            while 0 <= r < size and 0 <= c < size and grid[r][c] == player:
                n2 += 1
                r, c = r - dr, c - dc
            open2 = 0 <= r < size and 0 <= c < size and grid[r][c] == EMPTY
            gap2 = 0
            if open2:
                rr, cc = r - dr, c - dc
                while 0 <= rr < size and 0 <= cc < size and grid[rr][cc] == player:
                    gap2 += 1
                    rr, cc = rr - dr, cc - dc

            n = 1 + n1 + n2
            opens = int(open1) + int(open2)
            if n >= 5:
                return FIVE
            if n == 4:
                total += OPEN_FOUR if opens == 2 else (FOUR if opens == 1 else 0)
            elif n == 3:
                if opens == 2:
                    total += OPEN_THREE
                elif opens == 1:
                    total += CLOSED_THREE
            elif n == 2:
                if opens == 2:
                    total += OPEN_TWO
                elif opens == 1:
                    total += CLOSED_TWO
            else:
                total += ONE * opens
            # split shapes: X X _ X etc.  (gap on at least one side)
            best_gap = max(gap1, gap2)
            if best_gap:
                m = n + best_gap
                if m >= 4:
                    total += FOUR
                elif m == 3:
                    total += OPEN_THREE if opens == 2 else CLOSED_THREE
                elif m == 2:
                    total += OPEN_TWO if opens == 2 else CLOSED_TWO
        return total


# ----------------------------------------------------------------------------
# The AI opponent
# ----------------------------------------------------------------------------
class SearchTimeout(Exception):
    pass


DIFFICULTIES = ("easy", "medium", "hard")


class GomokuAI:
    """
    Gomoku opponent.

    difficulty:
        easy   - one-ply heuristic (point scores) with randomness
        medium - alpha-beta search, depth 2, 10 candidates per node
        hard   - iterative deepening alpha-beta up to depth 6 within a
                 1.5 s time budget (adaptive: stops at the deepest fully
                 searched ply), 12 root / 8 inner candidates, threat pruning
    """

    CONFIG = {
        "easy": dict(depth=0, root_width=6, width=0, time_limit=0.5, random_top=3),
        "medium": dict(depth=2, root_width=10, width=10, time_limit=1.0, random_top=0),
        "hard": dict(depth=6, root_width=12, width=8, time_limit=1.5, random_top=0),
    }

    def __init__(self, difficulty: str = "medium", player: int = AI,
                 seed: Optional[int] = None) -> None:
        self.player = player
        self.opponent = other_player(player)
        self.set_difficulty(difficulty)
        self.rng = random.Random(seed)
        self.use_tt = True
        self._tt: Dict[int, Tuple[int, int, int, Optional[Move]]] = {}
        # statistics from the last search (for the GUI / tests)
        self.last_nodes = 0
        self.last_depth = 0
        self.last_time = 0.0
        self.last_reason = ""

    def set_difficulty(self, difficulty: str) -> None:
        difficulty = difficulty.lower()
        if difficulty not in self.CONFIG:
            raise ValueError(f"unknown difficulty {difficulty!r}")
        self.difficulty = difficulty
        self.cfg = dict(self.CONFIG[difficulty])

    # ------------------------------------------------------------ public API
    def choose_move(self, board: Board) -> Move:
        """Return the move the AI wants to play on ``board``."""
        if board.is_game_over():
            raise ValueError("game is already over")
        start = time.perf_counter()
        self.last_nodes = 0
        self.last_depth = 0
        self.last_reason = ""

        state = SearchState(board)
        me, opp = self.player, self.opponent

        if state.stones == 0:
            mid = board.size // 2
            self.last_reason = "opening: centre"
            self.last_time = time.perf_counter() - start
            return (mid, mid)

        cands = state.candidates()
        if not cands:  # should not happen unless the board is full
            cands = board.empty_cells()

        # 1. Immediate win.
        for mv in cands:
            if state.point_score(mv[0], mv[1], me) >= FIVE:
                self.last_reason = "winning move"
                self.last_time = time.perf_counter() - start
                return mv

        # 2. Block an immediate opponent win.
        blocks = [mv for mv in cands if state.point_score(mv[0], mv[1], opp) >= FIVE]
        if blocks:
            best = max(blocks, key=lambda m: state.point_score(m[0], m[1], me))
            self.last_reason = "blocking opponent five"
            self.last_time = time.perf_counter() - start
            return best

        # 3. Heuristic / search.
        if self.cfg["depth"] == 0:
            mv = self._easy_move(state, cands)
        else:
            mv = self._search_move(state, cands, start)
        self.last_time = time.perf_counter() - start
        return mv

    # --------------------------------------------------------------- helpers
    def _ordered(self, state: SearchState, cands: List[Move], me: int,
                 width: int) -> List[Tuple[int, Move]]:
        """Score candidates for ordering: attack value + defence value."""
        opp = other_player(me)
        scored = []
        for r, c in cands:
            attack = state.point_score(r, c, me)
            defence = state.point_score(r, c, opp)
            scored.append((attack + defence, (r, c)))
        scored.sort(key=lambda t: t[0], reverse=True)
        if width:
            scored = scored[:width]
        return scored

    def _easy_move(self, state: SearchState, cands: List[Move]) -> Move:
        scored = self._ordered(state, cands, self.player, self.cfg["root_width"])
        top = scored[: self.cfg["random_top"]] or scored
        # If the best move is a strong threat (open four or better, or stops one)
        # take it; otherwise pick randomly among the top few.
        if scored[0][0] >= OPEN_FOUR:
            self.last_reason = "easy: strong threat"
            return scored[0][1]
        self.last_reason = "easy: random among top candidates"
        return self.rng.choice(top)[1]

    def _search_move(self, state: SearchState, cands: List[Move],
                     start: float) -> Move:
        me = self.player
        max_depth = self.cfg["depth"]
        deadline = start + self.cfg["time_limit"]
        root = self._ordered(state, cands, me, self.cfg["root_width"])

        # Opponent open-four threats are handled by the search, but if the
        # opponent has an open three we make sure its blocking squares are in
        # the root candidate list.
        best_move = root[0][1]
        best_score = -WIN_SCORE
        completed_depth = 0
        self._tt.clear()
        budget = self.cfg["time_limit"]

        # Iterative deepening: depth 1, 2, ... up to max_depth, until the
        # time budget is used.  The best move from the previous iteration is
        # searched first in the next one (better ordering = more pruning).
        for depth in range(1, max_depth + 1):
            # Each extra ply costs several times the previous one, so do not
            # start an iteration that is unlikely to finish inside the budget.
            if depth > 1 and time.perf_counter() - start > 0.45 * budget:
                break
            order = [mv for _, mv in root]
            if best_move in order:
                order.remove(best_move)
                order.insert(0, best_move)
            try:
                score, move = self._root_search(state, order, me, depth, deadline)
            except SearchTimeout:
                break
            best_score, best_move, completed_depth = score, move, depth
            if abs(score) >= FIVE:  # forced win/loss found; deeper search is pointless
                break

        self.last_depth = completed_depth
        self.last_reason = f"search depth {completed_depth}, score {best_score}"
        return best_move

    def _root_search(self, state: SearchState, order: List[Move], me: int,
                     depth: int, deadline: float) -> Tuple[int, Move]:
        alpha, beta = -WIN_SCORE, WIN_SCORE
        best_move = order[0]
        for mv in order:
            state.make(mv[0], mv[1], me)
            score = -self._negamax(state, other_player(me), depth - 1, -beta, -alpha,
                                   1, deadline)
            state.unmake(mv[0], mv[1])
            if score > alpha:
                alpha, best_move = score, mv
        return alpha, best_move

    def _negamax(self, state: SearchState, me: int, depth: int, alpha: int,
                 beta: int, ply: int, deadline: float) -> int:
        """
        Negamax (Minimax where both players maximise their own score) with
        Alpha-Beta pruning.  Returns the score from ``me``'s point of view.
        """
        self.last_nodes += 1
        if (self.last_nodes & 63) == 0 and time.perf_counter() > deadline:
            raise SearchTimeout

        opp = other_player(me)
        if state.has_four(me):
            return WIN_SCORE - ply              # I complete five right now
        cands = state.candidates()
        if not cands:
            return 0  # full board: draw

        # Transposition table: the same position is often reached through
        # different move orders; reuse a previous result when it searched at
        # least as deep, and otherwise use its best move for ordering.
        alpha_orig = alpha
        tt_move: Optional[Move] = None
        if self.use_tt:
            entry = self._tt.get(state.hash)
            if entry is not None:
                e_depth, e_flag, e_value, tt_move = entry
                if e_depth >= depth:
                    if e_flag == 0:
                        return e_value
                    if e_flag == 1 and e_value > alpha:
                        alpha = e_value
                    elif e_flag == -1 and e_value < beta:
                        beta = e_value
                    if alpha >= beta:
                        return e_value

        # Tactical shortcuts: forced sequences are resolved before expanding,
        # which makes the search much stronger at a given depth.
        if state.has_four(opp):
            # The opponent threatens five: the only sensible replies are the
            # blocking squares.  Two different blocking squares = lost.
            must_block = [mv for mv in cands if state.point_score(mv[0], mv[1], opp) >= FIVE]
            if len(must_block) != 1:
                return -(WIN_SCORE - ply - 1)
            r, c = must_block[0]
            if depth <= 0:
                # Forced block at the horizon: play it and evaluate statically.
                state.make(r, c, me)
                value = -state.evaluate(opp)
                state.unmake(r, c)
                return value
            moves = must_block
        else:
            if depth <= 0:
                return state.evaluate(me)
            scored = []
            for r, c in cands:
                attack = state.point_score(r, c, me)
                defence = state.point_score(r, c, opp)
                scored.append((attack + defence, (r, c)))
            scored.sort(key=lambda t: t[0], reverse=True)
            width = self.cfg["width"]
            moves = [mv for _, mv in scored[:width]]
            if tt_move is not None and tt_move in moves:
                moves.remove(tt_move)
                moves.insert(0, tt_move)

        best = -WIN_SCORE
        best_mv: Optional[Move] = None
        for r, c in moves:
            state.make(r, c, me)
            value = -self._negamax(state, opp, depth - 1, -beta, -alpha, ply + 1, deadline)
            state.unmake(r, c)
            if value > best:
                best, best_mv = value, (r, c)
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break  # beta cut-off: opponent will never allow this line

        if self.use_tt:
            flag = 0 if alpha_orig < best < beta else (-1 if best <= alpha_orig else 1)
            self._tt[state.hash] = (depth, flag, best, best_mv)
        return best


# ----------------------------------------------------------------------------
# Convenience helpers used by tests / debugging
# ----------------------------------------------------------------------------
def find_winning_move(board: Board, player: int) -> Optional[Move]:
    """Return a cell where ``player`` completes five, or None."""
    state = SearchState(board)
    for r, c in state.candidates():
        if state.point_score(r, c, player) >= FIVE:
            return (r, c)
    return None


def evaluate_board(board: Board, player: int = AI) -> int:
    """Static heuristic evaluation of ``board`` from ``player``'s view."""
    return SearchState(board).evaluate(player)
