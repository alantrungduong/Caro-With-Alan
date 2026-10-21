import random
import time

import pytest

from ai import (
    CLOSED_THREE,
    CLOSED_TWO,
    DIFFICULTIES,
    FIVE,
    FOUR,
    OPEN_FOUR,
    OPEN_THREE,
    OPEN_TWO,
    GomokuAI,
    SearchState,
    evaluate_board,
    find_winning_move,
    score_line,
)
from board import AI, BOARD_SIZE, EMPTY, HUMAN, Board

N = BOARD_SIZE


def setup(cells):
    b = Board()
    for r, c, p in cells:
        assert b.place(r, c, p), (r, c)
    return b


# ---------------------------------------------------------------- heuristic
@pytest.mark.parametrize("line,expected", [
    ("....11111......", FIVE),
    ("....1111.......", OPEN_FOUR),
    ("21111..........", FOUR),
    ("1111...........", FOUR),            # blocked by the board edge
    ("...111.1.......", FOUR),            # split four
    ("...11.11.......", FOUR),
    (".....111.......", OPEN_THREE),
    ("...11.1........", OPEN_THREE),      # split open three
    ("2111...........", CLOSED_THREE),
    ("2.111.2........", CLOSED_THREE),    # cannot become an open four
    ("......11.......", OPEN_TWO),
    ("211............", CLOSED_TWO),
    ("..111...111....", 2 * OPEN_THREE),
    ("...............", 0),
    ("...22222.......", 0),               # opponent stones only
    ("." * 30, 0),                        # full-length 30-cell line
    ("." * 24 + "111111", FIVE),          # six at the far end of a 30-cell line
])
def test_pattern_scores_for_human(line, expected):
    assert score_line(line)[0] == expected


def test_pattern_scores_are_symmetric_for_ai():
    assert score_line("....22222......") == (0, FIVE)
    assert score_line("....2222.......") == (0, OPEN_FOUR)
    assert score_line("..111.2222.....")[1] == OPEN_FOUR


def test_evaluation_prefers_stronger_ai_position():
    weak = setup([(15, 15, AI), (0, 0, HUMAN)])
    strong = setup([(15, 13, AI), (15, 14, AI), (15, 15, AI),
                    (0, 0, HUMAN), (0, 1, HUMAN), (0, 2, HUMAN)])
    strong.place(15, 16, AI)  # open four for AI
    assert evaluate_board(strong, AI) > evaluate_board(weak, AI)
    assert evaluate_board(strong, HUMAN) < 0


def board_from_state(state):
    b = Board()
    for r in range(N):
        for c in range(N):
            if state.grid[r][c] != EMPTY:
                b.grid[r, c] = state.grid[r][c]
                b.history.append((r, c, state.grid[r][c]))
    return b


def test_incremental_scores_match_full_rescore():
    """make/unmake must keep totals, four-counters, hash and candidates in sync."""
    rng = random.Random(42)
    state = SearchState(Board())
    moves = []
    for _ in range(60):
        r, c = rng.randrange(N), rng.randrange(N)
        if state.grid[r][c] != EMPTY:
            continue
        player = HUMAN if len(moves) % 2 == 0 else AI
        state.make(r, c, player)
        moves.append((r, c))
        fresh = SearchState(board_from_state(state))
        assert state.total == fresh.total
        assert state.fours == fresh.fours
        assert state.hash == fresh.hash
        assert set(state.candidates()) == set(fresh.candidates())
    for r, c in reversed(moves):
        state.unmake(r, c)
    assert state.total == [0, 0]
    assert state.fours == [0, 0]
    assert state.hash == 0
    assert state.stones == 0


def test_candidates_are_near_stones_only():
    state = SearchState(setup([(15, 15, HUMAN), (0, 0, AI)]))
    cands = set(state.candidates())
    assert (15, 16) in cands and (13, 13) in cands and (17, 17) in cands
    assert (1, 1) in cands and (2, 2) in cands
    assert (15, 18) not in cands and (10, 10) not in cands and (29, 29) not in cands
    assert len(cands) == 24 + 8   # 5x5 ring minus centre, plus 3x3 corner ring


# --------------------------------------------------------------- behaviour
@pytest.mark.parametrize("level", DIFFICULTIES)
def test_ai_plays_centre_on_empty_board(level):
    ai = GomokuAI(level, seed=0)
    assert ai.choose_move(Board()) == (15, 15)


@pytest.mark.parametrize("level", DIFFICULTIES)
def test_ai_takes_immediate_win(level):
    board = setup([(7, 3, AI), (7, 4, AI), (7, 5, AI), (7, 6, AI),
                   (0, 0, HUMAN), (1, 0, HUMAN), (2, 0, HUMAN), (3, 0, HUMAN)])
    ai = GomokuAI(level, seed=0)
    move = ai.choose_move(board)
    assert move in {(7, 2), (7, 7)}
    board.place(*move, AI)
    assert board.winner == AI


@pytest.mark.parametrize("level", DIFFICULTIES)
def test_ai_takes_split_win(level):
    board = setup([(4, 4, AI), (5, 5, AI), (7, 7, AI), (8, 8, AI),
                   (0, 0, HUMAN), (0, 1, HUMAN), (0, 2, HUMAN), (0, 3, HUMAN)])
    ai = GomokuAI(level, seed=0)
    assert ai.choose_move(board) == (6, 6)


@pytest.mark.parametrize("level", DIFFICULTIES)
def test_ai_takes_win_at_far_edge(level):
    board = setup([(29, 25, AI), (29, 26, AI), (29, 27, AI), (29, 28, AI),
                   (0, 0, HUMAN), (0, 1, HUMAN), (0, 2, HUMAN), (0, 3, HUMAN)])
    ai = GomokuAI(level, seed=0)
    assert ai.choose_move(board) in {(29, 24), (29, 29)}


@pytest.mark.parametrize("level", DIFFICULTIES)
def test_ai_blocks_immediate_opponent_win(level):
    board = setup([(7, 3, HUMAN), (7, 4, HUMAN), (7, 5, HUMAN), (7, 6, HUMAN),
                   (0, 0, AI), (1, 0, AI), (2, 0, AI)])
    ai = GomokuAI(level, seed=0)
    assert ai.choose_move(board) in {(7, 2), (7, 7)}


@pytest.mark.parametrize("level", DIFFICULTIES)
def test_ai_blocks_vertical_edge_threat(level):
    board = setup([(0, 5, HUMAN), (1, 5, HUMAN), (2, 5, HUMAN), (3, 5, HUMAN),
                   (20, 20, AI), (20, 21, AI), (20, 22, AI)])
    ai = GomokuAI(level, seed=0)
    assert ai.choose_move(board) == (4, 5)  # the only blocking square


@pytest.mark.parametrize("level", DIFFICULTIES)
def test_ai_blocks_threat_in_far_corner(level):
    board = setup([(29, 29, HUMAN), (28, 28, HUMAN), (27, 27, HUMAN), (26, 26, HUMAN),
                   (10, 10, AI), (10, 11, AI), (10, 12, AI)])
    ai = GomokuAI(level, seed=0)
    assert ai.choose_move(board) == (25, 25)


@pytest.mark.parametrize("level", DIFFICULTIES)
def test_ai_prefers_winning_over_blocking(level):
    board = setup([(7, 3, HUMAN), (7, 4, HUMAN), (7, 5, HUMAN), (7, 6, HUMAN),
                   (3, 3, AI), (4, 3, AI), (5, 3, AI), (6, 3, AI)])
    ai = GomokuAI(level, seed=0)
    assert ai.choose_move(board) == (2, 3)   # (7,3) is occupied, so the win is (2,3)


@pytest.mark.parametrize("level", ["medium", "hard"])
def test_ai_answers_open_three(level):
    """An open three must be blocked (or out-threatened) or the AI loses."""
    board = setup([(15, 13, HUMAN), (15, 14, HUMAN), (15, 15, HUMAN), (13, 13, AI), (13, 14, AI)])
    ai = GomokuAI(level, seed=0)
    assert ai.choose_move(board) in {(15, 12), (15, 16), (15, 11), (15, 17)}


def test_hard_finds_forced_win_with_double_threat():
    """AI has an open three and a three: the search should create a winning double threat."""
    board = setup([(15, 15, AI), (15, 16, AI), (15, 17, AI),       # open three in row 15
                   (12, 18, AI), (13, 18, AI),                     # two in column 18
                   (5, 5, HUMAN), (5, 6, HUMAN), (6, 5, HUMAN), (20, 20, HUMAN), (21, 21, HUMAN)])
    ai = GomokuAI("hard", seed=0)
    move = ai.choose_move(board)
    board.place(*move, AI)
    # After the move the AI must have at least a four (unstoppable or forcing).
    state = SearchState(board)
    assert state.has_four(AI) or state.score_of(AI) >= OPEN_FOUR


@pytest.mark.parametrize("level", DIFFICULTIES)
def test_ai_always_plays_legal_moves(level):
    rng = random.Random(123)
    ai = GomokuAI(level, seed=7)
    for _ in range(2):
        board = Board()
        player = HUMAN
        while not board.is_game_over():
            if player == HUMAN:
                move = rng.choice(board.empty_cells())
            else:
                move = ai.choose_move(board)
                assert board.is_valid_move(*move), (move, str(board))
            assert board.place(*move, player)
            player = AI if player == HUMAN else HUMAN
        # every level should beat a random mover
        assert board.winner == AI, str(board)


def test_ai_raises_when_game_over():
    board = setup([(0, 0, HUMAN), (0, 1, HUMAN), (0, 2, HUMAN), (0, 3, HUMAN), (0, 4, HUMAN)])
    with pytest.raises(ValueError):
        GomokuAI("medium").choose_move(board)


def test_find_winning_move_helper():
    board = setup([(2, 2, HUMAN), (3, 3, HUMAN), (4, 4, HUMAN), (5, 5, HUMAN)])
    assert find_winning_move(board, HUMAN) in {(1, 1), (6, 6)}
    assert find_winning_move(board, AI) is None


def test_easy_is_random_but_legal():
    board = setup([(15, 15, HUMAN)])
    moves = {GomokuAI("easy", seed=s).choose_move(board) for s in range(20)}
    assert len(moves) > 1
    for move in moves:
        assert board.is_valid_move(*move)


def test_invalid_difficulty_rejected():
    with pytest.raises(ValueError):
        GomokuAI("impossible")


# ------------------------------------------------------------- performance
def _selfplay_position(n_moves, seed=5):
    """A realistic position reached by Medium-vs-Medium self play, AI to move."""
    rng = random.Random(seed)
    board = Board()
    h = GomokuAI("medium", player=HUMAN, seed=seed)
    a = GomokuAI("medium", player=AI, seed=seed + 1)
    player = HUMAN
    while board.move_count < n_moves and not board.is_game_over():
        if player == HUMAN and rng.random() < 0.3:
            # occasional random human move keeps the game going
            move = rng.choice([m for m in SearchState(board).candidates()])
        else:
            move = (h if player == HUMAN else a).choose_move(board)
        board.place(*move, player)
        player = AI if player == HUMAN else HUMAN
    # Back off to a "quiet" position: nobody has a four or an open three, the
    # game is not over, and it is the AI side (O) to move, so the search
    # really has to think instead of answering a forced threat.
    while True:
        state = SearchState(board)
        strongest = max((max(s) for s in state.line_scores), default=0)
        if (not board.is_game_over() and player == AI
                and not state.has_four(HUMAN) and not state.has_four(AI)
                and strongest < OPEN_THREE):
            break
        removed = board.undo()
        assert removed is not None
        player = removed[2]
    assert not board.is_game_over()
    return board


@pytest.fixture(scope="module")
def positions():
    return {"early": _selfplay_position(6), "mid": _selfplay_position(30),
            "late": _selfplay_position(70)}


@pytest.mark.parametrize("stage", ["early", "mid", "late"])
@pytest.mark.parametrize("level,limit", [("easy", 0.5), ("medium", 1.0), ("hard", 2.2)])
def test_ai_response_time(positions, stage, level, limit):
    board = positions[stage]
    ai = GomokuAI(level, seed=0)
    start = time.perf_counter()
    move = ai.choose_move(board)
    elapsed = time.perf_counter() - start
    assert board.is_valid_move(*move)
    assert elapsed < limit, f"{level}/{stage} ({board.move_count} stones) took {elapsed:.2f}s"


def test_hard_searches_deeper_than_medium(positions):
    board = positions["mid"]
    medium, hard = GomokuAI("medium"), GomokuAI("hard")
    medium.choose_move(board)
    hard.choose_move(board)
    assert medium.last_depth == 2
    assert hard.last_depth >= 3
    assert hard.last_nodes > medium.last_nodes


def test_hard_respects_time_budget_on_dense_board():
    """Many stones -> many candidates; the time budget must still hold."""
    rng = random.Random(77)
    board = Board()
    player = HUMAN
    while board.move_count < 120:
        r, c = rng.randrange(N), rng.randrange(N)
        if board.place(r, c, player):
            if board.winner:
                board.undo()
                continue
            player = AI if player == HUMAN else HUMAN
    if player == HUMAN:
        board.undo()
    ai = GomokuAI("hard")
    start = time.perf_counter()
    move = ai.choose_move(board)
    assert board.is_valid_move(*move)
    assert time.perf_counter() - start < 2.2
