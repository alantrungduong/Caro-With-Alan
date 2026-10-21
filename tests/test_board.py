import pytest

from board import AI, BOARD_SIZE, EMPTY, HUMAN, Board, other_player

N = BOARD_SIZE  # 30


def fill(board, cells, player):
    for r, c in cells:
        assert board.place(r, c, player), (r, c)


def test_board_is_30_by_30_and_empty():
    b = Board()
    assert b.size == N == 30
    assert b.grid.shape == (30, 30)
    assert b.move_count == 0
    assert b.winner == EMPTY
    assert b.last_move is None
    assert not b.is_game_over()
    assert len(b.empty_cells()) == 900


def test_place_and_last_move():
    b = Board()
    assert b.place(15, 15, HUMAN)
    assert b.get(15, 15) == HUMAN
    assert b.last_move == (15, 15)
    assert b.move_count == 1


def test_occupied_cell_rejected():
    b = Board()
    assert b.place(3, 4, HUMAN)
    assert not b.place(3, 4, AI)
    assert not b.place(3, 4, HUMAN)
    assert b.get(3, 4) == HUMAN
    assert b.move_count == 1


@pytest.mark.parametrize("row,col", [(-1, 0), (0, -1), (30, 0), (0, 30), (15, 30), (99, 99)])
def test_out_of_bounds_rejected(row, col):
    b = Board()
    assert not b.is_valid_move(row, col)
    assert not b.place(row, col, HUMAN)
    assert b.move_count == 0


def test_corners_are_valid():
    b = Board()
    for r, c in ((0, 0), (0, 29), (29, 0), (29, 29)):
        assert b.place(r, c, HUMAN)
    assert b.move_count == 4


def test_invalid_player_raises():
    b = Board()
    with pytest.raises(ValueError):
        b.place(0, 0, 7)


def test_horizontal_win():
    b = Board()
    fill(b, [(5, 2), (5, 3), (5, 4), (5, 5)], HUMAN)
    assert b.winner == EMPTY
    assert b.place(5, 6, HUMAN)
    assert b.winner == HUMAN
    assert b.winning_line == [(5, 2), (5, 3), (5, 4), (5, 5), (5, 6)]
    assert b.is_game_over()


def test_vertical_win_completed_from_middle():
    b = Board()
    fill(b, [(1, 9), (2, 9), (3, 9), (5, 9)], AI)
    assert b.place(4, 9, AI)
    assert b.winner == AI
    assert b.winning_line == [(1, 9), (2, 9), (3, 9), (4, 9), (5, 9)]


def test_main_diagonal_win():
    b = Board()
    fill(b, [(10, 10), (11, 11), (12, 12), (13, 13)], HUMAN)
    assert b.place(14, 14, HUMAN)
    assert b.winner == HUMAN
    assert b.winning_line == [(10, 10), (11, 11), (12, 12), (13, 13), (14, 14)]


def test_anti_diagonal_win():
    b = Board()
    fill(b, [(20, 4), (19, 5), (18, 6), (17, 7)], AI)
    assert b.place(16, 8, AI)
    assert b.winner == AI
    assert set(b.winning_line) == {(20, 4), (19, 5), (18, 6), (17, 7), (16, 8)}


# ---- wins touching all four edges and the corners of the 30x30 board -------
EDGE_CASES = {
    "top edge horizontal": [(0, 10), (0, 11), (0, 12), (0, 13), (0, 14)],
    "bottom edge horizontal": [(29, 25), (29, 26), (29, 27), (29, 28), (29, 29)],
    "left edge vertical": [(12, 0), (13, 0), (14, 0), (15, 0), (16, 0)],
    "right edge vertical": [(0, 29), (1, 29), (2, 29), (3, 29), (4, 29)],
    "top-left corner diagonal": [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)],
    "bottom-right corner diagonal": [(25, 25), (26, 26), (27, 27), (28, 28), (29, 29)],
    "top-right corner anti-diagonal": [(0, 29), (1, 28), (2, 27), (3, 26), (4, 25)],
    "bottom-left corner anti-diagonal": [(29, 0), (28, 1), (27, 2), (26, 3), (25, 4)],
    "far right row end": [(15, 25), (15, 26), (15, 27), (15, 28), (15, 29)],
    "far bottom column end": [(25, 15), (26, 15), (27, 15), (28, 15), (29, 15)],
}


@pytest.mark.parametrize("name,cells", list(EDGE_CASES.items()), ids=list(EDGE_CASES))
def test_win_at_board_edges(name, cells):
    for last in range(5):  # whichever stone is placed last must trigger the win
        b = Board()
        order = cells[:last] + cells[last + 1:] + [cells[last]]
        fill(b, order[:-1], HUMAN)
        assert b.winner == EMPTY, name
        assert b.place(*order[-1], HUMAN), name
        assert b.winner == HUMAN, name
        assert set(b.winning_line) == set(cells), name


def test_no_wraparound_across_edges():
    """Four at the end of a row plus one at the start of the next row is NOT five."""
    b = Board()
    fill(b, [(3, 26), (3, 27), (3, 28), (3, 29), (4, 0)], HUMAN)
    assert b.winner == EMPTY
    fill(b, [(26, 29), (27, 29), (28, 29), (29, 29), (0, 29)], AI)
    assert b.winner == EMPTY


def test_six_in_a_row_also_wins():
    b = Board()
    fill(b, [(7, 0), (7, 1), (7, 2), (7, 4), (7, 5)], HUMAN)
    assert b.place(7, 3, HUMAN)
    assert b.winner == HUMAN
    assert len(b.winning_line) == 6


def test_four_in_a_row_is_not_a_win():
    b = Board()
    fill(b, [(7, 0), (7, 1), (7, 2), (7, 3)], HUMAN)
    assert b.winner == EMPTY
    assert not b.is_game_over()


def test_mixed_colours_do_not_win():
    b = Board()
    fill(b, [(7, 0), (7, 1), (7, 3), (7, 4)], HUMAN)
    assert b.place(7, 2, AI)
    assert b.winner == EMPTY


def test_no_moves_after_win():
    b = Board()
    fill(b, [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)], HUMAN)
    assert b.winner == HUMAN
    assert not b.place(10, 10, AI)
    assert not b.is_valid_move(10, 10)


def test_draw_detection():
    b = Board()
    # Fill the whole board with a pattern that never creates five in a row:
    # rows are XXOOXXOO.. and every row is inverted relative to the one above,
    # which keeps every row, column and diagonal run at length <= 2.
    for r in range(N):
        for c in range(N):
            player = HUMAN if ((c // 2) + r) % 2 == 0 else AI
            assert b.place(r, c, player), (r, c)
    assert b.winner == EMPTY, str(b)
    assert b.is_full()
    assert b.is_draw()
    assert b.is_game_over()
    assert b.find_winner() == (EMPTY, [])
    assert b.empty_cells() == []


def test_undo_restores_state():
    b = Board()
    fill(b, [(0, 0), (0, 1), (0, 2), (0, 3)], HUMAN)
    assert b.place(0, 4, HUMAN)
    assert b.winner == HUMAN
    assert b.undo() == (0, 4, HUMAN)
    assert b.winner == EMPTY
    assert b.winning_line == []
    assert b.get(0, 4) == EMPTY
    assert b.last_move == (0, 3)
    assert Board().undo() is None


def test_reset_clears_everything():
    b = Board()
    fill(b, [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)], HUMAN)
    b.reset()
    assert b.move_count == 0
    assert b.winner == EMPTY
    assert b.winning_line == []
    assert b.last_move is None
    assert len(b.empty_cells()) == 900


def test_copy_is_independent():
    b = Board()
    b.place(1, 1, HUMAN)
    c = b.copy()
    c.place(2, 2, AI)
    assert b.get(2, 2) == EMPTY
    assert c.get(2, 2) == AI
    assert c.move_count == 2 and b.move_count == 1


def test_other_player():
    assert other_player(HUMAN) == AI
    assert other_player(AI) == HUMAN
