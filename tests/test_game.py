import time

from board import AI, EMPTY, HUMAN
from game import (
    STATUS_AI_THINKING,
    STATUS_AI_TURN,
    STATUS_AI_WON,
    STATUS_DRAW,
    STATUS_HUMAN_TURN,
    STATUS_HUMAN_WON,
    Game,
)


def test_initial_state():
    g = Game("easy")
    assert g.status == STATUS_HUMAN_TURN
    assert g.difficulty == "easy"
    assert g.board.move_count == 0
    assert g.board.size == 30


def test_human_then_ai_turn_cycle():
    g = Game("medium")
    assert g.human_move(15, 15)
    assert g.status == STATUS_AI_TURN
    assert not g.human_move(15, 16)          # not the human turn any more
    move = g.ai_move_sync()
    assert move is not None
    assert g.board.get(*move) == AI
    assert g.status == STATUS_HUMAN_TURN
    assert g.board.move_count == 2


def test_human_cannot_play_occupied_or_invalid_cell():
    g = Game("easy")
    assert g.human_move(0, 0)
    g.ai_move_sync()
    assert not g.human_move(0, 0)
    assert not g.human_move(30, 30)
    assert not g.human_move(-1, 5)
    assert g.status == STATUS_HUMAN_TURN
    assert g.board.move_count == 2


def test_background_ai_thread_applies_move():
    g = Game("medium")
    g.human_move(15, 15)
    assert g.start_ai_move()
    assert not g.start_ai_move()             # already running
    assert g.status in (STATUS_AI_THINKING, STATUS_AI_TURN)
    deadline = time.time() + 10
    move = None
    while move is None and time.time() < deadline:
        move = g.poll_ai()
        time.sleep(0.01)
    assert move is not None
    assert g.board.get(*move) == AI
    assert g.status == STATUS_HUMAN_TURN
    assert "nodes" in g.last_ai_info
    assert g.poll_ai() is None               # nothing pending any more


def test_restart_discards_pending_ai_result():
    g = Game("hard")
    g.human_move(15, 15)
    g.start_ai_move()
    g.restart()                              # restart while the AI is thinking
    time.sleep(1.0)
    assert g.poll_ai() is None              # stale result is ignored
    assert g.board.move_count == 0
    assert g.status == STATUS_HUMAN_TURN


def test_restart_resets_game_state():
    g = Game("easy")
    b = g.board
    for c in range(4):                       # set up a human win directly on the board
        b.place(0, c, HUMAN)
    assert g.human_move(0, 4)
    assert g.status == STATUS_HUMAN_WON
    assert len(b.winning_line) == 5
    g.set_difficulty("hard")
    g.restart()
    assert g.status == STATUS_HUMAN_TURN
    assert b.move_count == 0
    assert b.winner == EMPTY
    assert b.winning_line == []
    assert b.last_move is None
    assert g.last_ai_move is None
    assert g.last_ai_info == ""
    assert g.difficulty == "hard"            # difficulty is kept across restarts
    assert g.human_move(15, 15)              # and the game is playable again


def test_set_difficulty_changes_ai():
    g = Game("easy")
    g.set_difficulty("hard")
    assert g.ai.difficulty == "hard"
    assert g.ai.cfg["depth"] >= 4
    g.set_difficulty("medium")
    assert g.ai.cfg["depth"] == 2


def test_ai_win_status():
    g = Game("medium")
    b = g.board
    for c in range(4):
        b.place(10, c, AI)
    for r, c in ((0, 0), (0, 1), (0, 2), (1, 1)):
        b.place(r, c, HUMAN)
    g.current_player = AI
    assert g.status == STATUS_AI_TURN
    g.ai_move_sync()
    assert g.status == STATUS_AI_WON
    assert len(b.winning_line) >= 5
    assert not g.human_move(5, 5)


def test_draw_status():
    g = Game("easy")
    b = g.board
    for r in range(30):
        for c in range(30):
            b.place(r, c, HUMAN if ((c // 2) + r) % 2 == 0 else AI)
    assert g.status == STATUS_DRAW
    assert g.is_over()
