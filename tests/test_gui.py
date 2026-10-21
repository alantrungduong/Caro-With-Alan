"""
Headless GUI tests (SDL dummy video driver, set in conftest.py).

They verify that the Pygame interface initialises, renders, maps mouse
positions to cells accurately at every zoom/pan, reacts to the buttons and
keyboard shortcuts, and keeps rendering frames while the AI thinks.
They cannot verify how the window looks on a real screen.
"""
import time

import pytest

pygame = pytest.importorskip("pygame")

from board import AI, BOARD_SIZE, HUMAN
from game import STATUS_AI_THINKING, STATUS_HUMAN_TURN, Game
from gui import (
    MAX_CELL,
    MIN_CELL,
    VIEW_H,
    VIEW_W,
    VIEW_X,
    VIEW_Y,
    WINDOW_H,
    WINDOW_W,
    ZOOM_STEP,
    Camera,
    GomokuGUI,
)


@pytest.fixture
def gui():
    g = GomokuGUI(Game("easy"))
    yield g
    pygame.quit()


# ------------------------------------------------------------------ camera
def test_window_fits_1366x768():
    assert WINDOW_W <= 1366 and WINDOW_H <= 768


def test_fit_shows_whole_board():
    cam = Camera()
    r0, r1, c0, c1 = cam.visible_range()
    assert (r0, r1, c0, c1) == (0, BOARD_SIZE - 1, 0, BOARD_SIZE - 1)
    assert cam.is_visible(0, 0) and cam.is_visible(29, 29)


def _check_round_trip(cam):
    """Every visible cell centre must map back to the same cell."""
    r0, r1, c0, c1 = cam.visible_range()
    checked = 0
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            cx, cy = cam.cell_center(r, c)
            if VIEW_X <= cx < VIEW_X + VIEW_W and VIEW_Y <= cy < VIEW_Y + VIEW_H:
                assert cam.px_to_cell((cx, cy)) == (r, c), (r, c, cam.cell, cam.ox, cam.oy)
                # the corners of the cell rect also belong to the cell
                rect = cam.cell_rect(r, c)
                for px, py in ((rect.left, rect.top), (rect.right - 1, rect.bottom - 1)):
                    if VIEW_X <= px < VIEW_X + VIEW_W and VIEW_Y <= py < VIEW_Y + VIEW_H:
                        assert cam.px_to_cell((px, py)) == (r, c)
                checked += 1
    assert checked > 0


def test_mapping_at_fit_zoom():
    _check_round_trip(Camera())


@pytest.mark.parametrize("steps", [1, 3, 6, 10, -1, -3])
def test_mapping_after_zoom_at_cursor(steps):
    cam = Camera()
    anchor = (VIEW_X + 300, VIEW_Y + 200)
    for _ in range(abs(steps)):
        before_cell = cam.px_to_cell(anchor)
        larger_before = cam.board_px > max(VIEW_W, VIEW_H)
        cam.zoom(ZOOM_STEP if steps > 0 else 1 / ZOOM_STEP, anchor)
        larger_after = cam.board_px > max(VIEW_W, VIEW_H)
        if steps > 0 and larger_before and larger_after:
            # Once the board is larger than the viewport (no recentring),
            # zooming keeps the cell under the cursor in place.
            assert cam.px_to_cell(anchor) == before_cell
    assert MIN_CELL <= cam.cell <= MAX_CELL
    _check_round_trip(cam)


def test_zoom_anchor_is_exact_when_board_fills_viewport():
    cam = Camera()
    for _ in range(5):
        cam.zoom(ZOOM_STEP)                      # board now larger than the viewport
    assert cam.board_px > max(VIEW_W, VIEW_H)
    anchor = (VIEW_X + 123, VIEW_Y + 456)
    bx = (anchor[0] - VIEW_X - cam.ox) / cam.cell  # exact board coordinate under cursor
    by = (anchor[1] - VIEW_Y - cam.oy) / cam.cell
    cam.zoom(ZOOM_STEP, anchor)
    assert abs((anchor[0] - VIEW_X - cam.ox) / cam.cell - bx) < 1e-9
    assert abs((anchor[1] - VIEW_Y - cam.oy) / cam.cell - by) < 1e-9


@pytest.mark.parametrize("dx,dy", [(-500, -300), (300, 0), (0, 400), (-5000, 5000)])
def test_mapping_after_pan(dx, dy):
    cam = Camera()
    for _ in range(8):
        cam.zoom(ZOOM_STEP)
    cam.pan(dx, dy)
    # the board never leaves a gap when it is larger than the viewport
    assert cam.ox <= 0 and cam.oy <= 0
    assert cam.ox >= VIEW_W - cam.board_px and cam.oy >= VIEW_H - cam.board_px
    _check_round_trip(cam)


def test_zoom_limits():
    cam = Camera()
    for _ in range(100):
        cam.zoom(ZOOM_STEP)
    assert cam.cell == MAX_CELL
    for _ in range(100):
        cam.zoom(1 / ZOOM_STEP)
    assert cam.cell == MIN_CELL
    _check_round_trip(cam)


def test_outside_viewport_maps_to_none():
    cam = Camera()
    assert cam.px_to_cell((0, 0)) is None
    assert cam.px_to_cell((WINDOW_W - 1, WINDOW_H - 1)) is None
    # inside the viewport but outside the (centred, smaller) board
    assert cam.px_to_cell((VIEW_X + 2, VIEW_Y + 2)) is None


def test_ensure_visible_pans_to_cell():
    cam = Camera()
    for _ in range(8):
        cam.zoom(ZOOM_STEP, (VIEW_X + 10, VIEW_Y + 10))   # zoom in on the top-left corner
    assert not cam.is_visible(29, 29)
    cam.ensure_visible(29, 29)
    assert cam.is_visible(29, 29)
    _check_round_trip(cam)


# --------------------------------------------------------------- GUI logic
def test_render_frame(gui):
    gui.draw()  # must not raise
    gui.camera.zoom(ZOOM_STEP ** 5, (VIEW_X + 100, VIEW_Y + 100))
    gui.draw()
    gui.camera.zoom(ZOOM_STEP ** -8)
    gui.draw()


def test_click_places_x_at_every_zoom(gui):
    for zoom_steps in (0, 4, -2):
        gui.restart()
        for _ in range(abs(zoom_steps)):
            gui.camera.zoom(ZOOM_STEP if zoom_steps > 0 else 1 / ZOOM_STEP)
        pos = gui.cell_center(15, 15)
        assert gui.cell_at(pos) == (15, 15)
        gui.on_mouse_down(pos, 1)
        gui.on_mouse_up(pos, 1)
        assert gui.game.board.get(15, 15) == HUMAN, zoom_steps


def test_drag_pans_without_placing_stone(gui):
    for _ in range(6):
        gui.camera.zoom(ZOOM_STEP)
    ox, oy = gui.camera.ox, gui.camera.oy
    start = gui.cell_center(10, 10)
    gui.on_mouse_down(start, 1)
    gui.on_motion((start[0] - 60, start[1] - 40))
    gui.on_mouse_up((start[0] - 60, start[1] - 40), 1)
    assert gui.game.board.move_count == 0
    assert (gui.camera.ox, gui.camera.oy) != (ox, oy)


def test_wheel_event_zooms_at_cursor(gui):
    pos = gui.cell_center(15, 15)
    gui.handle_event(pygame.event.Event(pygame.MOUSEMOTION, pos=pos, rel=(0, 0), buttons=(0, 0, 0)))
    before = gui.camera.cell
    gui.handle_event(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=1, flipped=False))
    assert gui.camera.cell > before
    gui.handle_event(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=-1, flipped=False))
    assert gui.camera.cell == pytest.approx(before)
    # wheel outside the viewport does nothing
    gui.handle_event(pygame.event.Event(pygame.MOUSEMOTION, pos=(WINDOW_W - 5, 5), rel=(0, 0), buttons=(0, 0, 0)))
    gui.handle_event(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=1, flipped=False))
    assert gui.camera.cell == pytest.approx(before)


def test_click_outside_board_ignored(gui):
    gui.on_click((5, 5))
    gui.on_click((VIEW_X + 2, VIEW_Y + 2))
    assert gui.game.board.move_count == 0


def test_ai_replies_through_update_loop(gui):
    gui.on_click(gui.cell_center(15, 15))
    deadline = time.time() + 10
    while gui.game.board.move_count < 2 and time.time() < deadline:
        gui.update()
        gui.draw()
        time.sleep(0.005)
    assert gui.game.board.move_count == 2
    assert gui.game.status == STATUS_HUMAN_TURN


def test_gui_keeps_rendering_while_hard_ai_thinks(gui):
    """Responsiveness: frames keep coming while the AI thread searches."""
    gui.game.set_difficulty("hard")
    gui.game.ai.cfg["time_limit"] = 1.0
    gui.game.ai.cfg["depth"] = 8          # make sure it uses the whole budget
    board = gui.game.board
    # a quiet position with no forced move so the search really runs
    for r, c, p in ((15, 15, HUMAN), (14, 14, AI), (16, 16, HUMAN), (13, 13, AI)):
        board.place(r, c, p)
    gui.on_click(gui.cell_center(10, 20))
    gui.ai_delay_until = 0
    gui.update()                           # starts the AI thread
    assert gui.game.status == STATUS_AI_THINKING
    frames_before = gui.frames
    frame_times = []
    while gui.game.status == STATUS_AI_THINKING:
        t = time.perf_counter()
        gui.update()
        gui.draw()
        frame_times.append(time.perf_counter() - t)
    assert gui.frames - frames_before >= 20      # many frames rendered meanwhile
    assert max(frame_times) < 0.25               # no frame was blocked by the search
    assert gui.game.board.move_count == 6


def test_difficulty_restart_and_view_buttons(gui):
    hard = next(b for b in gui.diff_buttons if b.value == "hard")
    gui.on_click(hard.rect.center)
    assert gui.game.difficulty == "hard"
    gui.on_click(gui.cell_center(3, 3))
    assert gui.game.board.move_count == 1
    gui.on_click(gui.zoom_in_button.rect.center)
    assert gui.camera.cell > 22
    gui.on_click(gui.zoom_out_button.rect.center)
    gui.on_click(gui.fit_button.rect.center)
    assert gui.camera.is_visible(0, 0) and gui.camera.is_visible(29, 29)
    gui.on_click(gui.restart_button.rect.center)
    assert gui.game.board.move_count == 0
    assert gui.game.difficulty == "hard"


def test_keyboard_shortcuts(gui):
    key = lambda k: pygame.event.Event(pygame.KEYDOWN, key=k)
    gui.handle_event(key(pygame.K_3))
    assert gui.game.difficulty == "hard"
    gui.handle_event(key(pygame.K_1))
    assert gui.game.difficulty == "easy"
    before = gui.camera.cell
    gui.handle_event(key(pygame.K_EQUALS))
    assert gui.camera.cell > before
    gui.handle_event(key(pygame.K_RIGHT))
    gui.handle_event(key(pygame.K_f))
    assert gui.camera.is_visible(29, 29)
    gui.on_click(gui.cell_center(2, 2))
    gui.handle_event(key(pygame.K_r))
    assert gui.game.board.move_count == 0
    gui.handle_event(pygame.event.Event(pygame.QUIT))
    assert gui.running is False
