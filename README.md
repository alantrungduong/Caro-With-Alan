# Caro AI — Human vs Computer

A complete **Gomoku (Caro) game on a 30 × 30 board** written in Python.
You play **X** against a computer opponent (**O**) powered by classical
game-tree search: **Minimax with Alpha-Beta pruning** and a hand-written
pattern heuristic.

> **This is not a Deep Learning model.** Minimax / Alpha-Beta is a classical
> AI search algorithm from the 1950s–60s. There is no neural network, no
> training data and no learned weights: the computer "thinks" by looking a few
> moves ahead and scoring the resulting positions with fixed rules.

## Features

- 30 × 30 board (900 cells), five or more in a row wins (rows, columns, both diagonals).
- Human plays X and moves first; the AI replies automatically.
- Zoomable, pannable board: mouse-wheel zoom at the cursor, drag or arrow keys to pan, `F` to fit.
- Three difficulty levels: **Easy / Medium / Hard** (switchable at any time).
- Highlight of the last move and of the winning five (green cells + line).
- Draw detection, restart button (`R`), current-turn indicator, game-over message.
- The AI runs in a background thread, so the window stays responsive while it thinks.
- 136 automated tests (board rules, AI tactics, timing, game flow, GUI mapping and responsiveness).

## Technology stack

| Component | Choice |
|-----------|--------|
| Language  | Python 3.10+ (developed and tested on 3.13) |
| Graphics  | Pygame 2.x |
| Board storage | NumPy array (`board.py`) |
| AI | Minimax (negamax form) + Alpha-Beta pruning + iterative deepening + transposition table |
| Tests | pytest |

## Installation

```bash
# from the project folder
python -m pip install -r requirements.txt
```

## Run the game

```bash
python main.py                 # starts at Medium
python main.py --level hard    # easy | medium | hard
```

## Run the tests

```bash
python -m pytest -q
```

The GUI tests run headless (SDL "dummy" video driver), so no window opens.

## How to play

1. Click an empty cell to place your **X**. The computer answers with **O**.
2. Mouse wheel zooms in/out at the cursor; drag with any mouse button (or use
   the arrow keys / WASD) to pan; `F` or the **Fit** button shows the whole board.
3. The side panel shows whose turn it is, the difficulty buttons
   (`1` / `2` / `3` keys), the **Restart** button (`R`) and view controls.
4. First player to get five (or more) in a row wins. If the board fills up it
   is a draw. Press `R` or **Restart** to play again.

## Project structure

```
Lab06/
├── main.py          # entry point (python main.py)
├── board.py         # board state, move validation, win / draw detection
├── ai.py            # heuristic, candidate moves, Minimax + Alpha-Beta search
├── game.py          # turn controller, difficulty, restart, background AI thread
├── gui.py           # Pygame interface (viewport camera, panel, rendering)
├── requirements.txt
├── README.md
└── tests/
    ├── conftest.py
    ├── test_board.py  # rules: wins in all directions and at every edge, draw, invalid moves
    ├── test_ai.py     # heuristic patterns, win/block tactics, legality, timing
    ├── test_game.py   # turn cycle, restart, AI thread, statuses
    └── test_gui.py    # click/zoom/pan mapping, buttons, keys, responsiveness
```

## How the AI works

### Minimax

Minimax explores the game tree: "if I play here, what is the opponent's
best reply, and my best reply to that, …" down to a fixed depth. Leaf
positions are scored with the heuristic below; the AI picks the move whose
worst-case outcome (assuming the opponent plays best) is the best. The code
uses the *negamax* formulation, where each side maximises its own score and
the score flips sign when the turn changes (`GomokuAI._negamax` in `ai.py`).

### Alpha-Beta pruning

Alpha-Beta gives exactly the same answer as plain Minimax but skips branches
that cannot change the result. `alpha` is the best score the side to move is
already guaranteed; `beta` is the best the opponent can hold it to. When a
branch proves at least as good as `beta`, the opponent would never allow it,
so its remaining moves are not searched (a *cut-off*). With good move
ordering this roughly squares the number of positions that can be searched in
the same time.

Extra search techniques used on top of Alpha-Beta:

- **Candidate move generation**: only empty cells within distance 2 of an
  existing stone are considered (kept incrementally in a set), so the 900-cell
  board never has to be scanned during search.
- **Move ordering**: candidates are sorted by a fast "what would this stone
  create / what would it stop" score, and only the best 8–12 are expanded.
- **Immediate win / forced block**: a five-completing move is played at once;
  an opponent four is answered only with the blocking square (and two
  unblockable threats are recognised as a loss) before any deeper search.
- **Iterative deepening with a time budget**: Hard searches depth 1, 2, 3, …
  and stops when its budget is used, always returning the best move of the
  deepest fully searched ply (adaptive depth).
- **Transposition table**: positions reached by different move orders are
  hashed (Zobrist hashing) and their results reused.

### Heuristic evaluation

Every row, column and diagonal is turned into a string (`.` empty, `1` X,
`2` O) and matched against shape patterns, strongest first. Each shape has
a value; a line's score is the sum of the shapes it contains, and the board
score for a player is the sum over all 162 lines. The evaluation is
`own score − opponent score`, so the AI both attacks and defends.

| Shape | Example | Value |
|-------|---------|-------|
| Five | `XXXXX` | 10 000 000 |
| Open four | `_XXXX_` | 1 000 000 |
| Closed / split four | `OXXXX_`, `XXX_X`, `XX_XX` | 100 000 |
| Open three | `__XXX_`, `_XX_X_` | 10 000 |
| Closed three | `OXXX__`, `X_X_X` | 1 000 |
| Open two | `__XX__`, `_X_X_` | 500 |
| Closed two | `OXX___` | 50 |
| Single stone | `_X_` | 10 |

(`O` = opponent stone or board edge.) Because the search only changes four
lines per move, line scores are updated incrementally and memoised, which
keeps evaluation cheap.

### Difficulty levels

| Level | Method | Typical time per move |
|-------|--------|-----------------------|
| Easy | No look-ahead. Scores candidate cells with the fast shape heuristic and picks randomly among the top 3. Still takes an immediate win and blocks an immediate five. | < 0.05 s |
| Medium | Alpha-Beta search, depth 2, 10 candidates per node. | ≈ 0.05 s |
| Hard | Iterative-deepening Alpha-Beta up to depth 6 inside a 1.5 s budget, 12 root / 8 inner candidates, transposition table, forced-move pruning. | 0.5 – 1.6 s |

Timings measured on the development laptop (Python 3.13); see
`tests/test_ai.py::test_ai_response_time` for the enforced limits.

## Known limitations

- Standard free-style rules only: no Renju restrictions (no forbidden
  double-threes / overlines for X); six or more in a row also counts as a win.
- Hard is tactically strong but not unbeatable: the horizon is 4–6 plies, and
  the candidate limit (8–12 per node) can hide very quiet long-term plans.
- The AI time budget is wall-clock based, so on a slow machine Hard may search
  one ply less rather than take longer; on very old hardware it may slightly
  exceed 2 s.
- The window has a fixed size (1200 × 700); it is designed for 1366 × 768 or
  larger displays and is not resizable.
- Headless tests cover rendering calls and coordinate mapping, but the visual
  look was checked manually only on Windows 11.
