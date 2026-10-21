"""
main.py - Entry point for Caro AI (Gomoku, Human vs Computer, 30 x 30 board).

Run:
    python main.py                 # start at Medium difficulty
    python main.py --level hard    # start at Easy / Medium / Hard
"""

from __future__ import annotations

import argparse
import sys

from ai import DIFFICULTIES


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Caro AI - Human vs Computer (Gomoku)")
    parser.add_argument("--level", choices=DIFFICULTIES, default="medium",
                        help="starting AI difficulty (default: medium)")
    args = parser.parse_args(argv)

    from gui import run  # imported here so `--help` works without a display

    run(args.level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
