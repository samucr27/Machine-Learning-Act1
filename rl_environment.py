# =====================================================================
# EXAMPLE FILE - JONATHAN (P3)
#
# Description: this file defines the ENVIRONMENT of the Reinforcement
# Learning exercise: the 10x10 map and the reward system. The map and the
# reward values below are only an EXAMPLE.
#
# What to edit:
#   - GRID: redesign the map your own way.
#   - REWARDS: choose your own reward / penalty values.
#   - REWARD_TABLE: rewrite the "why" column with your own explanation.
#
# What NOT to change:
#   - The map must be 10x10 with exactly 1 A, 1 T, 68 o, 20 #, 10 D, and
#     at least one path from A to T (the assert statements will warn you).
#   - Keep the five keys of REWARDS: normal, invalid, wall, danger, goal.
#   - Keep the names GRID, START, GOAL, ENV_LEGEND, REWARDS, REWARD_TABLE
#     and get_grid(), because other files import them.
#
# When you are done: DELETE this header block.
# =====================================================================
"""Data with Roots - Reinforcement Learning environment (10x10 grid)."""

from collections import Counter

# ---------------------------------------------------------------------------
# ENVIRONMENT (10 x 10)
#   A = agent start | T = target | o = available path | # = wall | D = danger zone
# ---------------------------------------------------------------------------
GRID = [
    "Aoo#oooo#o",
    "o#oDo#oooo",
    "o#ooo#Do#o",
    "oDo#ooo#oo",
    "#Do#DooooD",
    "ooooo##ooo",
    "o#Dooooo#o",
    "ooo#o#Dooo",
    "#oooooo#Do",
    "ooDo#ooooT",
]
ROWS, COLS = len(GRID), len(GRID[0])

CELL_NAMES = {"A": "Start", "T": "Goal", "o": "Path", "#": "Wall", "D": "Danger Zone"}

REQUIRED_COUNTS = {"A": 1, "T": 1, "o": 68, "#": 20, "D": 10}


def _find(char):
    for r, row in enumerate(GRID):
        for c, ch in enumerate(row):
            if ch == char:
                return (r, c)
    raise ValueError(f"Character {char!r} not found in GRID")


START = _find("A")
GOAL = _find("T")

# Safety check: the activity requires exactly 100 cells with this distribution.
assert ROWS == 10 and COLS == 10, "The environment must be 10x10"
assert dict(Counter("".join(GRID))) == REQUIRED_COUNTS, "Wrong cell distribution in GRID"

ENV_LEGEND = [
    {"char": "A", "meaning": "Agent start position", "count": REQUIRED_COUNTS["A"]},
    {"char": "T", "meaning": "Target (goal)", "count": REQUIRED_COUNTS["T"]},
    {"char": "o", "meaning": "Available path", "count": REQUIRED_COUNTS["o"]},
    {"char": "#", "meaning": "Wall / obstacle", "count": REQUIRED_COUNTS["#"]},
    {"char": "D", "meaning": "Danger Zone (allowed, but penalized)", "count": REQUIRED_COUNTS["D"]},
]

# ---------------------------------------------------------------------------
# REWARD SYSTEM
# ---------------------------------------------------------------------------
REWARDS = {
    "normal": -1,    # valid move to a normal cell (start or path)
    "invalid": -5,   # tries to leave the grid
    "wall": -10,     # tries to enter a wall (#)
    "danger": -20,   # enters a Danger Zone (D)
    "goal": 100,     # reaches the target (T)
}

REWARD_TABLE = [
    {"event": "Moving to a valid normal position", "reward": REWARDS["normal"],
     "why": "Small cost per step, so shorter routes earn a higher return."},
    {"event": "Attempting an invalid movement (outside the grid)", "reward": REWARDS["invalid"],
     "why": "Discourages leaving the map; the agent stays in the same cell."},
    {"event": "Hitting a wall (#)", "reward": REWARDS["wall"],
     "why": "Stronger penalty than an invalid move; the agent stays in the same cell."},
    {"event": "Entering a Danger Zone (D)", "reward": REWARDS["danger"],
     "why": "Allowed but expensive, so the agent learns to walk around it."},
    {"event": "Reaching the goal (T)", "reward": REWARDS["goal"],
     "why": "Large positive reward that ends the episode and anchors the value of the path."},
]


def get_grid():
    """Return the map as a list of lists of characters."""
    return [list(row) for row in GRID]