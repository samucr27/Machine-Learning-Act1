"""
Data with Roots - Reinforcement Learning environment (Activity 4, branch R2A3)
Author: Jonathan Alejandro Yacuma Rivera

Defines the 10x10 grid world used by the Q-Learning agent (rl_model.py):
    - GRID:          the map, with A, T, o, # and D cells.
    - ENV_LEGEND:    meaning and number of cells of each character.
    - REWARDS:       reward or penalty of every type of event.
    - REWARD_TABLE:  the reward system explained, shown on the Application page.

Map design:
    The agent starts at A (0, 0) in the top-left corner and the target T is at (9, 9)
    in the bottom-right corner. The 20 walls (#) block the direct routes and create
    corridors, and the 10 Danger Zones (D) are placed on the most tempting shortcuts,
    next to corridor entrances and around the target. Of the 219 possible routes of
    18 moves (the minimum distance), only ONE avoids every Danger Zone, so the agent
    must really learn the map: any other short route costs at least one penalty.
"""

from collections import Counter

# ---------------------------------------------------------------------------
# ENVIRONMENT (10 x 10)
#   A = agent start | T = target | o = available path | # = wall | D = danger zone
# ---------------------------------------------------------------------------
GRID = [
    "Aooo#ooooo",
    "o##o#o##Do",
    "ooDoooo#oo",
    "#oo#D#ooo#",
    "ooDooo#Doo",
    "o#oo#oooDo",
    "oooDoo#ooo",
    "o#ooo#oD#o",
    "oo#Doooooo",
    "oooo#oooDT",
]
ROWS, COLS = len(GRID), len(GRID[0])

CELL_NAMES = {"A": "Start", "T": "Goal", "o": "Path", "#": "Wall", "D": "Danger Zone"}

REQUIRED_COUNTS = {"A": 1, "T": 1, "o": 68, "#": 20, "D": 10}


def _find(char):
    """Return the (row, column) of the first cell that contains char."""
    for r, row in enumerate(GRID):
        for c, ch in enumerate(row):
            if ch == char:
                return (r, c)
    raise ValueError(f"Character {char!r} not found in GRID")


def _goal_is_reachable():
    """Breadth-first search from A to T moving only through cells that are not walls."""
    start, goal = _find("A"), _find("T")
    pending, visited = [start], {start}
    while pending:
        r, c = pending.pop(0)
        if (r, c) == goal:
            return True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS and GRID[nr][nc] != "#" and (nr, nc) not in visited:
                visited.add((nr, nc))
                pending.append((nr, nc))
    return False


START = _find("A")
GOAL = _find("T")

# Safety checks: the activity requires a 10x10 map with exactly this distribution of
# cells, and the target must be reachable from the start.
assert ROWS == 10 and all(len(row) == 10 for row in GRID), "The environment must be 10x10"
assert dict(Counter("".join(GRID))) == REQUIRED_COUNTS, "Wrong cell distribution in GRID"
assert _goal_is_reachable(), "There is no path from A to T"

ENV_LEGEND = [
    {"char": "A", "meaning": "Agent start position", "count": REQUIRED_COUNTS["A"]},
    {"char": "T", "meaning": "Target (goal)", "count": REQUIRED_COUNTS["T"]},
    {"char": "o", "meaning": "Available path", "count": REQUIRED_COUNTS["o"]},
    {"char": "#", "meaning": "Wall / obstacle (cannot be crossed)", "count": REQUIRED_COUNTS["#"]},
    {"char": "D", "meaning": "Danger Zone (allowed, but penalized)", "count": REQUIRED_COUNTS["D"]},
]

# ---------------------------------------------------------------------------
# REWARD SYSTEM
# ---------------------------------------------------------------------------
REWARDS = {
    "normal": -1,    # valid move to a normal cell (start or path)
    "invalid": -4,   # tries to leave the grid
    "wall": -6,      # tries to enter a wall (#)
    "danger": -15,   # enters a Danger Zone (D)
    "goal": 100,     # reaches the target (T)
}

REWARD_TABLE = [
    {"event": "Moving to a valid normal position", "reward": REWARDS["normal"],
     "why": "Each step has a small cost, so among the routes that reach T the agent prefers the "
            "one with fewer moves."},
    {"event": "Attempting an invalid movement (outside the grid)", "reward": REWARDS["invalid"],
     "why": "The agent stays in the same cell and loses 4 points: it wastes a step and pays more "
            "than a normal move, so it learns not to push against the borders of the map."},
    {"event": "Hitting a wall (#)", "reward": REWARDS["wall"],
     "why": "Also leaves the agent in place, with a higher cost than an invalid move because walls "
            "are inside the map and block the corridors the agent must learn to go around."},
    {"event": "Entering a Danger Zone (D)", "reward": REWARDS["danger"],
     "why": "The agent can cross a D, but it costs as much as 15 normal moves. Since the safe route "
            "has the same length as the shortcuts through D, crossing a Danger Zone is never worth it."},
    {"event": "Reaching the goal (T)", "reward": REWARDS["goal"],
     "why": "Large positive reward that ends the episode. It is much bigger than all the penalties "
            "of a route, so reaching T is always the agent's main objective."},
]


def get_grid():
    """Return the map as a list of lists of characters (used by the templates)."""
    return [list(row) for row in GRID]