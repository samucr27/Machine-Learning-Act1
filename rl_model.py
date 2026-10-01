"""
Data with Roots - Reinforcement Learning engine (Activity 4, branch R2A3)
Author: Sergio Steeven Moreno Forero

Q-Learning on the 10x10 grid world defined in rl_environment.py.

Structure of this module:
    1. CONFIG / PARAM_NOTES  -> training hyperparameters and their explanation.
    2. GridEnvironment       -> rules of the environment: what happens after each action.
    3. QFunction             -> Q(s, a) approximated with scikit-learn's SGDRegressor
                                (predict() estimates Q-values, partial_fit() updates them).
    4. choose_action()       -> epsilon-greedy strategy (exploration vs. exploitation).
    5. train()               -> Q-Learning training loop over many episodes.
    6. evaluate()            -> greedy run of the learned policy, without exploration.
    7. get_q_table()         -> learned Q-values of the four actions in every valid state.
    8. train_agent()         -> public entry point used by app.py (POST "Train Agent").

Learning flow implemented in train():
    Environment -> Initial State -> Action Selection (epsilon-greedy: explore / exploit)
    -> Environment Interaction -> Reward -> Next State -> Q-value Update (partial_fit)
    -> Next Action -> ... -> Goal or step limit (end of episode)
"""

import io
import base64

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor

from rl_environment import GRID, ROWS, COLS, START, CELL_NAMES, REWARDS

# ---------------------------------------------------------------------------
# ACTIONS
# ---------------------------------------------------------------------------
ACTIONS = ["Up", "Down", "Left", "Right"]
N_ACTIONS = len(ACTIONS)
ACTION_DELTAS = {"Up": (-1, 0), "Down": (1, 0), "Left": (0, -1), "Right": (0, 1)}

N_STATES = ROWS * COLS
N_FEATURES = N_STATES * N_ACTIONS

# ---------------------------------------------------------------------------
# 1. TRAINING CONFIGURATION
# ---------------------------------------------------------------------------
CONFIG = {
    "episodes": 300,
    "gamma": 0.95,
    "epsilon_start": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.985,
    "max_steps": 100,
    "learning_rate": 0.1,
    "seed": 42,
}

PARAM_NOTES = [
    {"name": "Training episodes", "key": "episodes", "value": CONFIG["episodes"],
     "note": f"Number of complete attempts, each one from A until T or the step limit. "
             f"{CONFIG['episodes']} episodes give the agent enough experience to explore the 10x10 map "
             f"and then refine the best route."},
    {"name": "Discount factor (γ)", "key": "gamma", "value": CONFIG["gamma"],
     "note": "Weight of future rewards. A value close to 1 keeps the +100 reward of T relevant even "
             "18 steps away (0.95^17 ≈ 0.42), so the agent plans the whole route instead of only the next move."},
    {"name": "Initial epsilon (ε)", "key": "epsilon_start", "value": CONFIG["epsilon_start"],
     "note": "Probability of exploring. It starts at 1.0, so the first episodes are completely random "
             "and the agent discovers walls, Danger Zones and the target."},
    {"name": "Minimum epsilon", "key": "epsilon_min", "value": CONFIG["epsilon_min"],
     "note": "Lower limit of ε. The agent always keeps 5% of random actions, so it never stops "
             "testing alternatives during training."},
    {"name": "Epsilon decay", "key": "epsilon_decay", "value": CONFIG["epsilon_decay"],
     "note": f"After each episode ε is multiplied by {CONFIG['epsilon_decay']}, so it reaches the minimum "
             f"around episode 200 and the agent shifts gradually from exploration to exploitation."},
    {"name": "Max steps per episode", "key": "max_steps", "value": CONFIG["max_steps"],
     "note": "Ends an episode that does not reach T. The shortest route needs 18 moves, so 100 steps "
             "leave room to explore without wandering forever."},
    {"name": "Learning rate (eta0)", "key": "learning_rate", "value": CONFIG["learning_rate"],
     "note": "Step size of every SGDRegressor.partial_fit() update: how much each new experience "
             "moves the current Q-value estimate toward the Q-Learning target."},
    {"name": "Random seed", "key": "seed", "value": CONFIG["seed"],
     "note": "Fixes the random choices of the agent, so the same training results are obtained every time."},
]


# ---------------------------------------------------------------------------
# 2. ENVIRONMENT
# ---------------------------------------------------------------------------
class GridEnvironment:
    """
    10x10 grid world. The state is the agent's position (row, column).

    Rules applied after every action:
      - Leaving the grid  -> invalid move: the agent stays in place (REWARDS["invalid"]).
      - Entering a wall # -> collision:    the agent stays in place (REWARDS["wall"]).
      - Entering a D      -> allowed, but penalized (REWARDS["danger"]).
      - Entering T        -> goal reached, the episode ends (REWARDS["goal"]).
      - Any other cell    -> normal valid move (REWARDS["normal"]).
    The episode also ends when the maximum number of steps is reached.
    """

    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.state = START
        self.steps_taken = 0

    def reset(self):
        """Start a new episode at A."""
        self.state = START
        self.steps_taken = 0
        return self.state

    def step(self, action_idx):
        """
        Apply one action and return a transition with everything that happened:
        state, action, next_state, cell_type, reward, done and the end reason.
        """
        state = self.state
        action = ACTIONS[action_idx]
        dr, dc = ACTION_DELTAS[action]
        nr, nc = state[0] + dr, state[1] + dc

        if not (0 <= nr < ROWS and 0 <= nc < COLS):
            next_state, cell_type, reward, reached_goal = state, "Invalid move", REWARDS["invalid"], False
        else:
            ch = GRID[nr][nc]
            if ch == "#":
                next_state, cell_type, reward, reached_goal = state, "Wall", REWARDS["wall"], False
            elif ch == "D":
                next_state, cell_type, reward, reached_goal = (nr, nc), "Danger Zone", REWARDS["danger"], False
            elif ch == "T":
                next_state, cell_type, reward, reached_goal = (nr, nc), "Goal", REWARDS["goal"], True
            else:
                next_state, cell_type, reward, reached_goal = (nr, nc), CELL_NAMES[ch], REWARDS["normal"], False

        self.steps_taken += 1
        self.state = next_state
        out_of_steps = self.steps_taken >= self.max_steps
        done = reached_goal or out_of_steps

        return {
            "state": state,
            "action": action,
            "action_idx": action_idx,
            "next_state": next_state,
            "cell_type": cell_type,
            "reward": reward,
            "done": done,
            "end_reason": "Goal reached" if reached_goal else ("Step limit" if out_of_steps else ""),
            "reached_goal": reached_goal,
        }


# ---------------------------------------------------------------------------
# 3. Q-FUNCTION APPROXIMATION WITH SGDRegressor
# ---------------------------------------------------------------------------
class QFunction:
    """
    Q(s, a) estimated with a linear SGDRegressor.

    Features: one-hot vector of the (state, action) pair, of size 100 states x 4 actions = 400.
    Each pair has its own weight, so Q(s, a) = weight of that pair, and a partial_fit()
    on one pair only changes that Q-value (the same behavior as a Q-table, but learned
    incrementally by the regression model).
    """

    def __init__(self, learning_rate, seed):
        self.model = SGDRegressor(
            loss="squared_error",
            penalty=None,
            fit_intercept=False,
            learning_rate="constant",
            eta0=learning_rate,
            random_state=seed,
        )
        self._identity = np.eye(N_FEATURES)
        # First call to partial_fit() creates the weights (all Q-values start at 0).
        self.model.partial_fit(np.zeros((1, N_FEATURES)), [0.0])

    @staticmethod
    def _index(state, action_idx=0):
        return (state[0] * COLS + state[1]) * N_ACTIONS + action_idx

    def predict(self, state):
        """predict(): estimated Q-values of the four actions in a state."""
        i = self._index(state)
        return self.model.predict(self._identity[i:i + N_ACTIONS])

    def update(self, state, action_idx, target):
        """partial_fit(): move Q(state, action) toward the Q-Learning target."""
        i = self._index(state, action_idx)
        self.model.partial_fit(self._identity[i:i + 1], [target])


# ---------------------------------------------------------------------------
# 4. EPSILON-GREEDY ACTION SELECTION
# ---------------------------------------------------------------------------
def choose_action(q_state, epsilon, rng):
    """
    Exploration:  with probability epsilon, a random action.
    Exploitation: otherwise, the action with the highest Q-value (ties broken at random).
    Returns (action_index, "explore" | "exploit").
    """
    if rng.random() < epsilon:
        return int(rng.integers(N_ACTIONS)), "explore"
    best = np.flatnonzero(np.isclose(q_state, q_state.max()))
    return int(rng.choice(best)), "exploit"
