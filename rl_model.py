# =====================================================================
# EXAMPLE FILE - SERGIO (P2)
#
# Description: this file is the Q-Learning ENGINE of the Reinforcement
# Learning exercise: hyperparameters, environment step logic, epsilon-greedy
# action selection, training, evaluation without exploration and Q-values.
# Everything below is only an EXAMPLE implementation.
#
# What to edit:
#   - CONFIG: episodes, gamma, epsilon values, max steps, etc. (and explain
#     them in PARAM_NOTES).
#   - env_step(): the rules that decide what happens after each action.
#   - _train() / _evaluate(): the Q-Learning loop and the final evaluation.
#   - You may improve or rewrite anything, as long as it still uses
#     SGDRegressor with predict() and partial_fit().
#
# What NOT to change:
#   - The four actions: Up, Down, Left, Right.
#   - The keys returned by train_agent(): summary, evaluation, q_table and
#     curve_plot (the Application page depends on them).
#   - PARAM_NOTES items must keep the keys: name, key, value, note.
#   - The map and the rewards live in rl_environment.py (Jonathan's file).
#
# When you are done: DELETE this header block.
# =====================================================================
"""
Data with Roots - Reinforcement Learning engine (Activity 4, branch R2A3)

Q-Learning on a 10x10 grid world. Q(s, a) is estimated with scikit-learn's
SGDRegressor: predict() estimates Q-values and partial_fit() updates the model
from every observed transition (incremental learning).
"""

import io
import base64

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor

from rl_environment import GRID, ROWS, COLS, START, CELL_NAMES, REWARDS

N_STATES = ROWS * COLS

ACTIONS = ["Up", "Down", "Left", "Right"]
N_ACTIONS = len(ACTIONS)
ACTION_DELTAS = {"Up": (-1, 0), "Down": (1, 0), "Left": (0, -1), "Right": (0, 1)}

# ---------------------------------------------------------------------------
# TRAINING CONFIGURATION
# ---------------------------------------------------------------------------
CONFIG = {
    "episodes": 300,
    "gamma": 0.95,
    "epsilon_start": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.98,
    "max_steps": 100,
    "learning_rate": 0.1,
    "seed": 42,
}

PARAM_NOTES = [
    {"name": "Training episodes", "key": "episodes", "value": CONFIG["episodes"],
     "note": f"Number of complete attempts (from A until T or the step limit). {CONFIG['episodes']} episodes give enough experience for a 10x10 map."},
    {"name": "Discount factor (γ)", "key": "gamma", "value": CONFIG["gamma"],
     "note": "Close to 1 so the +100 reward at the goal still matters ~18 steps away (0.95^17 ≈ 0.42)."},
    {"name": "Initial epsilon (ε)", "key": "epsilon_start", "value": CONFIG["epsilon_start"],
     "note": "Starts at 1.0: the agent explores completely at random at the beginning."},
    {"name": "Minimum epsilon", "key": "epsilon_min", "value": CONFIG["epsilon_min"],
     "note": "Keeps 5% exploration so the agent never stops testing alternatives."},
    {"name": "Epsilon decay", "key": "epsilon_decay", "value": CONFIG["epsilon_decay"],
     "note": f"ε is multiplied by {CONFIG['epsilon_decay']} after each episode, shifting gradually from exploration to exploitation."},
    {"name": "Max steps per episode", "key": "max_steps", "value": CONFIG["max_steps"],
     "note": "Ends an episode that does not reach T, avoiding infinite wandering."},
    {"name": "Learning rate (eta0)", "key": "learning_rate", "value": CONFIG["learning_rate"],
     "note": "Step size of every SGDRegressor.partial_fit() update."},
    {"name": "Random seed", "key": "seed", "value": CONFIG["seed"],
     "note": "Makes the training reproducible: the same results appear every time."},
]

# ---------------------------------------------------------------------------
# ENVIRONMENT LOGIC
# ---------------------------------------------------------------------------
def env_step(state, action_idx):
    """
    Apply one action. Returns (next_state, cell_type, reward, done).
    - Cannot leave the grid or enter a wall (the agent stays in place).
    - Can enter a Danger Zone (with penalty).
    - Episode ends when the goal is reached (the step limit is handled by the caller).
    """
    dr, dc = ACTION_DELTAS[ACTIONS[action_idx]]
    nr, nc = state[0] + dr, state[1] + dc

    if not (0 <= nr < ROWS and 0 <= nc < COLS):
        return state, "Invalid move", REWARDS["invalid"], False

    ch = GRID[nr][nc]
    if ch == "#":
        return state, "Wall", REWARDS["wall"], False
    if ch == "D":
        return (nr, nc), "Danger Zone", REWARDS["danger"], False
    if ch == "T":
        return (nr, nc), "Goal", REWARDS["goal"], True
    return (nr, nc), CELL_NAMES[ch], REWARDS["normal"], False


# ---------------------------------------------------------------------------
# Q-FUNCTION APPROXIMATION (SGDRegressor)
# Features: one-hot vector over the 100 states x 4 actions = 400 features.
# ---------------------------------------------------------------------------
_EYE = np.eye(N_STATES * N_ACTIONS)


def _base(state):
    return (state[0] * COLS + state[1]) * N_ACTIONS


def q_values(model, state):
    """predict(): estimated Q(s, a) for the four actions of a state."""
    b = _base(state)
    return model.predict(_EYE[b:b + N_ACTIONS])


def choose_action(model, state, epsilon, rng):
    """Epsilon-greedy selection. Returns (action_index, 'explore' | 'exploit')."""
    if rng.random() < epsilon:
        return int(rng.integers(N_ACTIONS)), "explore"
    q = q_values(model, state)
    best = np.flatnonzero(np.isclose(q, q.max()))
    return int(rng.choice(best)), "exploit"


def _new_model():
    model = SGDRegressor(
        loss="squared_error", penalty=None, alpha=0.0, fit_intercept=False,
        learning_rate="constant", eta0=CONFIG["learning_rate"],
        random_state=CONFIG["seed"],
    )
    model.partial_fit(np.zeros((1, N_STATES * N_ACTIONS)), [0.0])  # initialise weights
    return model


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------
def _train():
    rng = np.random.default_rng(CONFIG["seed"])
    model = _new_model()
    epsilon = CONFIG["epsilon_start"]
    episode_rewards, successes = [], 0

    for _ in range(CONFIG["episodes"]):
        state, total = START, 0
        for _ in range(CONFIG["max_steps"]):
            action, _mode = choose_action(model, state, epsilon, rng)
            next_state, _cell, reward, done = env_step(state, action)

            # Q-learning target: r + gamma * max_a' Q(s', a')  (just r at the goal)
            target = reward
            if not done:
                target += CONFIG["gamma"] * float(np.max(q_values(model, next_state)))

            b = _base(state) + action
            model.partial_fit(_EYE[b:b + 1], [target])   # Q-value update

            total += reward
            state = next_state
            if done:
                successes += 1
                break

        episode_rewards.append(total)
        epsilon = max(CONFIG["epsilon_min"], epsilon * CONFIG["epsilon_decay"])

    return model, episode_rewards, successes, epsilon


# ---------------------------------------------------------------------------
# EVALUATION (greedy policy, no exploration)
# ---------------------------------------------------------------------------
def _evaluate(model):
    state, total = START, 0
    path, steps = [START], []
    seen = {START}
    goal_reached = False
    counts = {"danger": 0, "wall": 0, "invalid": 0}

    for i in range(1, CONFIG["max_steps"] + 1):
        action = int(np.argmax(q_values(model, state)))
        next_state, cell, reward, done = env_step(state, action)
        steps.append({"step": i, "state": state, "action": ACTIONS[action],
                      "next_state": next_state, "cell_type": cell, "reward": reward})
        total += reward
        if cell == "Danger Zone":
            counts["danger"] += 1
        elif cell == "Wall":
            counts["wall"] += 1
        elif cell == "Invalid move":
            counts["invalid"] += 1

        path.append(next_state)
        if done:
            goal_reached = True
            break
        if next_state in seen:      # policy is stuck in a loop
            break
        seen.add(next_state)
        state = next_state

    path_index = {}
    for n, cell in enumerate(path):
        path_index.setdefault(cell, n)

    return {
        "goal_reached": goal_reached, "moves": len(steps), "total_reward": total,
        "path": path, "path_index": path_index, "steps": steps,
        "danger_hits": counts["danger"], "wall_hits": counts["wall"],
        "invalid_hits": counts["invalid"],
    }


def _q_table(model):
    rows = []
    for r in range(ROWS):
        for c in range(COLS):
            if GRID[r][c] == "#":
                continue
            q = q_values(model, (r, c))
            rows.append({
                "state": (r, c), "cell": GRID[r][c],
                "Up": float(q[0]), "Down": float(q[1]),
                "Left": float(q[2]), "Right": float(q[3]),
                "best": ACTIONS[int(np.argmax(q))],
            })
    return rows


def _curve_plot(episode_rewards):
    window = 25
    rewards = np.array(episode_rewards, dtype=float)
    moving = np.convolve(rewards, np.ones(window) / window, mode="valid")

    fig, ax = plt.subplots(figsize=(8, 4), dpi=110)
    ax.plot(rewards, color="#adb5bd", linewidth=0.8, label="Reward per episode")
    ax.plot(range(window - 1, len(rewards)), moving, color="#3B6E5C", linewidth=2.2,
            label=f"Moving average ({window} episodes)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title("Learning curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# PUBLIC API (called by app.py)
# ---------------------------------------------------------------------------
def train_agent():
    """Train the agent, evaluate the learned policy and return everything the page needs."""
    model, episode_rewards, successes, final_epsilon = _train()
    episodes = CONFIG["episodes"]
    return {
        "summary": {
            "episodes": episodes,
            "successes": successes,
            "success_pct": round(100 * successes / episodes, 2),
            "avg_reward": round(float(np.mean(episode_rewards)), 2),
            "final_epsilon": round(final_epsilon, 4),
        },
        "evaluation": _evaluate(model),
        "q_table": _q_table(model),
        "curve_plot": _curve_plot(episode_rewards),
    }