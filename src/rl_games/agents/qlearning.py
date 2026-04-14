import pickle
from collections import defaultdict
from pathlib import Path
from typing import Self

import gymnasium as gym
import numpy as np

_N_ACTIONS = 4

class QLearningAgent:
    def __init__(
        self,
        env_id: str,
        *,
        lr: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.999,
    ) -> None:
        self.env_id = env_id
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.training_episodes = 0

        self.q_table: dict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(_N_ACTIONS)
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def select_action(self, state: tuple, *, deterministic: bool = False) -> int:
        if not deterministic and np.random.random() < self.epsilon:
            return np.random.randint(_N_ACTIONS)
        return int(np.argmax(self.q_table[state]))

    def predict(self, obs, *, deterministic=True):
        return self.select_action(obs, deterministic=deterministic), None

    # ------------------------------------------------------------------
    # core RL
    # ------------------------------------------------------------------

    def _update(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        done: bool,
    ) -> None:
        best_next = 0.0 if done else float(np.max(self.q_table[next_state]))
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.lr * td_error

    def train(self, total_episodes: int = 10_000, log_interval: int = 100) -> list[float]:
        env = gym.make(self.env_id)
        rewards_history: list[float] = []

        for episode in range(1, total_episodes + 1):
            obs, _ = env.reset()
            state = obs
            total_reward = 0.0
            done = False

            # Environment loop
            while not done:
                # Select action
                action = self.select_action(state)
                # Take action
                next_obs, reward, terminated, truncated, _ = env.step(action)
                is_terminal = terminated or truncated
                # Update state
                next_state = next_obs
                # Update Q-table
                self._update(state, action, reward, next_state, is_terminal)
                # Update state
                state = next_state
                # Update total reward
                total_reward += reward
                done = is_terminal

            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            self.training_episodes += 1
            rewards_history.append(total_reward)

            if episode % log_interval == 0:
                avg = np.mean(rewards_history[-log_interval:])
                print(
                    f"Episode {episode}/{total_episodes} | "
                    f"Avg Reward: {avg:.2f} | "
                    f"Epsilon: {self.epsilon:.4f} | "
                    f"States visited: {len(self.q_table)}"
                )

        env.close()
        return rewards_history

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "q_table": dict(self.q_table),
            "epsilon": self.epsilon,
            "training_episodes": self.training_episodes,
            "env_id": self.env_id,
            "lr": self.lr,
            "gamma": self.gamma,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved Q-Learning agent to {path}")

    @classmethod
    def load(cls, path: Path) -> Self:
        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301

        agent = cls(
            env_id=data["env_id"],
            lr=data["lr"],
            gamma=data["gamma"],
            epsilon_start=data["epsilon"],
            epsilon_end=data["epsilon_end"],
            epsilon_decay=data["epsilon_decay"],
        )
        agent.q_table = defaultdict(
            lambda: np.zeros(_N_ACTIONS), data["q_table"]
        )
        agent.training_episodes = data["training_episodes"]
        return agent

    def info(self) -> str:
        return (
            f"Q-Learning agent for {self.env_id}\n"
            f"  Episodes trained : {self.training_episodes}\n"
            f"  States visited   : {len(self.q_table)}\n"
            f"  Epsilon          : {self.epsilon:.4f}\n"
            f"  LR / Gamma       : {self.lr} / {self.gamma}"
        )