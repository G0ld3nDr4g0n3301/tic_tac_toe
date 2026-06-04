from typing import Dict, List, Tuple
import numpy as np
from world.world import TicTacToeEnvironment
from model.model import TD0Agent
import json
import os


class AgentManager:
    def __init__(self, env: TicTacToeEnvironment) -> None:
        self.env: TicTacToeEnvironment = env

    def save_agent(self, agent: TD0Agent, filepath: str) -> None:
        """Save agent table to the file

        Args:
            agent: agent to save
            filepath: path to file
        """
        raw_table: Dict[str, float] = {str(k): float(v) for k, v in agent.table.items()}

        data = {"table": raw_table}

        with open(filepath, "w") as file:
            json.dump(data, file, indent=4)

        print(f"model saved to {filepath}")

    def load_agent(self, agent: TD0Agent, filepath: str) -> None:
        """Load agent from file

        Args:
            agent: agent to load table to
            filepath: path to file with table
        """
        if not os.path.exists(filepath):
            print(f"There is no such path as {filepath}, failed to load agent")
            return

        with open(filepath, "r") as file:
            data = json.load(file)

        agent.table.clear()
        for k, v in data["table"].items():
            agent.table[k] = v
        print(f"Model loaded from {filepath}")

    def train_self_play(
        self, games_count: int, lr: float = 0.2, epsilon: float = 0.2
    ) -> TD0Agent:
        """Train agent on itself

        Args:
            games_count: how many games to play
            lr: learning rate
            exploration: probability of making random move

        Returns:
            tranined agent
        """

        agentX: TD0Agent = TD0Agent(lr, epsilon, 1)
        agentO: TD0Agent = TD0Agent(lr, epsilon, 2)

        agentO.table = agentX.table  # Agent learns for both roles

        stats = {"X_wins": 0, "O_wins": 0, "Draws": 0}

        min_epsilon = 0.01
        min_lr = 0.01

        for game in range(games_count):
            progress: float = float(game) / games_count

            current_epsilon = max(min_epsilon, epsilon * (1 - progress))
            current_lr = max(min_lr, lr * (1 - progress))

            agentX.exploration_chance = current_epsilon
            agentO.exploration_chance = current_epsilon
            agentX.learning_rate = current_lr
            agentO.learning_rate = current_lr

            state: np.ndarray = self.env.reset()
            done = False

            last_state_after_move: Dict[int, np.ndarray] = {
                1: np.array([]),
                2: np.array([]),
            }

            while not done:
                current_role: int = self.env.turn
                current_agent: TD0Agent = agentX if current_role == 1 else agentO

                available_actions: List[Tuple[int, int]] = (
                    self.env.get_available_actions()
                )

                action: Tuple[int, int] = current_agent.choose_action(
                    state, available_actions
                )

                new_state: np.ndarray
                winner: int

                new_state, _, done, winner = self.env.step(action)

                if done:
                    other_agent: TD0Agent = (
                        agentO if current_agent == agentX else agentX
                    )
                    opponent_role = 1 if current_role == 2 else 2
                    if winner == 0:
                        current_agent.update_value(last_state_after_move[current_role], new_state, 0.5)
                        other_agent.update_value(
                            last_state_after_move[opponent_role], new_state, 0.5
                        )
                        stats["Draws"] += 1
                    else:
                        current_agent.update_value(state, new_state, 1.0)
                        other_agent.update_value(
                            last_state_after_move[opponent_role], new_state, 0.0
                        )
                        if winner == 1:
                            stats["X_wins"] += 1
                        else:
                            stats["O_wins"] += 1
                else:
                    if len(last_state_after_move[current_role]) != 0:
                        current_agent.update_value(
                            last_state_after_move[current_role], new_state, value=None
                        )

                    last_state_after_move[current_role] = new_state.copy()

                state = new_state

                if done and (game + 1) % (games_count / 10) == 0:
                    print(f"Games played: {game + 1}/{games_count}. epsilon: {current_epsilon}. lr: {current_lr}. Stats: {stats}")

        return agentX
