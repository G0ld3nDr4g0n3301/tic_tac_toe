from typing import List, Tuple, Optional
from collections import defaultdict
import numpy as np
from random import random


class TD0Agent:
    def __init__(
        self, learning_rate: float = 0.2, exploration_chance: float = 0.2, role: int = 1
    ) -> None:
        """Init parameters

        Args:
            learning_rate: how much the value of previous move will change
            exploration_chance: probability of making random move
            role: 1 if playing Xs, 2 if playing Os.
        """

        self.learning_rate: float = learning_rate
        self.exploration_chance: float = exploration_chance
        self.role = role
        self.table: defaultdict[str, float] = defaultdict(lambda: 0.5)

    def get_string_representation(self, state: np.ndarray) -> str:
        """Get string representation of the board state

        Args:
            board: a field matrix

        Returns:
            a string representation
        """

        return "".join(state.astype(str).flatten())

    def get_canonical_form(self, state: np.ndarray) -> str:
        """Get a canonical form

        Args:
            state: matrix state of the board

        Returns:
            the lexigraphically least of string representations of the symmetries of the state
        """

        symmetries: List[np.ndarray] = []

        for i in range(4):
            rotated = np.rot90(state, i)
            symmetries.append(rotated)
            symmetries.append(np.fliplr(rotated))

        representations: List[str] = sorted(
            [self.get_string_representation(sym) for sym in symmetries]
        )

        return representations[0]

    def get_filtered_state(self, state: np.ndarray) -> np.ndarray:
        """Get a state where agent plays always with Xs

        Args:
            state: old state to transform

        Returns:
            transformed state
        """

        if self.role == 1:
            return state

        return np.where(state == 2, 1, np.where(state == 1, 2, 0))

    def choose_action(
        self, state: np.ndarray, available_actions: List[Tuple[int, int]]
    ) -> Tuple[int, int]:
        """Pick an action

        Args:
            state: current state
            available_actions: a list of possible actions retrieved using env.get_available_actions()

        Returns:
            Action(row and column)
        """

        best_value: float = -np.inf
        best_action: Tuple[int, int]
        agent_state: np.ndarray = self.get_filtered_state(state.copy())

        if random() < self.exploration_chance:
            rand_idx: int = np.random.choice(len(available_actions))
            best_action = available_actions[rand_idx]
        else:
            for action in available_actions:
                new_state: np.ndarray = agent_state.copy()
                new_state[action[0], action[1]] = 1
                canonical: str = self.get_canonical_form(new_state)
                value: float = self.table[canonical]
                if value > best_value:
                    best_action = action
                    best_value = value

        return best_action

    def update_value(
        self, state: np.ndarray, new_state: np.ndarray, value: Optional[float]
    ) -> None:
        """Update the values in decision table

        Args:
            state: state of the previous move
            new_state: state after the move
            value: in case of gameover, value is non empty and contains 1 if it was a win, 0 if it was a loss, 0.5 otherwise
        """

        new_value: float
        if value is not None:
            self.table[self.get_canonical_form(new_state)] = value
            new_value = value
        else:
            new_value = self.table[self.get_canonical_form(new_state)]

        old_value = self.table[self.get_canonical_form(state)]
        self.table[self.get_canonical_form(state)] += self.learning_rate * (
            new_value - old_value
        )
