import numpy as np
from typing import *

class TicTacToeEnvironment:
    
    def __init__(self, field: Optional[List] = None) -> None:
        """Initializes the environment

        Args:
            field: Preset field the game starts with

        Raises:
            ValueError: if field argument is of inappropriate size
        """
        self.board: np.ndarray = np.zeros((3,3)).astype(int)
        self.turn: int = 1 # X moves first by default
        self.done: bool = False
        
        if field is not None:
            if len(field) == 9:
                self.board = np.array(field).astype(int).reshape((3,3))
            else:
                raise ValueError
        
    def reset(self) -> TicTacToeEnvironment:
        """Resets the board for new game
        
        Returns:
            Environment ready for new game
        """

        self.board = np.zeros((3,3)).astype(int)
        self.turn = 1 if self.turn == 2 else 2  # We change who moves first in each new game
        self.done = False

        return self

    def check_winner(self) -> Tuple[bool, int | None]:
        """Checks if someone has won
        
        Returns:
            Tuple of self.done and the winner. If the game is not finished or it's a draw, the winner is None.
        """

        # somehow efficiently check the win conditions on self.board

        return (self.done, winner)

    def step(self, action) -> Tuple[np.ndarray, None, bool, None]:
        """Apply player's action

        Args:
            action: action of a player
        
        Returns:
            Tuple of (next_state, reward, done, info). next_state is the board state after the move. 
            reward and info are empty and are present for compatibility with API only. 
            done is a flag if the game has ended.

        Raises:
            ValueError: if the move is prohibited.
        """

        # call check_winner somewhere here

        return (next_state,None,done,None)

    # ToDo
    def render(self, mode = "gui" | "headless"):
        pass
