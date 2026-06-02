import numpy as np
import pygame
from typing import Optional, List, Tuple, Literal


class TicTacToeEnvironment:
    def __init__(self, field: Optional[List[int]] = None) -> None:
        """Initializes the environment

        Args:
            field: Preset field the game starts with

        Raises:
            ValueError: if field argument is of inappropriate size
        """
        self.board: np.ndarray = np.zeros((3, 3)).astype(int)
        self.first_turn: int = 1  # X moves first by default
        self.turn: int = self.first_turn
        self.done: bool = False
        self.screen: pygame.display | None = None
        self.screen_size: int = 300
        self.cell_size: int = 100

        if field is not None:
            if len(field) == 9:
                self.board = np.array(field).astype(int).reshape((3, 3))
            else:
                raise ValueError

    def reset(self) -> np.ndarray:
        """Resets the board for new game

        Returns:
            Environment ready for new game
        """

        self.board = np.zeros((3, 3)).astype(int)
        # We change who moves first in each new game
        self.first_turn = 1 if self.first_turn == 2 else 2
        self.turn = self.first_turn
        self.done = False

        return self.board.copy()

    def check_winner(self) -> Tuple[bool, int]:
        """Checks if someone has won

        Returns:
            Tuple of gameover flag and the winner. If the game is not finished or it's a draw, the winner is 0.
        """
        for i in range(3):
            if np.all(self.board[i, :] == 1) or np.all(self.board[:, i] == 1):
                return (True, 1)
            if np.all(self.board[i, :] == 2) or np.all(self.board[:, i] == 2):
                return (True, 2)

        if np.all(np.diag(self.board) == 1) or np.all(np.diag(np.fliplr(self.board)) == 1):
            return (True, 1)
        if np.all(np.diag(self.board) == 2) or np.all(np.diag(np.fliplr(self.board)) == 2):
            return (True, 2)

        if np.count_nonzero(self.board) >= 9:
            return (True, 0)

        return (False, 0)

    def check_appropriate(self, action: Tuple[int, int]) -> bool:
        """Checks if the move is appropriate

        Args:
            action: tuple with a move - a row index and a column index.

        Returns:
            True if the move is allowed, false otherwise
        """

        if self.done:
            return False

        if action[0] < 0 or action[0] > 2:
            return False

        if action[1] < 0 or action[1] > 2:
            return False

        if self.board[action[0]][action[1]] != 0:
            return False

        return True

    def step(self, action: Tuple[int, int]) -> Tuple[np.ndarray, None, bool, int]:
        """Apply player's action

        Args:
            action: action of a player in form of tuple with two indicies(row and column)

        Returns:
            Tuple of (next_state, reward, done, info). next_state is the board state after the move.
            reward is empty and is present for compatibility with API only.
            done is a flag if the game has ended.
            info contains a symbol of winner(0 in case of a draw or running game).

        Raises:
            ValueError: if the move is prohibited.
        """

        if not self.check_appropriate(action):
            raise ValueError("The move is inappropriate")

        self.board[action[0]][action[1]] = self.turn

        done: bool = False
        info: int = 0
        done, info = self.check_winner()

        self.done = done
        self.turn = 1 if self.turn == 2 else 2

        return (self.board.copy(), None, done, info)

    def get_available_actions(self) -> List[Tuple[int, int]]:
        """Returns list of available actions

        Returns:
            List of tuples with row and column indices of possible moves
        """

        return [tuple(idx) for idx in np.argwhere(self.board == 0)]

    def render(self, mode: Literal["gui", "cli", "headless"] = "gui") -> None:
        """Renders the representation of game state
        
        Args:
            mode: either "gui" for pygame gui,"cli" for command line output or "headless" for no output.
        """

        if mode == "headless":
            return
        if mode == "cli":
            print_board: np.ndarray = np.where(self.board == 0, "_", np.where(self.board == 1, "X", "O"))
            print('*' * 8)
            print('')
            for row in print_board:
                print(f"\t{' '.join(row)}")
            print('')
            print('-' * 8)
            return
        if mode == "gui":
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
                pygame.display.set_caption('Tic-Tac-Toe')

            self.screen.fill((255,255,255))

            for i in range(1,3):
                pygame.draw.line(self.screen, (0,0,0), (i*self.cell_size,0), (i*self.cell_size, self.screen_size), 2)
                pygame.draw.line(self.screen, (0,0,0), (0, i*self.cell_size), (self.screen_size, i*self.cell_size), 2)

            for row in range(3):
                for col in range(3):
                    cell_value = self.board[row,col]
                    center = (col * self.cell_size + self.cell_size // 2, row * self.cell_size + self.cell_size // 2)

                    if cell_value == 1:
                        pygame.draw.line(self.screen, (255,0,0), (center[0] - 30, center[1] - 30), (center[0] + 30, center[1] + 30), 5)
                        pygame.draw.line(self.screen, (255,0,0), (center[0] - 30, center[1] + 30), (center[0] + 30, center[1] - 30), 5)
                    elif cell_value == 2:
                        pygame.draw.circle(self.screen, (0,0,255), center, 30, 5)
            
            pygame.display.flip()

    def close(self) -> None:
        """Quits pygame on destruction of the environment
        """
        pygame.quit()