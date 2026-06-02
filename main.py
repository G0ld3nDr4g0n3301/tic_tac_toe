import pygame
from world.world import TicTacToeEnvironment

def main():
    env: TicTacToeEnvironment = TicTacToeEnvironment()
    running: bool = True
    done: bool = False
    env.render()
    while(running):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if done:
                    env.reset()
                    done = False
                    env.render()
                    continue
                x, y = pygame.mouse.get_pos()
                col: int = x // env.cell_size
                row: int = y // env.cell_size
                _, _, done, winner = env.step((row,col))
                if done:
                    symbol: str = "Draw"
                    if winner == 1:
                        symbol = 'X'
                    if winner == 2:
                        symbol = 'O'
                    print(f"Winner is {symbol}")
                env.render()
    env.close()

if __name__ == "__main__":
    main()