import pygame
from world.world import TicTacToeEnvironment
from manager import AgentManager
from model.model import TD0Agent


def main() -> None:
    env: TicTacToeEnvironment = TicTacToeEnvironment()
    manager: AgentManager = AgentManager(env)

    trained_agent: TD0Agent = manager.train_self_play(100000, 0.6, 0.5)
    manager.save_agent(trained_agent, "agent1.dump")

    ai_agent: TD0Agent = TD0Agent(exploration_chance=0.0, role=2)
    manager.load_agent(ai_agent, "agent1.dump")

    running: bool = True
    done: bool = False
    state = env.reset()
    env.render()

    while running:
        if env.turn == 2 and not done:
            pygame.time.wait(300)
            actions = env.get_available_actions()
            ai_action = ai_agent.choose_action(state, actions)
            state, _, done, winner = env.step(ai_action)
            env.render(mode="gui")
            if done:
                print(f"Gameover! The winner is {winner}")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN and env.turn == 1 and not done:
                x, y = pygame.mouse.get_pos()
                col: int = x // env.cell_size
                row: int = y // env.cell_size
                if env.check_appropriate((row, col)):
                    state, _, done, winner = env.step((row, col))
                    env.render(mode="gui")
                    if done:
                        print(f"Gameover! The winner is {winner}")

            if event.type == pygame.MOUSEBUTTONDOWN and done:
                state = env.reset()
                done = False
                env.render(mode="gui")
    env.close()


if __name__ == "__main__":
    main()
