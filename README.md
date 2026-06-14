# TD(0)

The project contains an implementation of TD(0) algorithm, which updates agent's decision table(key-value structure with the state of the environment as key and current approximation of true value function as a value) based on the outcome of the game.

The formula is: $V(s_{n-1}) = V(s_{n-1}) + \nu * (V(s_{n}) - V(s_{n-1}))$, where V(x) is an approximation of the value function(a value in our decision table), s_x is a game state on Xth agent move, and \nu is a learning rate hyperparameter.

Also, the algorithm takes game field symmetries into account, and uses canonical forms to find identical states. The states present in the decision table as keys are only the ones whose string representations are lexigraphically smallest among all the symmetries.

The hyperparameters of an agent include:
1. Game count. Represents how many games do you want to play.
2. Starting learning rate.
3. Starting Epsilon. It's a probability that agent will make a random move, and not the first one from the list of moves with highest value from the decision table.

In my implementation, I added decay feature to the epsilon and learning rate parameters, so throughout the learning process theese parameters decrease linearly from their starting values down until the minimal values.