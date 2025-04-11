Durak AI: Deep Q-Learning Card Game Agent

This repository contains my implementation of Durak, a popular card game, featuring AI opponents powered by deep reinforcement learning. The project demonstrates how neural networks can learn optimal strategies for complex card games.

The AI uses:

Deep Q-Network (DQN) architecture with two neural networks (primary and target)
Experience replay with a memory size of 10,000 game states
Epsilon-greedy exploration strategy (starting at 1.0, decaying to 0.1)

State representation that encodes:

Cards in player's hand (36 features)
Previously played cards (36 features)
Cards currently on the field (36 features)
Trump suit indicator (4 features)
Relative hand sizes of all players (4 features)


Key Components

Game engine with complete Durak rules implementation
Custom reward system (+5 for successful attacks, +10 for successful defenses, +120 for winning)
Neural network with two hidden layers (might get increased later): (128 neurons each)
Automatic training system that allows agents to learn by playing against each other

Training Process:
The agents learn through self-play where they:

Make decisions based on their current policy
Store experiences in replay memory
Sample from memory to update network weights
Periodically update the target network
Gradually reduce exploration as they improve

Usage
The repository includes code to:

Train new agents from scratch
Play games with pre-trained models
Observe AI decision-making in action
And even play against AI opponents at different difficulty levels

Feel free to contribute to this project by reporting issues or suggesting new agent strategies!
