# PongAI Game

PongAI is a modern take on the classic game of Pong, enhanced with the power of Artificial Intelligence. This project enables users to train a neural network to play Pong, compete against the AI, or enjoy a classic two-player game.

This project utilizes the entire game frame as input for the neural network model, which allows the AI to learn and make decisions based on the current state of the game. The network architecture is designed around a Deep Q-Network (DQN) model, consisting of convolutional layers that process the frame, followed by dense layers that output the decision for the next move.

## Features

- **AI Training**: Train an AI model to learn playing Pong by observing and adapting to gameplay.
- **Single Player vs. AI**: Test your skills against a trained AI model.
- **Two-Player Mode**: Enjoy the classic Pong game with friends.

## Requirements

The game and AI are built using several powerful libraries. Ensure you have the following installed:

- Python 3.x
- Pygame
- NumPy
- TensorFlow
- Matplotlib
- OpenCV-Python

## Installation

Before running the game, install the required libraries using pip:


```bash
pip install pygame numpy tensorflow matplotlib opencv-python
```



## Structure

- "main.py": The script used to train the AI model.
- "main1vsKI.py": A script that allows you to play against the trained AI.
- "main_2Players.py": A script for a classic two-player game without AI.

## Usage

### Training the AI

To train the model, run:


```bash
python main.py
```



This will initiate the training process. The model will learn by playing against itself. Progress can be monitored via the console output.

### Playing Against the AI

Once the model is trained, you can test your skills against it:


```bash
python main1vsKI.py
```



Use the up and down arrow keys to control your paddle.

### Two-Player Mode

To play a game of Pong with a friend:


```bash
python main_2Players.py
```



Player 1 uses the W and S keys, while Player 2 uses the up and down arrow keys.

## Custom Game Environment

The game leverages a custom Pong environment ("pong_game.py") tailored for AI interaction and classic gameplay.

## AI Model

The AI ("model.py") is built using TensorFlow and employs a Deep Q-Network (DQN) to learn the game.

## Contributing

Contributions are welcome! If you have suggestions for improving the game or the AI, feel free to fork the repository and submit a pull request.

