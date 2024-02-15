# Required imports for the game, model, and utilities
import pygame
import numpy as np
from pong_game import PongGame
from model import DQNModel  # Make sure this is the path to your model file
import matplotlib.pyplot as plt
import time

# Initialize variables for storing game scores and the best score
numColArray = []
numColBest = 0


def main():
    """Main function to run the Pong game with DQN model."""
    # For plotting
    plt.ion()
    fig, ax = plt.subplots()
    
    # Initialize game
    game = PongGame()
    clock = pygame.time.Clock()
    stateBib = [1,-1]
    
    # Model setup
    input_shape = (game.HEIGHTM, game.WIDTHM, 2)  # Input shape for the model
    action_size = 2  # Number of possible actions (up, down)
    model = DQNModel(input_shape, action_size,False) # Initialize the model
    numColBest = model.MaxScore
    batch_size = 32

    # Initial states
    game.update(0, 0)
    game.draw()
    clock.tick(1000)
    prev_state = game.get_game_state()
    game.update(0, 0)
    game.draw()
    clock.tick(1000)
    current_state = game.get_game_state()
    state = np.stack((prev_state, current_state), axis=-1)  # Combine two frames
    
    # Game loop variables
    gamePoints_left = 0
    gamePoints_right = 0
    running = True
    numFrames = 0
    
    while running:
        numFrames +=1
        
        # Quit if window is closed
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Model prediction 
        action_left = model.predict(state)

        
        # Prepare mirrored state for right paddle
        mirrored_state = np.flip(state, axis=1)
        
        # Predict action for right paddle
        action_right = model.predict(mirrored_state)

        # Update game state based on model predictions
        game_over = game.update(stateBib[action_left], stateBib[action_right])
        
        # Draw/update the game screen
        game.draw()
        clock.tick(1000)

        # Prepare for the next state
        prev_state = current_state
        current_state = game.get_game_state()
        stateOld = state
        state = np.stack((prev_state, current_state), axis=-1)
        
        # Calculate rewards and remember experiences
        r_l, r_r = game.calculate_rewards()
        model.remember(stateOld, action_left, r_l, state, game_over)  # For left paddle
        model.remember(mirrored_state, action_right, r_r, np.flip(state, axis=1), game_over)  # For right paddle

        

        # Handle game over and training
        if game_over:
            print("----")
            # Log score
            numColArray.append(game.numColOld) 
            
            # Save model if score improved
            if numColArray[-1] > numColBest:
                model.save_model_and_parameters(numColArray[-1])
                numColBest = numColArray[-1]
            
            #Plot Scores
            ax.clear()  # Clear the current plot
            ax.plot(numColArray)  # Plot the updated array
            plt.draw()
            plt.pause(0.5)
            
            
            print(f"Game Over. Score - Left: {game.score_left}, Right: {game.score_right}")

            numFrames = 0
            
            # Train model and update target model if enough memory collected
            if len(model.memory) > batch_size:
                model.train(batch_size)
            model.update_target_model()
            
            # Reset game for next round
            game.update(0, 0)
            # Draw/update the game screen
            game.draw()
            clock.tick(1000)
            prev_state = game.get_game_state()
            game.update(0, 0)
            # Draw/update the game screen
            game.draw()
            clock.tick(1000)
            current_state = game.get_game_state()
            state = np.stack((prev_state, current_state), axis=-1)

    pygame.quit()

if __name__ == '__main__':
    main()
