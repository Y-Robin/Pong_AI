# Import necessary libraries
import pygame
import numpy as np
from pong_game import PongGame # Custom game environment for Pong
from model import DQNModel  # Deep Q-Network model, adjust the import path as necessary


def player_main():
    # Initialize the Pygame library
    pygame.init()
    
    # Create an instance of the Pong game
    game = PongGame()
    clock = pygame.time.Clock()
    
    # Load the trained DQN model to play against
    model = DQNModel(input_shape=(game.HEIGHT, game.WIDTH, 2), action_size=2, load_saved_model=True)
    
    # Mapping of model actions to game actions
    stateBib = [1,-1]
    
    # Fetch the initial game state
    game.update(0, 0)
    prev_state = game.get_game_state()
    game.update(0, 0)
    current_state = game.get_game_state()
    
    # Combine the initial and next states to form the full state representation
    state = np.stack((prev_state, current_state), axis=-1)
    
    # Game loop
    running = True
    while running:
    
        # Handle Pygame events to check for quit signals
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Handle player input from keyboard
        keys = pygame.key.get_pressed()
        player_action = 0  # Default action is 'do nothing'
        if keys[pygame.K_UP]:
            player_action = -1  # Assign action for moving up
        elif keys[pygame.K_DOWN]:
            player_action = 1   # Assign action for moving down


        # Prepare the state for the model (mirror state for opposing perspective)
        mirrored_state = np.flip(state, axis=1)
        action_right = model.predict(mirrored_state)

        # Update the game state based on the player's and model's actions
        game_over = game.update(player_action, stateBib[action_right])
        
        # Update the state representation with the latest game state
        prev_state = current_state
        current_state = game.get_game_state()
        state = np.stack((prev_state, current_state), axis=-1)
        
        # Render the current game state
        game.draw()
        clock.tick(60)

        # If the game is over, reset the game for a new round
        if game_over:
            print(f"Game Over. Score - Left: {game.score_left}, Right: {game.score_right}")
            game.reset_game()

    pygame.quit()

if __name__ == '__main__':
    player_main()
