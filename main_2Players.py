# main.py
import pygame
from pong_game import PongGame

def main():
    game = PongGame()
    clock = pygame.time.Clock()
    game.update(0, 0)
    game_state_image1 = game.get_game_state()
    game.update(0, 0)
    game_state_image2 = game.get_game_state()
    running = True
    while running:
        paddle_left_movement = 0
        paddle_right_movement = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        # Handling left paddle
        if keys[pygame.K_w]:
            paddle_left_movement = -1
        if keys[pygame.K_s]:
            paddle_left_movement = 1

        # Handling right paddle
        if keys[pygame.K_UP]:
            paddle_right_movement = -1
        if keys[pygame.K_DOWN]:
            paddle_right_movement = 1

        # Update game with current inputs
        game_over = game.update(paddle_left_movement, paddle_right_movement)
        game_state_image1 = game_state_image2
        game_state_image2 = game.get_game_state()
        game.draw()
        clock.tick(60)
        if game_over:
            print(f"Score - Left: {game.score_left}, Right: {game.score_right}")

    pygame.quit()

if __name__ == '__main__':
    main()
