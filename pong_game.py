# Import necessary libraries
import pygame
import numpy as np
import random
import cv2

# Scaling factor to adjust the game's resolution and element sizes
gamesizeMultiplyer = 0.4

class PongGame:
    def __init__(self):
        # Game dimension constants, scaled by the multiplier
        self.WIDTH, self.HEIGHT = 2000*gamesizeMultiplyer, 2000*gamesizeMultiplyer
        # Model dimensions for processing (e.g., by a neural network)
        self.WIDTHM, self.HEIGHTM = 100, 100
        # Ball and paddle dimensions
        self.BALL_RADIUS = 150*gamesizeMultiplyer
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 50*gamesizeMultiplyer, 200*gamesizeMultiplyer
        # Color definitions
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        
        # Initialize the Pygame library and set up the display
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption('Pong')

         # Initialize game elements (ball and paddles) with their positions and sizes
        self.ball = pygame.Rect(self.WIDTH // 2, self.HEIGHT // 2, self.BALL_RADIUS, self.BALL_RADIUS)
        self.paddle_left = pygame.Rect(100*gamesizeMultiplyer, self.HEIGHT // 2 - self.PADDLE_HEIGHT // 2, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        self.paddle_right = pygame.Rect(self.WIDTH - 150*gamesizeMultiplyer, self.HEIGHT // 2 - self.PADDLE_HEIGHT // 2, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        
        # Ball movement speed
        self.ball_speed_x = 30*gamesizeMultiplyer
        self.ball_speed_y = 30*gamesizeMultiplyer

        # Scoring and collision tracking
        self.numCol = 0
        self.numColOld = 0
        
        # Paddle movement speed
        self.paddle_speed = 50*gamesizeMultiplyer
        
        # Score tracking
        self.score_left = 0
        self.score_right = 0
        
        # Initial and current settings for game elements
        self.init_ball_speed_x = 30*gamesizeMultiplyer
        self.init_ball_speed_y = 30*gamesizeMultiplyer
        self.init_paddle_height = self.PADDLE_HEIGHT
        self.ball_speed_x = self.init_ball_speed_x
        self.ball_speed_y = self.init_ball_speed_y
        self.paddle_height = self.init_paddle_height

        # Timing for game difficulty adjustments
        self.last_increase_time = pygame.time.get_ticks()
        self.increase_interval = 5000   # Interval (in milliseconds) to check for difficulty increase
        self.min_paddle_height = 50*gamesizeMultiplyer
        
        # Track the last movement direction of paddles
        self.lastPaddleLeft = 0
        self.lastPaddleRight = 0
        
        # Initialize a font for rendering text
        self.font = pygame.font.SysFont('Arial', 40)
        
        # Initial drawing of game elements
        self.draw()
        
        
    def display_rewards(self, reward_left, reward_right):
         """Display the rewards for each paddle on the screen."""
         # For Debugging
        # reward_left_text = self.font.render(f'Left Reward: {reward_left}', True, self.WHITE)
        # reward_right_text = self.font.render(f'Right Reward: {reward_right}', True, self.WHITE)
        # self.screen.blit(reward_left_text, (50, 50))
        # self.screen.blit(reward_right_text, (self.WIDTH - 200, 70))

    def draw(self):
        """Draw the game elements on the screen and update the display."""
        # Fill the screen with black background
        self.screen.fill(self.BLACK)
        
        # Draw the ball and paddles
        pygame.draw.ellipse(self.screen, self.WHITE, self.ball)
        pygame.draw.rect(self.screen, self.WHITE, self.paddle_left)
        pygame.draw.rect(self.screen, self.WHITE, self.paddle_right)
        
        # Capture the current frame as image data for processing
        self.image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        self.image_data = np.transpose(self.image_data, (1, 0, 2))  # Transpose the image data
        
        # Calculate and display rewards (currently not displayed to the screen)
        reward_left, reward_right = self.calculate_rewards()
        self.display_rewards(reward_left, reward_right)

        # Update the Pygame display
        pygame.display.flip()
        
        
    def move_ball(self):
        """Update the ball's position based on its speed and handle collisions."""
        # Update ball position
        self.ball.x += self.ball_speed_x
        self.ball.y += self.ball_speed_y
        
        # Collision with top and bottom walls
        if (self.ball.top <= 0 and self.ball_speed_y < 0) or (self.ball.bottom >= self.HEIGHT and self.ball_speed_y > 0):
            self.ball_speed_y *= -1

        # Collision with paddles
        if (self.ball.colliderect(self.paddle_left) and self.ball_speed_x < 0) or (self.ball.colliderect(self.paddle_right)and self.ball_speed_x > 0):
            self.ball_speed_x *= -1 # Reverse X direction
            self.ball.x += self.ball_speed_x*2 # Move ball out of paddle to prevent sticking
            self.ball.y += self.ball_speed_y*2
            self.numCol += 1
            
            
    def reset_ball(self):
        """Reset the ball's position and speed to the center with random direction."""
        # Random speed and direction
        self.ball_speed_x = random.randint(20*gamesizeMultiplyer, 40*gamesizeMultiplyer) * random.choice([-1, 1])
        self.ball_speed_y = random.randint(20*gamesizeMultiplyer, 40*gamesizeMultiplyer) * random.choice([-1, 1])
        # Center position
        self.ball.x = self.WIDTH // 2
        self.ball.y = self.HEIGHT // 2


    def check_score(self):
        """Check if the ball has crossed the goal lines to update the score."""
        # Goal on the left side
        if self.ball.x <= 0:
            self.score_right += 1
            self.reset_ball()
            return True
        # Goal on the right side
        elif self.ball.x >= self.WIDTH:
            self.score_left += 1
            self.reset_ball()
            return True
        return False
        
    def increase_difficulty(self):
        """Periodically check to increase the game difficulty."""
        # No specific difficulty increase logic in this version
        current_time = pygame.time.get_ticks()
        if current_time - self.last_increase_time > self.increase_interval:
            self.last_increase_time = current_time

            # Increase ball speed
            self.ball_speed_x *= 1.0
            self.ball_speed_y *= 1.0
            # Decrease paddle size
            if self.paddle_height > self.min_paddle_height:
                self.paddle_height -= 0
                self.paddle_left.height = self.paddle_height
                self.paddle_right.height = self.paddle_height
                
    def get_game_state(self):
        """Capture and return the current game state as an image."""
        # Resize the captured image data for model processing
        retIm = cv2.resize(self.image_data[:,:,0], (self.WIDTHM, self.HEIGHTM ), interpolation=cv2.INTER_LINEAR)
        return retIm

    def reset_game(self):
        """Reset the game to its initial state after a score or game over."""
        # Reset score and paddle positions
        self.numColOld = self.numCol # Store the collision count
        self.numCol = 0 # Reset collision count
        
        self.paddle_height = self.init_paddle_height
        self.paddle_left.height = self.paddle_height
        self.paddle_right.height = self.paddle_height
        self.paddle_left.y = self.HEIGHT // 2 - self.PADDLE_HEIGHT // 2
        self.paddle_right.y = self.HEIGHT // 2 - self.PADDLE_HEIGHT // 2
        self.reset_ball()
        self.lastPaddleLeft = 0
        self.lastPaddleRight = 0
        
    def calculate_rewards(self):
        """Calculate the rewards for each paddle based on the game state."""
        # This method computes rewards for paddles based on their actions and the ball's state

        # Initialize rewards for both paddles
        reward_left = 0
        reward_right = 0

        # Check if the ball is moving towards the left paddle
        if self.ball_speed_x < 0:
            if self.ball.colliderect(self.paddle_left):
                # Reward for hitting the ball
                reward_left = 50  
            elif self.ball.x <= 120*gamesizeMultiplyer:
                # Penalty for missing the ball
                reward_left = -10  
            elif self.ball.y >= self.paddle_left.y+10*gamesizeMultiplyer and self.ball.y <= self.paddle_left.y+90*gamesizeMultiplyer:
                # y coordinate of ball is within paddle
                reward_left = 100/ self.ball.x 
            else:
                if self.ball.y > self.paddle_left.y and self.lastPaddleLeft == -1:
                    # Penalty for paddle moving away from the ball
                    reward_left = -100/ self.ball.x
                elif self.ball.y < self.paddle_left.y and self.lastPaddleLeft == 1:
                    # Penalty for paddle moving away from the ball
                    reward_left = -100/ self.ball.x
                else:
                    # Reward for following the ball
                    reward_left = 3
        
        
        # Check if the ball is moving towards the right paddle
        if self.ball_speed_x > 0:
            if self.ball.colliderect(self.paddle_right):
                # Reward for hitting the ball
                reward_right = 50
            elif self.ball.x >= self.WIDTH-120*gamesizeMultiplyer:
                # Penalty for missing the ball
                reward_right = -10  # Penalty for missing the ball
            elif self.ball.y >= self.paddle_right.y+10*gamesizeMultiplyer and self.ball.y <= self.paddle_right.y+90*gamesizeMultiplyer:
                # y coordinate of ball is within paddle
                reward_right = 100/ (self.WIDTH+1-self.ball.x)

            else:
                if self.ball.y > self.paddle_right.y and self.lastPaddleRight == -1:
                    # Penalty for paddle moving away from the ball
                    reward_right = -100/ (self.WIDTH+1-self.ball.x)
                elif self.ball.y < self.paddle_right.y and self.lastPaddleRight == 1:
                    # Penalty for paddle moving away from the ball
                    reward_right = -100/ (self.WIDTH+1-self.ball.x)
                else:
                    # Reward for following the ball
                    reward_right = 3

        return reward_left, reward_right

    def update(self, paddle_left_movement, paddle_right_movement):
        """Update the game state based on paddle movements and check for game over."""
        # Update paddle positions and check for game state changes
        self.lastPaddleLeft = paddle_left_movement
        self.lastPaddleRight = paddle_right_movement
        
        # Update paddle positions based on input
        self.paddle_left.y += paddle_left_movement * self.paddle_speed
        self.paddle_right.y += paddle_right_movement * self.paddle_speed

        # Ensure paddles stay within the screen
        self.paddle_left.y = max(self.paddle_left.y, 0)
        self.paddle_left.y = min(self.paddle_left.y, self.HEIGHT - self.paddle_height)
        self.paddle_right.y = max(self.paddle_right.y, 0)
        self.paddle_right.y = min(self.paddle_right.y, self.HEIGHT - self.paddle_height)

        # Move the ball
        self.move_ball()
        
        # Increase difficulty
        self.increase_difficulty()
        
        # Check for game over
        game_over = self.check_score()
        if game_over:
            self.reset_game()
        return game_over


    # Add more methods as needed for ball movement, collision, scoring, etc.
