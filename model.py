# Import necessary libraries
import random
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Flatten, Dense, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import gc
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import os
import json

class DQNModel:
    def __init__(self, input_shape, action_size,load_saved_model=False):
        # Initialize main parameters
        self.action_size = action_size
        
        # Paths for saving/loading the model and parameters
        self.model_path = "model_Store"
        self.params_path = "params_Store"
        
        # Minibatch size
        self.miniSize = 320
        
        # Load saved model if specified and exists, else create a new model
        if load_saved_model and os.path.exists(self.model_path):
            print("Loading saved model...")
            self.model = tf.keras.models.load_model(self.model_path)
            # Load training parameters
            if os.path.exists(self.params_path):
                with open(self.params_path, 'r') as f:
                    params = json.load(f)
                    self.epsilon = params['epsilon']
                    self.gamma = params['gamma']
                    self.MaxScore = params['MaxScore']
                    # Add other parameters as needed
        else:
            print("Creating new model...")
            self.model = self.build_model(input_shape)
            self.epsilon = 1.0
            self.gamma = 0.9
            self.MaxScore = 0
        
        # Clone model to create a target model for stable Q-value estimation
        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())
        
        # Experience replay memory
        self.memory = deque(maxlen=200000)

        # Exploration parameters
        self.epsilon_min = 0.01 # Minimum exploration rate
        self.epsilon_decay = 0.99 # Decay rate for exploration
        
        
        # Model analysis tools for visualizing conv layers
        self.visualization_model = tf.keras.models.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer('conv2dLast').output
        )
        
        self.visualization_model_2 = tf.keras.models.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer('cnn1').output
        )
        
        
        
    def save_model_and_parameters(self, MaxScore):
        # Save the model weights and training parameters
        self.model.save(self.model_path)
        params = {
            'epsilon': self.epsilon,
            'gamma': self.gamma,
            'MaxScore': MaxScore,
        }
        with open(self.params_path, 'w') as f:
            json.dump(params, f)
        
    def build_model(self, input_shape):
    
        
        # CNN architecture definition using Keras Sequential API
        initial_learning_rate = 0.0001
        lr_schedule = ExponentialDecay(
            initial_learning_rate,
            decay_steps=self.miniSize,
            decay_rate=0.995 ,
            staircase=False)
            
        # CNN architecture
        model = Sequential([
            Conv2D(16, (8, 8), padding='same', input_shape=input_shape,name='cnn1'),
            ReLU(),
            MaxPooling2D(pool_size=(4, 4)),  # Adding max pooling layer

            Conv2D(32, (4, 4), padding='same'),
            ReLU(),
            MaxPooling2D(pool_size=(2, 2)),  # Adding max pooling layer

            Conv2D(32, (3, 3), padding='same',name='conv2dLast'),
            ReLU(),
            MaxPooling2D(pool_size=(2, 2)),  # Adding max pooling layer

            Flatten(),
            Dense(256, activation='relu'),
            Dense(self.action_size,activation='linear')  # Output layer
        ])
        model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='mean_squared_error')
        model.summary()
        return model

    def remember(self, state, action, reward, next_state, done):
        # Store experience in replay memory
        self.memory.append((state, action, reward, next_state, done))
        
            

    def predict(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.expand_dims(state, axis=0)
        act_values = self.model(state)
        return np.argmax(act_values[0])

    def train(self, batch_size):
        #Training loop with experience replay
        for i in range(6): # Train in mini-batches
            if len(self.memory) < 1000:  # Ensure sufficient data
                return 
                
            # Extract components of experiences
            minibatch = random.sample(self.memory, self.miniSize)
            states = np.array([t[0] for t in minibatch])
            actions = np.array([t[1] for t in minibatch])
            rewards = np.array([t[2] for t in minibatch])
            next_states = np.array([t[3] for t in minibatch])
            dones = np.array([t[4] for t in minibatch])

            # Predict future rewards and update Q-values
            target_qs = self.target_model.predict(next_states)
            max_future_qs = np.amax(target_qs, axis=1)
            
            # Compute target Q values
            for j in range(self.miniSize):
                if rewards[j] < -0.8:
                    target_qs = rewards + self.gamma * -10 * (1 - dones)
                else:
                    target_qs = rewards + self.gamma * 5 * (1 - dones)

            # Compute Q values for current states
            current_qs = self.model.predict(states)
            for index in range(self.miniSize):
                current_qs[index][actions[index]] = target_qs[index]
            
            # Possible Visualization of batches and conv layers
            # self.inspect_batch(states, current_qs, rewards)
            # self.visualize_first_maxpool_output(states[0])
            
            
            # Train the model in one batch
            print(self.model.optimizer.learning_rate(self.model.optimizer.iterations))
            self.model.fit(states, current_qs, batch_size=batch_size, epochs=1, verbose=1)

            # Adjust the exploration rate
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                print(f"Epsilon: {self.epsilon}")
                
            # Manually free memory
            del minibatch, states, actions, rewards, next_states, dones, target_qs, max_future_qs, current_qs
            gc.collect()  # Call the garbage collector
            
            
    def inspect_batch(self,states, actions, rewards, pause_time=1.0):
        # Visualizes a batch of states along with their corresponding actions and rewards.
        for i in range(len(states)):
            # Display the first channel of the image state
            plt.subplot(1, 2, 1)
            plt.imshow(states[i][:,:,0].astype('uint8'))
            # Display the second channel of the image state
            plt.subplot(1, 2, 2)
            plt.imshow(states[i][:,:,1].astype('uint8'))
            plt.title(f"Action: {actions[i]}, Reward: {rewards[i]}")
            plt.show(block=False)
            plt.pause(pause_time)
            # Allow manual control over inspection
            key = input("Press Enter to continue to the next observation...")  # Wait for user input
            if key.lower() == 's':
                return
            plt.close()
                
    def visualize_first_maxpool_output(self, state):
        #  Visualizes the output of the first MaxPooling layer in the model for a given state.
        
        # Prepare the state for model input
        state = np.expand_dims(state, axis=0)
        
        # Predict the layer outputs for the given state
        output = self.visualization_model.predict(state)
        output2 = self.visualization_model_2.predict(state)
        
        # Visualize selected filters from the layer outputs
        plt.figure(figsize=(10, 10))
        idx = 1
        for i in range(0,output.shape[-1],50):  # Iterate over all filters
            plt.subplot(4, 4, idx)  # Display output from the first specified layer
            plt.imshow(output[0, :, :, i], cmap='gray')
            plt.axis('off')
            plt.subplot(4, 4, idx+1)  # Display output from the first specified layer
            plt.imshow(output2[0, :, :, 1], cmap='gray')
            plt.axis('off')
            idx += 2
            
            # Manual control to move through visualizations
            key = input("Press Enter to continue to the next observation...")  # Wait for user input
            if key.lower() == 's':
                return
        plt.show()
        

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

