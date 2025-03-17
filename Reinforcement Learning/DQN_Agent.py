import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from collections import deque
import matplotlib.pyplot as plt
import time

# Check for GPU availability
print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU is available: {gpus}")
    # Set memory growth to avoid consuming all memory
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Error setting memory growth: {e}")
else:
    print("No GPU found, using CPU")

class DQNAgent:
    def __init__(self, state_size=11, action_encoding_size=20, memory_size=2000, 
                 gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, 
                 learning_rate=0.001, batch_size=64, episodes=1000, 
                 target_update_freq=100):
        """
        Initialize DQN agent for the ChordalGraphEnv
        
        Args:
            state_size: Dimension of state features
            action_encoding_size: Size of action encoding
            memory_size: Size of replay memory
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Rate at which epsilon decreases
            learning_rate: Learning rate for the neural network
            batch_size: Batch size for training (larger for GPU)
            episodes: Number of episodes to train
            target_update_freq: Frequency to update target network
        """
        self.state_size = state_size
        self.action_encoding_size = action_encoding_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.episodes = episodes
        self.target_update_freq = target_update_freq
        
        # Build model with GPU support if available
        with tf.device('/gpu:0' if gpus else '/cpu:0'):
            self.model = self._build_model()
            self.target_model = self._build_model()
        
        self.update_target_model()
        
        # Metrics
        self.rewards_history = []
        self.clique_size_history = []
        self.loss_history = []
        self.training_times = []
        
    def _build_model(self):
        """Build a neural network model for DQN with optimizations for GPU"""
        model = Sequential([
            Dense(128, activation='relu', input_dim=self.state_size + self.action_encoding_size),
            Dropout(0.2),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(1, activation='linear')  # Output Q-value
        ])
        
        # Use mixed precision for better GPU performance
        if gpus:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(loss='mse', optimizer=optimizer)
        return model
    
    def update_target_model(self):
        """Update target model weights with the main model weights"""
        self.target_model.set_weights(self.model.get_weights())
    
    def get_state_features(self, state):
        """Extract numeric features from state dictionary"""
        features = np.array([
            state['num_vertices'],
            state['num_edges'],
            state['max_degree'],
            state['min_degree'],
            state['mean_degree'],
            state['max_clique_size'],
            state['avg_clique_size'],
            state['clique_size_std'],
            state['average_clustering'],
            state['global_clustering'],
            state['density']
        ], dtype=np.float32)  # Specify dtype for better GPU compatibility
        
        # Handle infinity values
        features = np.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return features
    
    def encode_action(self, action, all_actions):
        """
        Encode an action (pair of cliques) in a way that neural network can process
        This encoding captures the relative position of the cliques in the graph
        """
        # If no action is provided, return zeros
        if action is None:
            return np.zeros(self.action_encoding_size, dtype=np.float32)
        
        # Get clique sizes
        clique_a, clique_b = action
        size_a = len(clique_a)
        size_b = len(clique_b)
        
        # Calculate overlap
        overlap = len(set(clique_a).intersection(set(clique_b)))
        
        # Determine the relative position in the list of all actions
        try:
            action_index = all_actions.index(action)
            relative_position = action_index / (len(all_actions) - 1) if len(all_actions) > 1 else 0.5
        except ValueError:
            relative_position = 0.5  # Default if action not found
        
        # Create encoding
        encoding = np.zeros(self.action_encoding_size, dtype=np.float32)
        
        # Set values based on properties of the cliques
        encoding[0] = size_a / 20  # Normalize clique size
        encoding[1] = size_b / 20
        encoding[2] = overlap / 10  # Normalize overlap
        encoding[3] = relative_position
        
        # Encode the elements of each clique (limited to first 8 elements)
        for i, node in enumerate(list(clique_a)[:8]):
            if i + 4 < self.action_encoding_size:
                encoding[i + 4] = node / 40  # Normalize node index
        
        for i, node in enumerate(list(clique_b)[:8]):
            if i + 12 < self.action_encoding_size:
                encoding[i + 12] = node / 40  # Normalize node index
        
        return encoding
    
    def remember(self, state, action, reward, next_state, done, valid_actions):
        """Store experience in memory"""
        state_features = self.get_state_features(state)
        action_encoding = self.encode_action(action, valid_actions)
        
        if next_state:
            next_state_features = self.get_state_features(next_state)
        else:
            next_state_features = np.zeros_like(state_features)
        
        self.memory.append((state_features, action_encoding, reward, next_state_features, done, valid_actions))
    
    def act(self, state, valid_actions):
        """Choose an action using epsilon-greedy policy"""
        if not valid_actions:
            return None
        
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # Get Q-values for all valid actions
        state_features = self.get_state_features(state)
        
        # For efficiency, batch predict all actions at once
        batch_inputs = []
        for action in valid_actions:
            action_encoding = self.encode_action(action, valid_actions)
            state_action = np.concatenate([state_features, action_encoding])
            batch_inputs.append(state_action)
        
        # Convert to numpy array for batch prediction
        batch_inputs = np.array(batch_inputs)
        
        # Use GPU to predict in batch
        with tf.device('/gpu:0' if gpus else '/cpu:0'):
            q_values = self.model.predict(batch_inputs, verbose=0).flatten()
        
        # Get action with highest Q-value
        max_q_index = np.argmax(q_values)
        return valid_actions[max_q_index]
    
    def replay(self):
        """Train the model on a batch of experiences using GPU for batch processing"""
        if len(self.memory) < self.batch_size:
            return 0
        
        # Sample batch
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Extract components
        states = []
        targets = []
        
        for state_features, action_encoding, reward, next_state_features, done, valid_actions in minibatch:
            state_action = np.concatenate([state_features, action_encoding])
            
            target = -reward  # Negate reward since we're minimizing
            
            if not done and valid_actions:  # Check if there are valid actions
                # Get Q-values for next state's valid actions
                next_state_actions = []
                for next_action in valid_actions:
                    next_action_encoding = self.encode_action(next_action, valid_actions)
                    next_state_action = np.concatenate([next_state_features, next_action_encoding])
                    next_state_actions.append(next_state_action)
                
                # Batch prediction for efficiency
                with tf.device('/gpu:0' if gpus else '/cpu:0'):
                    next_q_values = self.target_model.predict(np.array(next_state_actions), verbose=0).flatten()
                    
                # Get maximum Q-value for next state
                if len(next_q_values) > 0:
                    target += self.gamma * np.max(next_q_values)
            
            # Store for batch training
            states.append(state_action)
            targets.append([target])
        
        # Convert to numpy arrays
        states = np.array(states)
        targets = np.array(targets)
        
        # Train in batch with GPU
        with tf.device('/gpu:0' if gpus else '/cpu:0'):
            history = self.model.fit(states, targets, batch_size=self.batch_size, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return history.history['loss'][0]
    
    def train(self, env):
        """Train the agent on the given environment using GPU acceleration"""
        self.env = env
        target_update_counter = 0
        
        for episode in range(self.episodes):
            # Reset environment
            env._reset()
            state = env._get_state()
            done = False
            episode_reward = 0
            steps = 0
            
            # Store initial number of cliques
            initial_cliques = len(env.clique_graph)
            
            episode_start_time = time.time()
            
            while not done and steps < 100:  # Limit steps to prevent infinite loops
                # Get valid clique merges
                valid_actions = env.valid_clique_merges()
                
                if not valid_actions:
                    break
                
                # Choose action
                action = self.act(state, valid_actions)
                
                if action is None:
                    break
                
                # Take action
                next_state, done = env.step(action)
                
                # Get reward (we're trying to minimize it)
                reward = env.generate_reward()
                episode_reward += reward
                
                # Remember experience
                self.remember(state, action, reward, next_state, done, valid_actions)
                
                # Update state
                state = next_state
                steps += 1
                
                # Train model
                loss = self.replay()
                self.loss_history.append(loss)
                
                # Update target network
                target_update_counter += 1
                if target_update_counter >= self.target_update_freq:
                    self.update_target_model()
                    target_update_counter = 0
            
            # Record metrics
            episode_time = time.time() - episode_start_time
            self.training_times.append(episode_time)
            
            final_cliques = len(env.clique_graph)
            self.rewards_history.append(episode_reward)
            self.clique_size_history.append((initial_cliques, final_cliques))
            
            # Print progress
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{self.episodes} - Reward: {episode_reward} - Epsilon: {self.epsilon:.4f}")
                print(f"Cliques reduced from {initial_cliques} to {final_cliques}")
                print(f"Episode time: {episode_time:.2f}s")
                if done:
                    print("Successfully made graph chordal!")
    
    def test(self, env, num_episodes=10, visualize=True):
        """Test the trained agent on the environment"""
        self.epsilon = 0  # No exploration during testing
        success_count = 0
        avg_reward = 0
        avg_steps = 0
        
        for episode in range(num_episodes):
            # Reset environment
            env._reset()
            state = env._get_state()
            done = False
            episode_reward = 0
            steps = 0
            
            initial_cliques = len(env.clique_graph)
            
            # Initial visualization
            if visualize and episode == 0:
                print("Initial graph:")
                env.visualize()
            
            while not done and steps < 100:
                # Get valid clique merges
                valid_actions = env.valid_clique_merges()
                
                if not valid_actions:
                    print("No valid actions available.")
                    break
                
                # Choose best action
                action = self.act(state, valid_actions)
                
                # Take action
                next_state, done = env.step(action)
                reward = env.generate_reward()
                episode_reward += reward
                
                # Update state
                state = next_state
                steps += 1
            
            final_cliques = len(env.clique_graph)
            avg_reward += episode_reward
            avg_steps += steps
            
            if done:
                success_count += 1
            
            print(f"Test Episode {episode + 1} - Reward: {episode_reward} - Steps: {steps}")
            print(f"Cliques reduced from {initial_cliques} to {final_cliques}")
            print(f"Is chordal: {env._is_chordal()}")
            
            # Final visualization
            if visualize and episode == 0:
                print("Final graph:")
                env.visualize()
        
        print(f"\nSuccess rate: {success_count/num_episodes:.2%}")
        print(f"Average reward: {avg_reward/num_episodes:.2f}")
        print(f"Average steps: {avg_steps/num_episodes:.2f}")
    
    def save_model(self, filepath):
        """Save the trained model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def plot_learning_curve(self):
        """Plot the learning curve of the agent"""
        plt.figure(figsize=(15, 10))
        
        # Plot rewards
        plt.subplot(2, 2, 1)
        plt.plot(self.rewards_history)
        plt.title('Rewards per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        # Plot clique reduction
        plt.subplot(2, 2, 2)
        initial_cliques = [x[0] for x in self.clique_size_history]
        final_cliques = [x[1] for x in self.clique_size_history]
        plt.plot(initial_cliques, label='Initial Cliques')
        plt.plot(final_cliques, label='Final Cliques')
        plt.title('Clique Reduction')
        plt.xlabel('Episode')
        plt.ylabel('Number of Cliques')
        plt.legend()
        
        # Plot loss
        plt.subplot(2, 2, 3)
        if self.loss_history:
            # Plot with moving average for smoothing
            window_size = min(50, len(self.loss_history))
            if window_size > 0:
                moving_avg = np.convolve(self.loss_history, np.ones(window_size)/window_size, mode='valid')
                plt.plot(moving_avg, label='Moving Average')
            plt.plot(self.loss_history, alpha=0.3, label='Raw Loss')
            plt.title('Training Loss')
            plt.xlabel('Training Step')
            plt.ylabel('Loss')
            plt.legend()
        
        # Plot training time
        plt.subplot(2, 2, 4)
        plt.plot(self.training_times)
        plt.title('Training Time per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Time (seconds)')
        
        plt.tight_layout()
        plt.show()
        
    def visualize_action_selection(self, state, valid_actions, num_to_show=5):
        """Visualize how the agent values different actions for a given state"""
        if not valid_actions:
            print("No valid actions available.")
            return
        
        state_features = self.get_state_features(state)
        
        # For efficiency, batch predict all actions at once
        batch_inputs = []
        for action in valid_actions:
            action_encoding = self.encode_action(action, valid_actions)
            state_action = np.concatenate([state_features, action_encoding])
            batch_inputs.append(state_action)
        
        # Convert to numpy array for batch prediction
        batch_inputs = np.array(batch_inputs)
        
        # Use GPU to predict in batch
        with tf.device('/gpu:0' if gpus else '/cpu:0'):
            q_values = self.model.predict(batch_inputs, verbose=0).flatten()
        
        # Create a list of (action, q_value) pairs
        action_values = list(zip(valid_actions, q_values))
        
        # Sort by Q-value (descending)
        action_values.sort(key=lambda x: x[1], reverse=True)
        
        # Show top actions
        print(f"Top {num_to_show} actions by Q-value:")
        for i, (action, q_value) in enumerate(action_values[:num_to_show]):
            clique_a, clique_b = action
            print(f"{i+1}. Q-value: {q_value:.4f}")
            print(f"   Clique A: {clique_a} (size: {len(clique_a)})")
            print(f"   Clique B: {clique_b} (size: {len(clique_b)})")
            print(f"   Overlap: {len(set(clique_a).intersection(set(clique_b)))}")
            print()