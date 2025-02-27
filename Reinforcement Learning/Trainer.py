import networkx as nx
import random
import numpy as np
from collections import deque
import time

class ChordalTrainer:
    def __init__(self, agent, env, gamma=0.95, buffer_size=10000, batch_size=32, 
                 reward_scaling=1.0, save_interval=np.inf, checkpoint_path="./checkpoints/"):
        self.agent = agent             # RL agent
        self.env = env                 # Graph environment
        # RF Parameters
        self.gamma = gamma             # Discount factor
        self.episode_buffer = deque(maxlen=buffer_size)  # Experience replay buffer
        self.batch_size = batch_size   # Training batch size
        self.reward_scaling = reward_scaling  # Scale rewards for numerical stability

        #Saving
        self.save_interval = save_interval  # Episodes between checkpoints
        self.checkpoint_path = checkpoint_path
        # Create checkpoint directory if it doesn't exist
        import os
        os.makedirs(checkpoint_path, exist_ok=True)

        # Performance tracking
        self.rewards_history = []
        self.episode_lengths = []
        self.best_reward = 0
        self.training_start_time = None

        # Cache for analysis results
        self.cache = {
            'simulation_data': None,
            'feature_importance': None,
            'action_patterns': None,
            'learning_progress': None,
            'state_transitions': None
        }

        # Feature names mapping
        self.feature_names = [
            'num_vertices', 'num_edges', 'max_degree', 'min_degree', 'mean_degree',
            'max_clique_size', 'avg_clique_size', 'clique_size_std', 'average_clustering',
            'global_clustering', 'density', 'diameter', 'radius', 'clique1_size', 'clique2_size'
        ]
        
    
    def run_episode(self, training=True, render=False):
        """Run a single episode of training or evaluation"""
        self.env._reset()
        state = self.env.get_state()
        
        episode_log = []
        episode_steps = 0
        done = False
        start_time = time.time()
        
        
        
        while not done and episode_steps < 100:  # Safety limit to prevent infinite loops
            # Get valid actions
            candidates = self.env.valid_clique_merges()
            
            if not candidates:
                break  # No valid actions available
            
            # Agent selects action
            action = self.agent.choose_action(state, candidates)
            
            if action is None:
                break  # Agent couldn't choose an action
            
            # Store pre-merge state
            episode_log.append({
                'state': state.copy(),  # Make a copy to prevent reference issues
                'action': action,
                'reward': 0  # Will be updated later
            })
            
            # Perform action
            try:
                next_state, done = self.env.step(action)
                
                if render:
                    self.env.render()  # Visualize if needed
                
                state = next_state
                episode_steps += 1
                
            except Exception as e:
                print(f"Error during environment step: {e}")
                break
        
        episode_time = time.time() - start_time
        
        # Calculate final reward
        try:
            final_reward = self.env.generate_reward() * self.reward_scaling
            
            if training:
                # Process episode for learning
                self._process_episode(episode_log, final_reward)
                
                # Update performance tracking
                self.rewards_history.append(final_reward)
                self.episode_lengths.append(episode_steps)
                
                # Save best model
                if final_reward > self.best_reward:
                    self.best_reward = final_reward
                    self.save_agent("best_model")
                
            # Log episode details
            print(f"Episode completed: {episode_steps} steps, reward={final_reward:.4f}, time={episode_time:.2f}s")
            
            return final_reward, episode_steps
            
        except Exception as e:
            print(f"Error calculating reward: {e}")
            return 0.0, episode_steps
    
    def _process_episode(self, episode_log, final_reward):
        """Process episode rewards with proper temporal credit assignment"""
        if not episode_log:
            return
            
        # Calculate step-by-step rewards with proper discounting
        rewards = []
        discounted_reward = 0
        
        # Work backwards through episode
        for step in reversed(episode_log):
            # Apply discounting: discount * future_reward + current_reward
            # For terminal state, only final_reward applies
            if not rewards:  # First iteration (terminal state)
                discounted_reward = final_reward
            else:
                discounted_reward = final_reward + self.gamma * discounted_reward
                
            # Update step with calculated reward
            step['reward'] = discounted_reward
            rewards.append(discounted_reward)
            
            # Add experience to buffer with next state info
            idx = episode_log.index(step)
            next_state = episode_log[idx+1]['state'] if idx < len(episode_log)-1 else None
            
            # Store in agent memory and episode buffer
            self.episode_buffer.append((
                step['state'],
                step['action'],
                step['reward'],
                next_state
            ))
    
    def train(self, num_episodes=1000, eval_interval=10, warmup_episodes=0):
        """Train the agent with experience replay and periodic evaluation"""
        self.training_start_time = time.time()

        print(f"Running simulation for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            # Run episode and collect experience
            episode_reward, steps = self.run_episode(training=True)
            
            # Skip learning during warmup phase to collect initial experiences
            if episode < warmup_episodes:
                continue
                
            # Experience replay learning
            if len(self.episode_buffer) >= self.batch_size:
                for _ in range(max(1, steps // 10)):  # Learn multiple times for longer episodes
                    batch = random.sample(list(self.episode_buffer), self.batch_size)
                    self.agent.learn(batch)
            
            # Decay exploration rate
            self.agent.decay_epsilon()
            
            # Periodic evaluation without exploration
            if (episode + 1) % eval_interval == 0:
                self._evaluate_agent(3)  # Run 3 evaluation episodes
                
            """# Save checkpoint
            if (episode + 1) % self.save_interval == 0:
                self.save_agent(f"checkpoint_{episode+1}")
                self._save_training_progress(episode+1)"""
                
            # Print progress
            if (episode + 1) % 10 == 0:
                self._print_training_stats(episode+1)
        
        # Final evaluation and saving
        final_performance = self._evaluate_agent(5)
        self.save_agent("final_model")
        
        total_time = time.time() - self.training_start_time
        print(f"Training completed: {num_episodes} episodes in {total_time:.2f}s")
        print(f"Final performance: {final_performance:.4f}")
        
        return self.rewards_history
    
    def _evaluate_agent(self, num_evals=3):
        """Evaluate agent performance without exploration"""
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0  # Disable exploration for evaluation
        
        eval_rewards = []
        for _ in range(num_evals):
            reward, _ = self.run_episode(training=False)
            eval_rewards.append(reward)
            
        self.agent.epsilon = original_epsilon  # Restore exploration
        
        avg_reward = np.mean(eval_rewards)
        print(f"Evaluation: avg_reward={avg_reward:.4f}")
        return avg_reward
    
    def _print_training_stats(self, episode):
        """Print training statistics"""
        recent_rewards = self.rewards_history[-10:]
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0
        avg_length = np.mean(self.episode_lengths[-10:]) if self.episode_lengths else 0
        
        elapsed_time = time.time() - self.training_start_time
        
        print(f"Episode {episode}: avg_reward={avg_reward:.4f}, avg_steps={avg_length:.1f}, " 
              f"epsilon={self.agent.epsilon:.3f}, time={elapsed_time:.1f}s")
    
    def save_agent(self, name):
        """Save agent model"""
        try:
            self.agent.save_model(f"{self.checkpoint_path}/{name}.npz")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def _save_training_progress(self, episode):
        """Save training progress metrics"""
        try:
            np.savez(
                f"{self.checkpoint_path}/progress_{episode}.npz",
                rewards=np.array(self.rewards_history),
                lengths=np.array(self.episode_lengths),
                best_reward=self.best_reward,
                episode=episode
            )
        except Exception as e:
            print(f"Error saving progress: {e}")
    
    def load_progress(self, filename):
        """Load training progress"""
        try:
            data = np.load(filename, allow_pickle=True)
            self.rewards_history = data['rewards'].tolist()
            self.episode_lengths = data['lengths'].tolist()
            self.best_reward = float(data['best_reward'])
            return int(data['episode'])
        except Exception as e:
            print(f"Error loading progress: {e}")
            return 0