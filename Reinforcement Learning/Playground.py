from collections import deque
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

class ChordalPlayground():
    def __init__(self, agent, env, 
                 gamma=0.95, buffer_size=10000, batch_size=32, reward_scaling=1.0,
                 analysis = True):
        #Intialise components
        self.agent = agent
        self.env = env

        #RF parameters
        self.gamma = gamma             # Discount factor
        self.episode_buffer = deque(maxlen=buffer_size)  # Experience replay buffer
        self.batch_size = batch_size   # Training batch size
        self.reward_scaling = reward_scaling  # Scale rewards for numerical stability

        self.rewards_history = []

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

    def run_episode(self):
        self.env._reset()

        state = self.env._get_state()

        done = False
        episode_log = []
        episode_steps = 0

        start_time = time.time()

        # Track episode metrics
        episode_metrics = {
            'states': [state.copy()],
            'actions': [],
            'clique_sizes': [],
            'state_metrics': []
        }

        while not done and episode_steps < 100:
            action_space = self.env.valid_clique_merges()
            if not action_space:
                break #No valid merges found

            action = self.agent.choose_action(state, action_space)
            if not action:
                break #Did not select an action

            # Store pre-merge state
            episode_log.append({
                'state': state.copy(),  # Make a copy to prevent reference issues
                'action': action,
                'reward': 0  # Will be updated later
            })

            # Perform action
            try:
                next_state, done = self.env.step(action)
                
                 # Record action details
                clique1_size = len(action[0])
                clique2_size = len(action[1])
                episode_metrics['clique_sizes'].append((clique1_size, clique2_size))
                episode_metrics['actions'].append(action)

                # Record state metrics
                state_metrics = {
                    'density': next_state['density'],
                    'clustering': next_state['global_clustering'],
                    'max_clique_size': next_state['max_clique_size'],
                    'num_edges': next_state['num_edges'],
                    'is_terminal': done
                }
                episode_metrics['state_metrics'].append(state_metrics)
                episode_metrics['states'].append(next_state.copy())

                state = next_state
                episode_steps += 1
                
            except Exception as e:
                print(f"Error during environment step: {e}")
                break
        episode_metrics['success'] = done
        episode_metrics['steps'] = episode_steps
        

        episode_time = time.time() - start_time

        try: #Give reward to agent
            final_reward = self.env.generate_reward() * self.reward_scaling
            self.rewards_history.append(final_reward)
            self._process_episode(episode_log, final_reward)
        except Exception as e:
            print(f"Error calculating reward: {e}")
        
        return episode_steps, episode_metrics

    def train(self, num_episodes=100, warmup_episodes=10):
        self.training_start_time = time.time()

        episode_data = []
        for episode in range(num_episodes):
            steps, episode_metrics = self.run_episode()
            
            # Skip learning during warmup phase to collect initial experiences
            if episode < warmup_episodes:
                continue

            episode_data.append(episode_metrics)
            # Experience replay learning
            if len(self.episode_buffer) >= self.batch_size:
                for _ in range(max(1, steps // 10)):  # Learn multiple times for longer episodes
                    batch = random.sample(list(self.episode_buffer), self.batch_size)
                    self.agent.learn(batch)
                # Decay exploration rate
                self.agent.decay_epsilon()
        
        avg_steps = np.mean([ep['steps'] for ep in episode_data])
        simulation_data = {
            'episode_data': episode_data,
            'avg_steps': avg_steps,
            'num_episodes': num_episodes,
            'timestamp': time.time()
        }
        
        # Cache the results
        self.cache['simulation_data'] = simulation_data
        

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

    def analyze_feature_importance(self, visualize=True):
        """
        Analyze which features have the strongest influence on agent decisions.
        
        Args:
            visualize: Whether to create visualization
            force_rerun: Whether to force a new analysis
            
        Returns:
            DataFrame of feature importance values
        """
        if not self.agent.memory or len(self.agent.memory) < 10:
            return "Not enough data in agent memory for feature importance analysis"
            
        # Extract features and Q-values from agent memory
        X = np.array([state_feat for state_feat, _, _, _, _ in self.agent.memory])
        y = np.array([q_val for _, _, _, _, q_val in self.agent.memory])
            
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
            
        # Use Lasso regression for feature importance
        lasso = Lasso(alpha=0.01)
        lasso.fit(X_scaled, y)
            
        # Create feature importance dataframe
        importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': np.abs(lasso.coef_)
        }).sort_values('Importance', ascending=False)
            
        # Cache the results
        self.cache['feature_importance'] = importance
        
        if visualize:
            plt.figure(figsize=(10, 6))
            top_features = importance.head(10)
            plt.barh(top_features['Feature'], top_features['Importance'])
            plt.xlabel('Importance')
            plt.title('Top Features Influencing Agent Decisions')
            plt.tight_layout()
            plt.show()
        
        return importance
    
    def analyze_action_patterns(self, visualise=True):
        episode_data = self.cache['simulation_data']['episode_data']
            
        # Collect all clique sizes across episodes
        all_clique_sizes = []
        all_strategies = []
        for episode in episode_data:
            clique_sizes = episode['clique_sizes']
            all_clique_sizes.extend(clique_sizes)
                
            # Determine merge strategies
            for clique1_size, clique2_size in clique_sizes:
                size_diff = abs(clique1_size-clique2_size)
                if size_diff <= 3:
                    strategy = "similar_sized"
                elif size_diff <= 5:
                    strategy = "moderate_difference"
                else:
                    strategy = "large_difference"
                all_strategies.append(strategy)
            
            # Calculate statistics
            avg_clique1 = np.mean([c[0] for c in all_clique_sizes]) if all_clique_sizes else 0
            avg_clique2 = np.mean([c[1] for c in all_clique_sizes]) if all_clique_sizes else 0
            strategy_counts = Counter(all_strategies)
            
            patterns = {
                "avg_clique1_size": avg_clique1,
                "avg_clique2_size": avg_clique2,
                "merge_strategies": strategy_counts,
                "clique_sizes": all_clique_sizes
            }
            
            # Cache results
            self.cache['action_patterns'] = patterns
        
        if visualise and patterns['clique_sizes']:
            # Plot merge strategies
            strategies = patterns["merge_strategies"]
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.bar(strategies.keys(), strategies.values())
            plt.title("Preferred Merge Strategies")
            plt.ylabel("Frequency")
            
            # Plot clique size distribution
            plt.subplot(1, 2, 2)
            clique_sizes = patterns["clique_sizes"]
            clique1_sizes = [c[0] for c in clique_sizes]
            clique2_sizes = [c[1] for c in clique_sizes]
            
            plt.hist([clique1_sizes, clique2_sizes], bins=range(1, 10), 
                     label=['Clique 1', 'Clique 2'], alpha=0.7)
            plt.title("Clique Size Distribution")
            plt.xlabel("Size")
            plt.ylabel("Frequency")
            plt.legend()
            
            plt.tight_layout()
            plt.show()
        
        return patterns
    
    def analyze_learning_progress(self, window_size=10, visualize=True):
        """
        Analyze how the agent's strategy evolves during training.
        
        Args:
            window_size: Size of the moving average window
            visualize: Whether to create visualization
            
        Returns:
            Dictionary with learning progress metrics
        """
        rewards = self.rewards_history
        if not rewards:
            return "No training data available"
        
        # Calculate moving averages
        windows = []
        for i in range(len(rewards) - window_size + 1):
            windows.append(np.mean(rewards[i:i+window_size]))
        
        # Calculate improvement rate
        if len(windows) > 1:
            improvement_rate = (windows[-1] - windows[0]) / len(windows)
        else:
            improvement_rate = 0
        
        progress = {
            "improvement_rate": improvement_rate,
            "final_performance": windows[-1] if windows else None,
            "initial_performance": windows[0] if windows else None,
            "best_performance": max(windows) if windows else None,
            "window_size": window_size,
            "rewards": rewards,
            "moving_averages": windows
        }
        
        # Cache results
        self.cache['learning_progress'] = progress
        
        if visualize and windows:
            plt.figure(figsize=(10, 6))
            plt.plot(range(window_size, len(rewards) + 1), windows)
            plt.axhline(y=np.mean(rewards), color='r', linestyle='-', alpha=0.3, label='Average')
            plt.title(f"Learning Progress (Window Size: {window_size})")
            plt.xlabel("Episode")
            plt.ylabel("Average Reward")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
        
        return progress
    
    def analyze_state_transitions(self, visualize=True, force_rerun=False):
        """
        Analyze how the agent transforms the graph toward chordality.
        
        Args:
            visualize: Whether to create visualization
            force_rerun: Whether to force a new analysis
            
        Returns:
            Dictionary with state transition metrics
        """
        # Use cached simulation data or run new simulation
        if self.cache['simulation_data'] is None or force_rerun:
            self.run_simulation(force_rerun=force_rerun)
            
        if self.cache['state_transitions'] is not None and not force_rerun:
            transitions = self.cache['state_transitions']
        else:
            episode_data = self.cache['simulation_data']['episode_data']
            
            # Extract successful episodes only for more meaningful analysis
            successful_episodes = [ep for ep in episode_data if ep['success']]
            
            if not successful_episodes:
                return "No successful episodes found for analysis"
            
            # Track metrics evolution
            metrics_evolution = {
                'density': [],
                'clustering': [],
                'max_clique_size': [],
                'num_edges': []
            }
            
            for episode in successful_episodes:
                metrics = episode['state_metrics']
                
                # Start with initial state
                episode_metrics = {
                    'density': [episode['states'][0]['density']],
                    'clustering': [episode['states'][0]['global_clustering']],
                    'max_clique_size': [episode['states'][0]['max_clique_size']],
                    'num_edges': [episode['states'][0]['num_edges']]
                }
                
                # Add metrics for each step
                for step_metrics in metrics:
                    for key in metrics_evolution.keys():
                        episode_metrics[key].append(step_metrics[key])
                
                # Add this episode's metrics to overall collection
                for key in metrics_evolution:
                    metrics_evolution[key].append(episode_metrics[key])
            
            # Average metric evolution across episodes
            avg_evolution = {}
            max_steps = max([len(seq) for seq in metrics_evolution['density']])
            
            for metric in metrics_evolution:
                # Pad sequences to same length
                padded_data = [seq + [seq[-1]]*(max_steps-len(seq)) 
                              for seq in metrics_evolution[metric]]
                avg_evolution[metric] = np.mean(padded_data, axis=0)
            
            transitions = {
                'avg_evolution': avg_evolution,
                'successful_episodes': len(successful_episodes),
                'max_steps': max_steps,
                'raw_metrics': metrics_evolution
            }
            
            # Cache results
            self.cache['state_transitions'] = transitions
        
        if visualize:
            avg_evolution = transitions['avg_evolution']
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.plot(avg_evolution['density'])
            plt.title('Graph Density Evolution')
            plt.xlabel('Steps')
            plt.ylabel('Density')
            
            plt.subplot(1, 3, 2)
            plt.plot(avg_evolution['clustering'])
            plt.title('Clustering Coefficient Evolution')
            plt.xlabel('Steps')
            
            plt.subplot(1, 3, 3)
            plt.plot(avg_evolution['max_clique_size'])
            plt.title('Max Clique Size Evolution')
            plt.xlabel('Steps')
            
            plt.tight_layout()
            plt.show()
        
        return transitions