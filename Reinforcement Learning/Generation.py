import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import pandas as pd
from collections import Counter
import time

class AgentAnalyzer:
    """
    A unified class for analyzing reinforcement learning agent behavior
    on graph modification tasks, with a focus on chordal graphs.
    """
    
    def __init__(self, agent, env, trainer):
        """
        Initialize the analyzer with agent, environment and trainer components.
        
        Args:
            agent: The RL agent to analyze
            env: The environment the agent interacts with
            trainer: The training manager that tracks progress
        """
        self.agent = agent
        self.env = env
        self.trainer = trainer
        
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
    
    def run_simulation(self, num_episodes=50, force_rerun=False):
        """
        Run simulation to collect agent behavior data.
        
        Args:
            num_episodes: Number of episodes to simulate
            force_rerun: Whether to force a new simulation even if cached data exists
            
        Returns:
            Dictionary containing all collected simulation data
        """
        if self.cache['simulation_data'] is not None and not force_rerun:
            print("Using cached simulation data. Set force_rerun=True to regenerate.")
            return self.cache['simulation_data']
            
        print(f"Running simulation for {num_episodes} episodes...")
        start_time = time.time()
        
        # Data structures for collecting metrics
        episode_data = []
        
        for episode in range(num_episodes):
            self.env._reset()
            state = self.env.get_state()
            done = False
            steps = 0
            
            # Track episode metrics
            episode_metrics = {
                'states': [state.copy()],
                'actions': [],
                'clique_sizes': [],
                'state_metrics': []
            }
            
            while not done and steps < 100:
                candidates = self.env.valid_clique_merges()
                if not candidates:
                    break
                    
                action = self.agent.choose_action(state, candidates)
                if action is None:
                    break
                
                # Record action details
                clique1_size = len(action[0])
                clique2_size = len(action[1])
                episode_metrics['clique_sizes'].append((clique1_size, clique2_size))
                episode_metrics['actions'].append(action)
                
                # Take step
                next_state, done = self.env.step(action)
                
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
                steps += 1
            
            episode_metrics['success'] = done
            episode_metrics['steps'] = steps
            episode_data.append(episode_metrics)
            
            if (episode + 1) % 10 == 0:
                print(f"Processed {episode + 1}/{num_episodes} episodes...")
        
        # Compile statistics
        success_rate = sum(1 for ep in episode_data if ep['success']) / len(episode_data)
        avg_steps = np.mean([ep['steps'] for ep in episode_data])
        
        simulation_data = {
            'episode_data': episode_data,
            'success_rate': success_rate,
            'avg_steps': avg_steps,
            'num_episodes': num_episodes,
            'timestamp': time.time()
        }
        
        # Cache the results
        self.cache['simulation_data'] = simulation_data
        
        print(f"Simulation completed in {time.time() - start_time:.2f}s")
        print(f"Success rate: {success_rate*100:.1f}%, Average steps: {avg_steps:.2f}")
        
        return simulation_data
    
    def analyze_feature_importance(self, visualize=True, force_rerun=False):
        """
        Analyze which features have the strongest influence on agent decisions.
        
        Args:
            visualize: Whether to create visualization
            force_rerun: Whether to force a new analysis
            
        Returns:
            DataFrame of feature importance values
        """
        if self.cache['feature_importance'] is not None and not force_rerun:
            importance = self.cache['feature_importance']
        else:
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
    
    def analyze_action_patterns(self, visualize=True, force_rerun=False):
        """
        Analyze common patterns in agent's clique selection strategy.
        
        Args:
            visualize: Whether to create visualization
            force_rerun: Whether to force a new analysis
            
        Returns:
            Dictionary with action pattern analysis
        """
        # Use cached simulation data or run new simulation
        if self.cache['simulation_data'] is None or force_rerun:
            self.run_simulation(force_rerun=force_rerun)
        
        if self.cache['action_patterns'] is not None and not force_rerun:
            patterns = self.cache['action_patterns']
        else:
            episode_data = self.cache['simulation_data']['episode_data']
            
            # Collect all clique sizes across episodes
            all_clique_sizes = []
            all_strategies = []
            
            for episode in episode_data:
                clique_sizes = episode['clique_sizes']
                all_clique_sizes.extend(clique_sizes)
                
                # Determine merge strategies
                for clique1_size, clique2_size in clique_sizes:
                    size_diff = abs(clique1_size - clique2_size)
                    if size_diff <= 1:
                        strategy = "similar_sized"
                    elif size_diff <= 3:
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
        
        if visualize and patterns['clique_sizes']:
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
        rewards = self.trainer.rewards_history
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
    
    def analyze_all(self, num_episodes=50, visualize=True, force_rerun=False):
        """
        Run all analyses and return comprehensive results.
        
        Args:
            num_episodes: Number of episodes to simulate if needed
            visualize: Whether to create visualizations
            force_rerun: Whether to force new analyses
            
        Returns:
            Dictionary with all analysis results
        """
        # Run simulation first if needed
        if self.cache['simulation_data'] is None or force_rerun:
            self.run_simulation(num_episodes=num_episodes, force_rerun=force_rerun)
        
        # Run all analyses
        print("Analyzing feature importance...")
        importance = self.analyze_feature_importance(visualize=visualize, force_rerun=force_rerun)
        
        print("\nAnalyzing action patterns...")
        patterns = self.analyze_action_patterns(visualize=visualize, force_rerun=force_rerun)
        
        print("\nAnalyzing learning progress...")
        progress = self.analyze_learning_progress(visualize=visualize)
        
        print("\nAnalyzing state transitions...")
        transitions = self.analyze_state_transitions(visualize=visualize, force_rerun=force_rerun)
        
        return {
            "feature_importance": importance,
            "action_patterns": patterns,
            "learning_progress": progress,
            "state_transitions": transitions,
            "simulation_data": self.cache['simulation_data']
        }
    
    def summarize_findings(self):
        """
        Generate a text summary of key findings from all analyses.
        
        Returns:
            String with summary of key findings
        """
        # Ensure we have data for all analyses
        if any(v is None for v in self.cache.values()):
            missing = [k for k, v in self.cache.items() if v is None]
            print(f"Missing data for: {missing}. Running all analyses...")
            self.analyze_all(visualize=False)
        
        # Build summary
        summary = []
        
        # Simulation overview
        sim_data = self.cache['simulation_data']
        summary.append(f"## Agent Performance Summary\n")
        summary.append(f"Success rate: {sim_data['success_rate']*100:.1f}%")
        summary.append(f"Average steps to solution: {sim_data['avg_steps']:.2f}")
        
        # Feature importance
        importance = self.cache['feature_importance']
        if isinstance(importance, pd.DataFrame):
            top_features = importance.head(5)
            summary.append(f"\n## Top 5 Important Features\n")
            for i, (feature, imp) in enumerate(zip(top_features['Feature'], top_features['Importance'])):
                summary.append(f"{i+1}. {feature}: {imp:.4f}")
        
        # Action patterns
        patterns = self.cache['action_patterns']
        if isinstance(patterns, dict):
            summary.append(f"\n## Merge Strategy Preferences\n")
            strategies = patterns['merge_strategies']
            total = sum(strategies.values())
            for strategy, count in strategies.most_common():
                summary.append(f"{strategy}: {count/total*100:.1f}%")
            
            summary.append(f"\nAverage clique sizes: {patterns['avg_clique1_size']:.2f}, "
                          f"{patterns['avg_clique2_size']:.2f}")
        
        # Learning progress
        progress = self.cache['learning_progress']
        if isinstance(progress, dict) and progress.get('improvement_rate') is not None:
            summary.append(f"\n## Learning Progress\n")
            summary.append(f"Improvement rate: {progress['improvement_rate']:.4f}")
            summary.append(f"Initial performance: {progress['initial_performance']:.4f}")
            summary.append(f"Final performance: {progress['final_performance']:.4f}")
            summary.append(f"Best performance: {progress['best_performance']:.4f}")
        
        # State transitions
        transitions = self.cache['state_transitions']
        if isinstance(transitions, dict) and 'avg_evolution' in transitions:
            summary.append(f"\n## Graph Transformation Patterns\n")
            avg_evolution = transitions['avg_evolution']
            density_change = avg_evolution['density'][-1] - avg_evolution['density'][0]
            clustering_change = avg_evolution['clustering'][-1] - avg_evolution['clustering'][0]
            clique_change = avg_evolution['max_clique_size'][-1] - avg_evolution['max_clique_size'][0]
            
            summary.append(f"Density change: {density_change:.4f}")
            summary.append(f"Clustering change: {clustering_change:.4f}")
            summary.append(f"Max clique size change: {clique_change:.4f}")
        
        return "\n".join(summary)