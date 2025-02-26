import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class AgentStrategyAnalyzer:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.action_history = []
        self.state_history = []
        self.reward_history = []
        self.feature_importance = {}
        self.merge_patterns = defaultdict(int)
        
    def record_action(self, state, action, reward):
        """Record an action taken by the agent for later analysis"""
        self.state_history.append(state.copy())
        self.action_history.append(action)
        self.reward_history.append(reward)
        
        # Track merge sizes
        if action:
            clique1_size = len(action[0])
            clique2_size = len(action[1])
            self.merge_patterns[(clique1_size, clique2_size)] += 1
    
    def analyze_agent(self, num_episodes=10):
        """Run episodes and analyze agent behavior"""
        print("Running analysis episodes...")
        
        # Clear previous history
        self.action_history = []
        self.state_history = []
        self.reward_history = []
        self.merge_patterns = defaultdict(int)
        
        # Set agent to evaluation mode (no exploration)
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0
        
        episode_rewards = []
        
        # Run episodes
        for ep in range(num_episodes):
            self.env._reset()
            state = self.env.get_state()
            done = False
            ep_reward = 0
            
            step_count = 0
            while not done and step_count < 1000:
                # Get valid actions
                candidates = self.env.valid_clique_merges()
                if not candidates:
                    break
                
                # Agent selects action
                action = self.agent.choose_action(state, candidates)
                if action is None:
                    break
                
                # Record action
                self.record_action(state, action, ep_reward)
                
                # Execute action
                next_state, done = self.env.step(action)
                state = next_state
                step_count += 1
            
            # Calculate final reward
            final_reward = self.env.generate_reward()
            episode_rewards.append(final_reward)
            print(f"Episode {ep+1}/{num_episodes} - Reward: {final_reward:.4f}, Steps: {step_count}")
        
        # Restore exploration
        self.agent.epsilon = original_epsilon
        
        # Run analyses
        results = {
            "episode_rewards": episode_rewards,
            "merge_patterns": self.analyze_merge_patterns(),
            "feature_importance": self.analyze_feature_importance(),
            "state_clusters": self.cluster_states(),
            "transition_patterns": self.analyze_state_transitions()
        }
        
        return results
    
    def analyze_merge_patterns(self):
        """Analyze patterns in clique merge sizes"""
        print("\nAnalyzing merge patterns...")
        
        # Convert to DataFrame for easier analysis
        merge_data = []
        for (size1, size2), count in self.merge_patterns.items():
            merge_data.append({
                'clique1_size': size1,
                'clique2_size': size2,
                'total_size': size1 + size2,
                'size_diff': abs(size1 - size2),
                'count': count,
                'frequency': count / sum(self.merge_patterns.values())
            })
        
        df = pd.DataFrame(merge_data)
        
        # Calculate statistics
        if not df.empty:
            avg_size1 = df['clique1_size'].mean()
            avg_size2 = df['clique2_size'].mean()
            avg_total = df['total_size'].mean()
            avg_diff = df['size_diff'].mean()
            
            # Preferred merge sizes
            most_common = df.sort_values('count', ascending=False).head(5)
            
            print(f"Average clique sizes in merges: {avg_size1:.2f} and {avg_size2:.2f}")
            print(f"Average total size: {avg_total:.2f}, Average size difference: {avg_diff:.2f}")
            print("Most common merge patterns:")
            print(most_common[['clique1_size', 'clique2_size', 'count', 'frequency']])
            
            # Plot merge patterns
            self._plot_merge_patterns(df)
            
            return {
                "dataframe": df,
                "avg_size1": avg_size1,
                "avg_size2": avg_size2,
                "avg_total": avg_total,
                "avg_diff": avg_diff,
                "most_common": most_common
            }
        else:
            print("No merge patterns recorded.")
            return None
    
    def _plot_merge_patterns(self, df):
        """Plot merge patterns"""
        plt.figure(figsize=(10, 8))
        
        # Create a pivot table for heatmap
        if len(df) > 1:
            sizes = list(range(1, max(df['clique1_size'].max(), df['clique2_size'].max()) + 1))
            pivot_data = np.zeros((len(sizes), len(sizes)))
            
            for _, row in df.iterrows():
                i, j = int(row['clique1_size'])-1, int(row['clique2_size'])-1
                if i < len(sizes) and j < len(sizes):
                    pivot_data[i, j] = row['count']
            
            # Plot heatmap
            sns.heatmap(pivot_data, annot=True, fmt='g', 
                        xticklabels=sizes, yticklabels=sizes, 
                        cmap='viridis')
            plt.title('Clique Merge Patterns')
            plt.xlabel('Clique 2 Size')
            plt.ylabel('Clique 1 Size')
            plt.savefig('merge_patterns.png')
            plt.close()
            
            # Plot size relationship
            plt.figure(figsize=(8, 6))
            plt.scatter(df['clique1_size'], df['clique2_size'], 
                        s=df['count']*20, alpha=0.6)
            plt.title('Clique Size Relationships')
            plt.xlabel('Clique 1 Size')
            plt.ylabel('Clique 2 Size')
            plt.grid(True)
            plt.savefig('clique_size_relationship.png')
            plt.close()
    
    def analyze_feature_importance(self):
        """Analyze which graph features correlate with higher rewards"""
        print("\nAnalyzing feature importance...")
        
        # Extract features from states
        features = []
        for state in self.state_history:
            feature_vec = {
                'num_vertices': state.get('num_vertices', 0),
                'num_edges': state.get('num_edges', 0),
                'max_degree': state.get('max_degree', 0),
                'min_degree': state.get('min_degree', 0),
                'mean_degree': state.get('mean_degree', 0),
                'max_clique_size': state.get('max_clique_size', 0),
                'avg_clique_size': state.get('avg_clique_size', 0),
                'clique_size_std': state.get('clique_size_std', 0),
                'average_clustering': state.get('average_clustering', 0),
                'global_clustering': state.get('global_clustering', 0),
                'density': state.get('density', 0),
                'diameter': state.get('diameter', 0),
                'radius': state.get('radius', 0)
            }
            features.append(feature_vec)
        
        if not features:
            print("No state features recorded.")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(features)
        
        # Calculate correlation with rewards
        df['reward'] = self.reward_history
        correlations = df.corr()['reward'].sort_values(ascending=False)
        
        # Display top correlations
        print("Feature correlations with reward:")
        print(correlations)
        
        # Store feature importance
        self.feature_importance = correlations.to_dict()
        
        # Plot feature importance
        self._plot_feature_importance(correlations)
        
        return {
            "correlations": correlations,
            "dataframe": df
        }
    
    def _plot_feature_importance(self, correlations):
        """Plot feature importance"""
        # Remove 'reward' from correlations
        correlations = correlations.drop('reward', errors='ignore')
        
        plt.figure(figsize=(10, 6))
        correlations.plot(kind='bar')
        plt.title('Feature Correlation with Reward')
        plt.xlabel('Features')
        plt.ylabel('Correlation')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
    
    def cluster_states(self):
        """Cluster states to identify patterns"""
        print("\nClustering states to identify patterns...")
        
        # Extract features from states
        features = []
        for state in self.state_history:
            feature_vec = [
                state.get('num_vertices', 0),
                state.get('num_edges', 0),
                state.get('max_degree', 0),
                state.get('min_degree', 0),
                state.get('mean_degree', 0),
                state.get('max_clique_size', 0),
                state.get('avg_clique_size', 0),
                state.get('clique_size_std', 0),
                state.get('average_clustering', 0),
                state.get('global_clustering', 0),
                state.get('density', 0)
            ]
            features.append(feature_vec)
        
        if not features or len(features) < 5:
            print("Not enough state data for clustering.")
            return None
        
        # Normalize data
        X = StandardScaler().fit_transform(features)
        
        # Use PCA to reduce dimensions
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Determine optimal number of clusters
        n_clusters = min(5, len(X) // 5)
        if n_clusters < 2:
            n_clusters = 2
            
        # Apply clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)
        
        # Calculate cluster statistics
        cluster_data = []
        for i in range(n_clusters):
            cluster_indices = np.where(clusters == i)[0]
            cluster_rewards = [self.reward_history[idx] for idx in cluster_indices]
            
            if cluster_rewards:
                avg_reward = np.mean(cluster_rewards)
                cluster_size = len(cluster_indices)
                
                cluster_data.append({
                    'cluster': i,
                    'size': cluster_size,
                    'avg_reward': avg_reward,
                    'percentage': cluster_size / len(clusters) * 100
                })
        
        # Plot clusters
        self._plot_state_clusters(X_pca, clusters, self.reward_history)
        
        cluster_df = pd.DataFrame(cluster_data).sort_values('avg_reward', ascending=False)
        print("Cluster statistics:")
        print(cluster_df)
        
        return {
            "pca_components": X_pca,
            "clusters": clusters, 
            "cluster_stats": cluster_df
        }
    
    def _plot_state_clusters(self, X_pca, clusters, rewards):
        """Plot clustered states"""
        plt.figure(figsize=(10, 8))
        
        # Normalize rewards for coloring
        norm_rewards = np.array(rewards)
        if len(norm_rewards) > 0 and np.std(norm_rewards) != 0:
            norm_rewards = (norm_rewards - np.min(norm_rewards)) / (np.max(norm_rewards) - np.min(norm_rewards))
        
        # Plot clusters with reward as color intensity
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                             c=clusters, cmap='viridis', 
                             alpha=0.6, s=50 + 100 * norm_rewards)
        
        plt.colorbar(scatter, label='Cluster')
        plt.title('State Clusters with Reward Intensity')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('state_clusters.png')
        plt.close()
    
    def analyze_state_transitions(self):
        """Analyze how the agent transitions between states"""
        print("\nAnalyzing state transitions...")
        
        if len(self.state_history) < 2:
            print("Not enough state data for transition analysis.")
            return None
        
        # Calculate state transitions based on key metrics
        transitions = []
        
        for i in range(len(self.state_history)-1):
            current = self.state_history[i]
            next_state = self.state_history[i+1]
            
            # Calculate key changes
            vertex_change = next_state.get('num_vertices', 0) - current.get('num_vertices', 0)
            edge_change = next_state.get('num_edges', 0) - current.get('num_edges', 0)
            density_change = next_state.get('density', 0) - current.get('density', 0)
            clustering_change = next_state.get('average_clustering', 0) - current.get('average_clustering', 0)
            max_clique_change = next_state.get('max_clique_size', 0) - current.get('max_clique_size', 0)
            
            transitions.append({
                'vertex_change': vertex_change,
                'edge_change': edge_change, 
                'density_change': density_change,
                'clustering_change': clustering_change,
                'max_clique_change': max_clique_change,
                'reward': self.reward_history[i]
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(transitions)
        
        # Calculate statistics
        avg_transitions = df.mean()
        print("Average transitions:")
        print(avg_transitions)
        
        # Plot transitions
        self._plot_transitions(df)
        
        return {
            "transition_df": df,
            "avg_transitions": avg_transitions 
        }
    
    def _plot_transitions(self, df):
        """Plot state transitions"""
        plt.figure(figsize=(12, 8))
        
        # Plot transition distributions
        plt.subplot(2, 2, 1)
        sns.histplot(df['vertex_change'], kde=True)
        plt.title('Vertex Changes')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        sns.histplot(df['edge_change'], kde=True)
        plt.title('Edge Changes')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        sns.histplot(df['density_change'], kde=True)
        plt.title('Density Changes')
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        sns.histplot(df['max_clique_change'], kde=True)
        plt.title('Max Clique Size Changes')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('state_transitions.png')
        plt.close()
        
        # Plot correlation between changes and rewards
        plt.figure(figsize=(10, 6))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Between State Changes and Rewards')
        plt.tight_layout()
        plt.savefig('transition_reward_correlation.png')
        plt.close()

    def run_analysis_report(self, num_episodes=10, save_file='agent_analysis_report.txt'):
        """Run full analysis and save report"""
        print(f"Running complete agent analysis over {num_episodes} episodes...")
        
        results = self.analyze_agent(num_episodes)
        
        # Save comprehensive report
        with open(save_file, 'w') as f:
            f.write("=== AGENT STRATEGY ANALYSIS REPORT ===\n\n")
            
            # Episode rewards
            f.write("1. EPISODE REWARDS\n")
            f.write("-----------------\n")
            rewards = results["episode_rewards"]
            f.write(f"Mean reward: {np.mean(rewards):.4f}\n")
            f.write(f"Std deviation: {np.std(rewards):.4f}\n")
            f.write(f"Min reward: {np.min(rewards):.4f}\n")
            f.write(f"Max reward: {np.max(rewards):.4f}\n\n")
            
            # Merge patterns
            f.write("2. MERGE PATTERNS\n")
            f.write("----------------\n")
            merge_results = results["merge_patterns"]
            if merge_results:
                f.write(f"Average clique sizes: {merge_results['avg_size1']:.2f} and {merge_results['avg_size2']:.2f}\n")
                f.write(f"Average total size: {merge_results['avg_total']:.2f}\n")
                f.write(f"Average size difference: {merge_results['avg_diff']:.2f}\n")
                f.write("Most common merge patterns:\n")
                for _, row in merge_results["most_common"].iterrows():
                    f.write(f"  Clique1: {int(row['clique1_size'])}, Clique2: {int(row['clique2_size'])}, " 
                            f"Count: {int(row['count'])}, Frequency: {row['frequency']:.2f}\n")
            f.write("\n")
            
            # Feature importance
            f.write("3. FEATURE IMPORTANCE\n")
            f.write("--------------------\n")
            if results["feature_importance"] and "correlations" in results["feature_importance"]:
                correlations = results["feature_importance"]["correlations"].drop('reward', errors='ignore')
                for feature, corr in correlations.items():
                    f.write(f"{feature}: {corr:.4f}\n")
            f.write("\n")
            
            # State clusters
            f.write("4. STATE CLUSTERS\n")
            f.write("----------------\n")
            if results["state_clusters"] and "cluster_stats" in results["state_clusters"]:
                cluster_stats = results["state_clusters"]["cluster_stats"]
                for _, row in cluster_stats.iterrows():
                    f.write(f"Cluster {int(row['cluster'])}: {int(row['size'])} states, "
                            f"{row['percentage']:.1f}% of total, "
                            f"Average reward: {row['avg_reward']:.4f}\n")
            f.write("\n")
            
            # Transition patterns
            f.write("5. STATE TRANSITIONS\n")
            f.write("-------------------\n")
            if results["transition_patterns"] and "avg_transitions" in results["transition_patterns"]:
                avg_trans = results["transition_patterns"]["avg_transitions"]
                for metric, value in avg_trans.items():
                    f.write(f"Average {metric}: {value:.4f}\n")
                            
            f.write("\n=== END OF REPORT ===\n")
            
        print(f"Analysis complete! Report saved to {save_file}")
        print("Visualization images saved to current directory.")
        
        return results

if __name__ == "__main__":
    import Enviroment
    import Agents

    env = Enviroment.ChordalGraphEnv()
    agent = Agents.KNNGraphAgent()
    analyzer = AgentStrategyAnalyzer(agent, env)
    results = analyzer.run_analysis_report(10)