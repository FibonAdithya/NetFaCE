import numpy as np
from collections import deque
from sklearn.neighbors import KDTree  # For efficient nearest neighbor search
import random

class KNNGraphAgent:
    def __init__(self, 
                 k=5, 
                 alpha=0.1, 
                 gamma=0.9, 
                 epsilon=0.8,
                 max_memory=1000,
                 min_epsilon=0.01,
                 epsilon_decay=0.995,
                 batch_size=32):
        """
        k: Number of nearest neighbors to consider
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration rate
        max_memory: Maximum stored experiences
        min_epsilon: Minimum exploration rate
        epsilon_decay: Rate at which epsilon decreases
        batch_size: Number of experiences to sample for batch learning
        """
        self.k = k
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_memory = max_memory
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Experience memory: (state_features, action, reward, next_state_features, q_value)
        self.memory = deque(maxlen=max_memory)
        
        # KDTree for efficient neighbor search
        self.kdtree = None
        self.state_features = []
        self.q_values = []
        
        # Performance tracking
        self.total_reward = 0
        self.episode_rewards = []

    def _normalize_features(self, features_vector):
        """Normalize features to improve KNN performance with NaN handling"""
        # Replace NaN values with zeros
        features_vector = np.nan_to_num(features_vector, nan=0.0)
        
        # Handle feature statistics for normalization
        if not hasattr(self, 'feature_stats'):
            self.feature_stats = {
                'mean': np.zeros(len(features_vector)),
                'std': np.ones(len(features_vector)),
                'samples': 0
            }
        
        # Update running statistics
        n = self.feature_stats['samples']
        if n > 0:
            # Incremental update of mean and std
            delta = features_vector - self.feature_stats['mean']
            self.feature_stats['mean'] += delta / (n + 1)
            delta2 = features_vector - self.feature_stats['mean']
            
            # Update variance estimate
            var_update = (self.feature_stats['std']**2 * n + delta * delta2) / (n + 1)
            # Ensure variance is non-negative
            var_update = np.maximum(var_update, 1e-8)
            self.feature_stats['std'] = np.sqrt(var_update)
        else:
            self.feature_stats['mean'] = features_vector
        
        self.feature_stats['samples'] += 1
        
        # Prevent division by zero
        std = np.maximum(self.feature_stats['std'], 1e-8)
        
        # Return normalized features
        normalized = (features_vector - self.feature_stats['mean']) / std
        
        # Final check to ensure no NaN values
        return np.nan_to_num(normalized, nan=0.0)

    def _features_to_state(self, features):
        """Convert raw state to normalized feature vector"""
        try:
            # Extract features and handle missing values
            feature_vector = np.array([
                float(features.get('num_vertices', 0)),
                float(features.get('num_edges', 0)),
                float(features.get('max_degree', 0)),
                float(features.get('min_degree', 0)),
                float(features.get('mean_degree', 0)),
                float(features.get('max_clique_size', 0)),
                float(features.get('avg_clique_size', 0)),
                float(features.get('clique_size_std', 0)),
                float(features.get('average_clustering', 0)),
                float(features.get('global_clustering', 0)),
                float(features.get('density', 0)),
                float(features.get('diameter', 0)),
                float(features.get('radius', 0)),
                float(features.get('clique1_size', 0)),
                float(features.get('clique2_size', 0))
            ], dtype=np.float64)
            
            # Check for infinity values and replace with large numbers
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1e6, neginf=-1e6)
            
            return self._normalize_features(feature_vector)
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            # Return a safe default vector
            return np.zeros(15, dtype=np.float64)

    def _update_kdtree(self):
        """Rebuild KDTree index after memory updates"""
        if len(self.state_features) >= self.k:
            try:
                # Convert to numpy array and ensure no NaN values
                features_array = np.array(self.state_features, dtype=np.float64)
                if np.isnan(features_array).any():
                    print("Warning: NaN values found in features, replacing with zeros")
                    features_array = np.nan_to_num(features_array, nan=0.0)
                
                self.kdtree = KDTree(features_array)
            except Exception as e:
                print(f"KDTree build failed: {e}")
                self.kdtree = None

    def get_q_value(self, state, action=None):
        """Estimate Q-value using k-NN weighted average"""
        try:
            state_vec = self._features_to_state(state)
            
            if self.kdtree is None or len(self.state_features) < self.k:
                return 0.0  # Initial estimate
            
            # Find k nearest neighbors
            k_neighbors = min(self.k, len(self.state_features))
            dists, indices = self.kdtree.query([state_vec], k=k_neighbors)
            
            # If distances are too large, we're in an unexplored state
            if np.mean(dists) > 10.0:  # Threshold for "too different"
                return 0.0
                
            # Get weighted average of Q-values (inverse distance weighting)
            weights = 1.0 / (dists[0] + 1e-6)  # Avoid division by zero
            weights /= weights.sum()
            
            neighbor_q_values = [self.q_values[i] for i in indices[0]]
            weighted_q = np.dot(weights, neighbor_q_values)
            
            return float(weighted_q)
        
        except Exception as e:
            print(f"Q-value estimation error: {e}")
            return 0.0

    def choose_action(self, state, merge_candidates):
        """ε-greedy action selection with candidate evaluation"""
        if not merge_candidates:
            return None
            
        if np.random.rand() < self.epsilon:
            return random.choice(merge_candidates)
            
        # Find best action based on current estimates
        best_action = None
        best_q_value = float('-inf')
        
        for candidate in merge_candidates:
            # Create candidate state representation
            candidate_state = {
                **state,
                'clique1_size': len(candidate[0]),
                'clique2_size': len(candidate[1])
            }
            
            q_value = self.get_q_value(candidate_state)
            
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = candidate
        
        # If we couldn't find a good action, explore randomly
        if best_action is None:
            return random.choice(merge_candidates)
                
        return best_action

    def store_experience(self, state, action, reward, next_state=None):
        """Store experience in memory with improved tracking"""
        try:
            state_features = self._features_to_state(state)
            
            # Track total reward
            self.total_reward += reward
            
            # If next_state not provided, use current state
            if next_state is None:
                next_state = state
                
            next_state_features = self._features_to_state(next_state)
            
            # Calculate target Q-value using Bellman equation
            next_q = self.get_q_value(next_state)
            target_q = reward + self.gamma * next_q
            
            # If we've seen similar states before, blend with existing Q-value
            if self.kdtree is not None and len(self.state_features) > 0:
                current_q = self.get_q_value(state)
                updated_q = current_q + self.alpha * (target_q - current_q)
            else:
                updated_q = reward  # Initial estimate based just on reward
            
            # Store values as native Python types to avoid numpy serialization issues
            self.memory.append((
                state_features.tolist(), 
                action, 
                float(reward), 
                next_state_features.tolist(), 
                float(updated_q)
            ))
            
            self.state_features.append(state_features.tolist())
            self.q_values.append(float(updated_q))
            
            # Update KDTree if necessary
            if len(self.state_features) % 10 == 0:  # Only rebuild periodically
                self._update_kdtree()
                
            return updated_q
            
        except Exception as e:
            print(f"Experience storage error: {e}")
            return 0.0
    
    def learn(self, batch=None):
        """Learn from batch of experiences or sample from memory"""
        if batch:
            # Learn from provided experiences
            for experience in batch:
                try:
                    if len(experience) == 3:
                        state, action, reward = experience
                        self.store_experience(state, action, reward)
                    elif len(experience) == 4:
                        state, action, reward, next_state = experience
                        self.store_experience(state, action, reward, next_state)
                except Exception as e:
                    print(f"Batch learning error: {e}")
        
        elif len(self.memory) >= self.batch_size:
            try:
                # Sample from memory for experience replay
                batch = random.sample(list(self.memory), self.batch_size)
                
                for state_feat, action, reward, next_state_feat, old_q in batch:
                    # Convert stored lists back to numpy arrays
                    state_feat_array = np.array(state_feat, dtype=np.float64)
                    next_state_feat_array = np.array(next_state_feat, dtype=np.float64)
                    
                    # Recalculate using current knowledge - we pass feature vector directly
                    next_q_value = self.get_q_value({'features': next_state_feat_array})
                    target_q = reward + self.gamma * next_q_value
                    
                    # Update Q-value with learning rate
                    new_q = old_q + self.alpha * (target_q - old_q)
                    
                    # Find index of this state in our features list
                    try:
                        # We need to compare lists with lists
                        for idx, feat in enumerate(self.state_features):
                            if feat == state_feat:  # Compare the lists directly
                                self.q_values[idx] = new_q
                                break
                    except (ValueError, IndexError) as e:
                        print(f"State index error: {e}")
            except Exception as e:
                print(f"Experience replay error: {e}")
                    
        # Decay exploration rate
        self.decay_epsilon()

    def decay_epsilon(self):
        """Reduce exploration over time"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_model(self, filename):
        """Save model state to file"""
        try:
            # Convert all numpy arrays to lists for better serialization
            np.savez(filename,
                     memory=np.array(list(self.memory), dtype=object),
                     state_features=np.array(self.state_features, dtype=object),
                     q_values=np.array(self.q_values, dtype=np.float64),
                     epsilon=self.epsilon,
                     feature_stats=self.feature_stats if hasattr(self, 'feature_stats') else None,
                     total_reward=self.total_reward,
                     episode_rewards=self.episode_rewards)
        except Exception as e:
            print(f"Model save error: {e}")

    def load_model(self, filename):
        """Load model state from file"""
        try:
            data = np.load(filename, allow_pickle=True)
            
            self.memory = deque(data['memory'].tolist(), maxlen=self.max_memory)
            self.state_features = data['state_features'].tolist()
            self.q_values = data['q_values'].tolist()
            self.epsilon = float(data['epsilon'])
            
            if 'feature_stats' in data and data['feature_stats'].item() is not None:
                self.feature_stats = data['feature_stats'].item()
                
            if 'total_reward' in data:
                self.total_reward = float(data['total_reward'])
                
            if 'episode_rewards' in data:
                self.episode_rewards = data['episode_rewards'].tolist()
                
            self._update_kdtree()
        except Exception as e:
            print(f"Model load error: {e}")
        
    def reset_episode(self):
        """Track episode completion and reset relevant stats"""
        self.episode_rewards.append(self.total_reward)
        self.total_reward = 0