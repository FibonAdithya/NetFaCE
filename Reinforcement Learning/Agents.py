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
                 max_memory=1000):
        """
        k: Number of nearest neighbors to consider
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration rate
        max_memory: Maximum stored experiences
        """
        self.k = k
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_memory = max_memory
        
        # Experience memory: (state, action, reward, next_state)
        self.memory = deque(maxlen=max_memory)
        
        # KDTree for efficient neighbor search
        self.kdtree = None
        self.state_features = []
        self.q_values = []

    def _features_to_state(self, features):
        """Convert raw state to normalized feature vector"""
        # Normalize feature
        return np.array([
            features['num_vertices'],
            features['num_edges'],
            features['max_degree'],
            features['min_degree'],
            features['mean_degree'],
            #clique features
            features['max_clique_size'],
            features['avg_clique_size'],
            features['clique_size_std'],
            features['average_clustering'],
            features['global_clustering'],
            features['density'],
            features.get('diameter', 0),  # Handle missing via .get()
            features.get('radius', 0)
        ])

    def _update_kdtree(self):
        """Rebuild KDTree index after memory updates"""
        if len(self.state_features) > 0:
            self.kdtree = KDTree(self.state_features)

    def get_q_value(self, state, action=None):
        """Estimate Q-value using k-NN weighted average"""
        state_vec = self._features_to_state(state)
        
        if len(self.memory) == 0:
            return np.random.uniform(0, 1)  # Initial random estimate

        # Find k nearest neighbors
        dists, indices = self.kdtree.query([state_vec], k=min(self.k, len(self.memory)))
        
        # Get weighted average of Q-values (inverse distance weighting)
        weights = 1 / (dists[0] + 1e-6)  # Avoid division by zero
        weights /= weights.sum()
        
        neighbor_q_values = [self.q_values[i] for i in indices[0]]
        return np.dot(weights, neighbor_q_values)

    def choose_action(self, state, merge_candidates):
        """ε-greedy action selection"""
        self.state_features = self._features_to_state(state)
        
        if np.random.rand() < self.epsilon:
            return random.choice(merge_candidates)
            
        # Find best action based on current estimates
        q_values = []
        for candidate in merge_candidates:
            # Create candidate state representation
            candidate_state = {
                **state,
                'clique1_size': len(candidate[0]),
                'clique2_size': len(candidate[1])
            }
            q_values.append(self.get_q_value(candidate_state))
            
        return merge_candidates[np.argmax(q_values)]

    def store_experience(self, state, action, reward, next_state):
        """Store experience in memory"""
        state_features = self._state_to_features(state)
        next_q = self.get_q_value(next_state)
        
        # Update Q-value using Bellman equation
        updated_q = reward + self.gamma * next_q
        old_q = self.get_q_value(state)
        new_q = old_q + self.alpha * (updated_q - old_q)
        
        # Add to memory
        self.memory.append((state, action, reward, next_state))
        self.state_features.append(state_features)
        self.q_values.append(new_q)
        
        # Maintain memory size
        if len(self.memory) > self.max_memory:
            self.memory.popleft()
            self.state_features.pop(0)
            self.q_values.pop(0)
            
        self._update_kdtree()

    def decay_epsilon(self, decay_rate=0.995, min_epsilon=0.01):
        """Reduce exploration over time"""
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)

    def save_model(self, filename):
        """Save model state to file"""
        np.savez(filename,
                 memory=self.memory,
                 state_features=self.state_features,
                 q_values=self.q_values,
                 epsilon=self.epsilon)

    def load_model(self, filename):
        """Load model state from file"""
        data = np.load(filename, allow_pickle=True)
        self.memory = data['memory'].tolist()
        self.state_features = data['state_features'].tolist()
        self.q_values = data['q_values'].tolist()
        self.epsilon = data['epsilon'].item()
        self._update_kdtree()
