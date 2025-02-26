import networkx as nx
import random

class ChordalTrainer:
    def __init__(self, agent, env, gamma=0.95):
        self.agent = agent          # Your RL agent
        self.env = env              # Graph environment
        self.gamma = gamma          # Discount factor
        self.episode_buffer = []    # Stores (state, action, reward) sequences
    
    def run_episode(self):
        self.env._reset()
        state = self.env.get_state()
        episode_log = []
        done = False
        
        while not done:
            # Get merge candidates
            candidates = self.env.valid_clique_merges()
            
            # Agent selects action
            action = self.agent.choose_action(state, candidates)
            
            # Store pre-merge state
            episode_log.append({
                'state': state,
                'action': action,
                'reward': 0  # Placeholder until terminal
            })
            
            # Perform merge
            next_state, done = self.env.step(action)
            state = next_state
        
        # Calculate final reward using your equation
        final_reward = self.get_reward(self.env)
        
        # Apply reward to all steps with discounting
        self.process_episode(episode_log, final_reward)
        
        return final_reward
    
    def process_episode(self, episode_log, final_reward):
        """Backpropagate discounted rewards through the episode"""
        discounted_reward = 0
        
        # Reverse through the episode steps
        for step in reversed(episode_log):
            discounted_reward = final_reward + self.gamma * discounted_reward
            step['reward'] = discounted_reward  # Update step reward
            
            # Store experience in replay buffer
            self.agent.memory.append(
                (step['state'], step['action'], step['reward'])
            )
            
            # Apply discount decay for previous steps
            discounted_reward *= self.gamma
    
    def calculate_reward(self, env):
        """Your custom reward equation"""
        base_reward = 1000  # Reward for success
        
        # Penalty components (customize weights)
        edge_penalty = -5 * (len(env.G.edges) - env.initial_edges)
        merge_penalty = -2 * env.steps
        cycle_penalty = -10 * len(nx.chordless_cycles(env.G))
        
        # Time penalty if applicable
        time_penalty = -0.1 * env.time_elapsed  
        
        return base_reward + edge_penalty + merge_penalty + cycle_penalty + time_penalty
    
    def train(self, num_episodes=1000, batch_size=32):
        rewards_history = []
        
        for episode in range(num_episodes):
            # Run episode and get final reward
            episode_reward = self.run_episode()
            rewards_history.append(episode_reward)
            
            # Experience replay
            if len(self.agent.memory) >= batch_size:
                batch = random.sample(self.agent.memory, batch_size)
                self.agent.learn(batch)
            
            # Exploration decay
            self.agent.decay_epsilon()
        
        return rewards_history