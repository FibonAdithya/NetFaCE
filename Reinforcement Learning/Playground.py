import numpy as np
from Trainer import ChordalTrainer
from Enviroment import ChordalGraphEnv
from Agents import KNNGraphAgent
import matplotlib.pyplot as plt

def analyze_strategy(agent):
    # 1. Preferred Merge Sizes
    merge_sizes = [len(a[0])+len(a[1]) for a in agent.memory]
    print(f"Average merge size: {np.mean(merge_sizes):.1f} nodes")
    
    # 2. Feature Correlation
    feature_impact = agent.q_net.feature_importances()
    print("Most influential features:")
    for feat, weight in sorted(feature_impact.items(), key=lambda x: -x[1]):
        print(f"- {feat}: {weight:.2f}")

class Playground():
    def __init__(self):
        from Trainer import ChordalTrainer
        from Enviroment import ChordalGraphEnv
        from Agents import KNNGraphAgent
        
        self.env = ChordalGraphEnv()
        self.agent = KNNGraphAgent()
        self.trainer = ChordalTrainer(agent, env)

    def run(self,num_of_episodes, save = False, load = False):
        filename = "Agent"
        if load:
            self.agent.load_model()

        # Train with progress tracking
        rewards = self.trainer.train(num_episodes=num_of_episodes)
        
        # Visualize learning progress
        plt.plot(rewards)
        plt.title("Learning Progress")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.show()
        
        # Extract strategic insights
        analyze_strategy(agent)

        if save:
            self.agent.save_model(filename)

if __name__ == "__main__":
    from Trainer import ChordalTrainer
    from Enviroment import ChordalGraphEnv
    from Agents import KNNGraphAgent

    env = ChordalGraphEnv()
    agent = KNNGraphAgent()
    trainer = ChordalTrainer(agent, env)
    
    # Train with progress tracking
    rewards = trainer.train(num_episodes=100)
    
    # Visualize learning progress
    plt.plot(rewards)
    plt.title("Learning Progress")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()
    
    # Extract strategic insights
    analyze_strategy(agent)