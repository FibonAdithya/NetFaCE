import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import data_generation

def generate_graph_data(V, samples = 1000):
    probability = np.linspace(0,1,samples)

    features_list = []
    chordal = []
        
    for p in probability:
        G = data_generation.generate_graph(V, p)
            
        features = data_generation.get_features(G)
        is_chordal_graph = data_generation.is_chordal(G)
            
        features_list.append(features)
        chordal.append(is_chordal_graph)
    
    return (features_list, chordal)

def visualise(dataset, selected_features = None,
              figsize = None, bins = 30, V = None):
    features_list, labels = dataset

    df = pd.DataFrame(features_list)
    df['is_chordal'] = labels
    
    if figsize is None:
        figsize = (15, 4*len(selected_features))

    fig, axes = plt.subplots(len(selected_features), 1, figsize=figsize)
    
    if len(selected_features) == 1:
        axes = [axes]
    
    for ax, feature in zip(axes, selected_features):
        # Create overlapping histograms
        sns.histplot(data=df, x=feature, hue='is_chordal', 
                    multiple="layer", bins=bins, alpha=0.5, ax=ax)
        
        # Customize the plot
        if V is None:
            ax.set_title(f'Distribution of {feature}')
        else:
            ax.set_title(f'Distribution of {feature} with {V} nodes')
        ax.set_xlabel(feature)
        ax.set_ylabel('Count')
        # Update legend labels
        ax.legend(title='Is Chordal', labels=['Non-Chordal', 'Chordal'])
        
        # Add summary statistics as text
        chordal_stats = df[df['is_chordal']][feature].describe()
        non_chordal_stats = df[~df['is_chordal']][feature].describe()
        
        stats_text = (f'Chordal - Mean: {chordal_stats["mean"]:.2f}, '
                     f'Std: {chordal_stats["std"]:.2f}\n'
                     f'Non-Chordal - Mean: {non_chordal_stats["mean"]:.2f}, '
                     f'Std: {non_chordal_stats["std"]:.2f}')
        
        ax.text(0.08, 0.95, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig




