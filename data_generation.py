import networkx as nx
import numpy as np
import random

def generate_graph(n, p):
    """
    Generate a random graph using Erdős-Rényi model
    
    Args:
        n: Number of nodes
        p: Probability of edge creation
    Returns:
        NetworkX graph object
    """
    return nx.erdos_renyi_graph(n, p)

def get_features(G):
    """
    Calculate all network-based features for a given graph
    
    Args:
        G: NetworkX graph object
    Returns:
        Dictionary of features
    """
    triangles = sum(nx.triangles(G).values())/3 #Divide by 3 to remove triple counting
    sum_deg_C_2 = 0 #Sum of (degree Choose 2)
    degrees = []
    for v,d in G.degree():
        degrees.append(d)
        sum_deg_C_2 += d*(d-1)/2
    
    try:
        global_clustering = triangles/sum_deg_C_2
    except ZeroDivisionError:
        global_clustering = 0
    
    features = {
        'num_vertices': len(G.nodes),
        'num_edges': len(G.edges),
        'max_degree': max(degrees),
        'min_degree': min(degrees),
        'mean_degree': np.mean(degrees),
        'average_clustering': nx.average_clustering(G),
        'global_clustering': global_clustering,
        'density': nx.density(G)
    }

    try:
        features['diameter'] = nx.diameter(G)
        features['radius'] = nx.radius(G)
    except nx.NetworkXError:
        # Handle disconnected graphs
        features['diameter'] = float('inf')
        features['radius'] = float('inf')
    
    return features

def is_chordal(G: nx.Graph) -> bool:
    """
    Check if a graph is chordal
    Args:
        G: NetworkX graph object
    Returns:
        Boolean indicating if graph is chordal
    """
    return nx.is_chordal(G)

def generate_dataset(num_samples, min_nodes = 5, max_nodes = 20):

    features_list = []
    chordal = []
    
    for _ in range(num_samples):
        n = random.randint(min_nodes, max_nodes)
        p = random.uniform(0.1, 0.9)
        G = generate_graph(n, p)
        
        features = get_features(G)
        is_chordal_graph = is_chordal(G)
        
        features_list.append(features)
        chordal.append(is_chordal_graph)
    
    return (features_list, chordal)