import numpy as np
import networkx as nx

def generate_semidefinite_matrix(n, rank=None):
    """
    Generates a positive semi-definite matrix of size n x n.
    """
    if rank is None:
        rank = n
    assert 1 <= rank <= n, "Rank must be between 1 and n"
    A = np.random.normal(size=(n, rank))
    return A @ A.T

def generate_graph(matrix):
    # Remove diagonal (self-edges) and keep absolute values for edge weights
    adj_matrix = np.abs(matrix - np.diag(np.diag(matrix)))

    # Create a graph from the adjacency matrix
    G = nx.from_numpy_array(adj_matrix)
    return G