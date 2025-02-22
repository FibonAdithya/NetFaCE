import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt

class Env():
    def __init__(self):
        self._get_graph()

    def _is_chordal(self):
        return nx.is_chordal(self.G)
    
    def _get_graph(self):
        #nodes = random.randint(10, 20)
        nodes = 20 # Keep it constant to make choosing cliques easier.
        self.G = self._generate_graph(nodes)
        # Keep regenerating while the graph is chordal
        while self._is_chordal():  # Fixed method call with ()
            print("Generated chordal graph, regenerating...")
            self.G = self._generate_graph(nodes)
            print(f"Nodes: {nodes}, Edges: {self.G.number_of_edges()}")

    def _get_features(self):
        """
        Calculate all network-based features for a given graph
        
        Args:
            G: NetworkX graph object
        Returns:
            Dictionary of features
        """
        triangles = sum(nx.triangles(self.G).values())/3 #Divide by 3 to remove triple counting
        sum_deg_C_2 = 0 #Sum of (degree Choose 2)
        degrees = []
        for v,d in self.G.degree():
            degrees.append(d)
            sum_deg_C_2 += d*(d-1)/2
        
        try:
            global_clustering = triangles/sum_deg_C_2
        except ZeroDivisionError:
            global_clustering = 0
        
        clique_gen = nx.find_cliques(self.G)

        clique_sizes = [len(clq) for clq in nx.find_cliques(self.G)]
        features = {
            'num_vertices': len(self.G.nodes),
            'num_edges': len(self.G.edges),
            'max_degree': max(degrees),
            'min_degree': min(degrees),
            'mean_degree': np.mean(degrees),
            'clique_sizes': clique_sizes,
            'average_clustering': nx.average_clustering(self.G),
            'global_clustering': global_clustering,
            'density': nx.density(self.G)
        }

        try:
            features['diameter'] = nx.diameter(self.G)
            features['radius'] = nx.radius(self.G)
        except nx.NetworkXError:
            # Handle disconnected graphs
            features['diameter'] = float('inf')
            features['radius'] = float('inf')
        return features
    
    def _get_non_chordal_cycles(self):
        pass

    def _generate_graph(self, nodes):
        # Generate random matrix with smaller values
        A = np.random.normal(scale=0.3, size=(nodes, nodes))  # Reduced scale
        matrix = A @ A.T
        
        # Create adjacency matrix with threshold
        adj_matrix = np.abs(matrix - np.diag(np.diag(matrix)))
        threshold = 0.2  # Increased threshold for sparsity
        adj_matrix[adj_matrix < threshold] = 0
        
        # Create graph with edge filtering
        G = nx.from_numpy_array(adj_matrix)
        # Remove isolated nodes for better connectivity
        G.remove_nodes_from(list(nx.isolates(G)))
        return G

    def visualize(self):
        if self._is_chordal():
            colour = "green"
        else:
            colour = "skyblue"
        nx.draw(self.G, with_labels=True, node_color=colour)
        plt.show()