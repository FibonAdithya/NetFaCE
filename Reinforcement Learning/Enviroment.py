import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
class ChordalGraphEnv():
    def _is_chordal(self):
        return nx.is_chordal(self.G)
    
    def _reset(self):
        """Generate a random graph and reset the enviroment
        """
        nodes = random.randint(20, 40)
        #nodes = 20 # Keep it constant to make choosing cliques easier.
        self.G = self._generate_graph(nodes)
        # Keep regenerating while the graph is chordal
        while self._is_chordal():  # Fixed method call with ()
            print("Generating graph...")
            self.G = self._generate_graph(nodes)

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
        
        clique_sizes = [len(clq) for clq in self.clique_graph]
        features = {
            'num_vertices': len(self.G.nodes),
            'num_edges': len(self.G.edges),
            'max_degree': max(degrees),
            'min_degree': min(degrees),
            'mean_degree': np.mean(degrees),
            #clique features
            'max_clique_size': max(clique_sizes),
            'avg_clique_size': np.mean(clique_sizes),
            'clique_size_std': np.std(clique_sizes),
            'average_clustering': nx.average_clustering(self.G),
            'global_clustering': global_clustering,
            'density': nx.density(self.G)
        }

        try:
            features['diameter'] = nx.diameter(self.G)
            features['radius'] = nx.radius(self.G)
        except nx.NetworkXError:
            features['diameter'] = np.inf
            features['radius'] = np.inf
        return features

    def _generate_graph(self, nodes):
        # Generate random matrix with smaller values
        A = np.random.normal(scale=0.3, size=(nodes, nodes))  # Reduced scale
        matrix = A @ A.T
        
        # Create adjacency matrix with threshold
        adj_matrix = np.abs(matrix - np.diag(np.diag(matrix)))
        threshold = 0.5  # Increased threshold for sparsity
        adj_matrix[adj_matrix < threshold] = 0
        
        # Create graph with edge filtering
        G = nx.from_numpy_array(adj_matrix)
        # Remove isolated nodes for better connectivity
        G.remove_nodes_from(list(nx.isolates(G)))
        return G
    
    def _get_state(self):
        self.clique_graph = self.find_clique_graph()
        features = self._get_features()
        return features

    def generate_reward(self):
        return self.heuristic_1()

    def step(self, merge):
        self.merge_cliques(merge)
        if self._is_chordal():
            return self._get_state(), True
        return self._get_state(), False

    def heuristic_1(self, m= 0):
        clique_sum = sum(len(clique) * (len(clique) + 1) / 2 for clique in self.clique_graph)
        l = len(self.clique_graph) - 1
        return m + l + clique_sum

    def find_clique_graph(self):
        cliques = [tuple(sorted(clique)) for clique in nx.find_cliques(self.G)]
        clique_graph = nx.Graph()
        
        # Add cliques as nodes
        clique_graph.add_nodes_from(cliques)
        
        # Add edges between cliques with shared nodes
        for i in range(len(cliques)):
            for j in range(i + 1, len(cliques)):
                if set(cliques[i]).intersection(set(cliques[j])):
                    clique_graph.add_edge(cliques[i], cliques[j])
        return clique_graph

    def valid_clique_merges(self):
        return list(self.clique_graph.edges())
    
    def merge_cliques(self, merge):
        clique_a, clique_b = merge
        # Ensure cliques are stored as sorted tuples for consistency
        set_a = set(clique_a)
        set_b = set(clique_b)
        
        for a in set_a:
            for b in set_b:
                if a != b:  # Prevent self-loops
                    self.G.add_edge(a, b)
        
    
    def is_clique(self, nodes):
        return all(self.G.has_edge(u, v) for u in nodes for v in nodes if u != v)
    
    def visualize(self):
        if self._is_chordal():
            colour = "green"
        else:
            colour = "skyblue"
        nx.draw(self.G, with_labels=True, node_color=colour)
        plt.show()