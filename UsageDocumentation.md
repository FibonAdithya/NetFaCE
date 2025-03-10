# Data Generation

This script provides utilities to generate random graphs, compute their features, and determine if they are chordal. It also includes functionality to create datasets containing graph features and their chordality status.

---

## Overview

This script uses the `NetworkX` library to:
- Generate random graphs using the Erdős-Rényi model.
- Compute key features of a graph, such as the number of nodes, edges, clustering coefficients, and density.
- Check whether a graph is **chordal** (i.e., every cycle of four or more vertices has a chord).
- Create datasets of graph features paired with their chordality status.

---

## Usage Examples

### 1. Generate a Random Graph and Compute Features

```python
import networkx as nx
from data_generation import generate_graph, get_features

G = generate_graph(10, 0.5)
features = get_features(G)
print(features)
```

### 2. Check if a Graph is chordal
```python
from data_generation import is_chordal

is_chordal_graph = is_chordal(G)
print(f"Is the graph chordal? {is_chordal_graph}")
```

### 3. Generate a Dataset
```python
from data_generation import generate_dataset

features_list, chordal_status = generate_dataset(100)
print(features_list[0], chordal_status[0])
```

# Feature Distribution

This script provides utilities to generate graph data using specified parameters and visualize the relationships between features and their chordality status.

---

## Overview

This script includes two main functionalities:
1. **Data Generation:** Generate features and chordality labels for graphs based on the Erdős-Rényi model.
2. **Visualization:** Create histograms to compare the distribution of selected graph features for chordal and non-chordal graphs.
3. **Correlation Analysis** Analyse the relationships between features and chordality status using a correlation matrix.

---

## Usage Examples

### 1. Generate and visualise the data

```python
import feature_distribution
import matplotlib.pyplot as plt

verticies = 10
dataset = feature_distribition.generate_graph_data(V = verticies, samples=1000)
features_list, chordal_labels = dataset
print(features_list[0], chordal_labels[0])

# Selected features to visualize
selected_features = ['num_vertices', 'density', 'max_degree']

# Visualize the dataset
fig = feature_distribution.visualise(dataset, selected_features=selected_features)
plt.show()
```

### 2. See Correlation Matrix
```python
from feature_distribution import correlation
from data_generation import generate_dataset
dataset = generate_dataset(100)

corr_matrix = correlation(dataset)
plt.show()
```
