This is a read me file to explain the code.

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
from data_generation import generate_dataset

features_list, chordal_status = generate_dataset(100)
print(features_list[0], chordal_status[0])