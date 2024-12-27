This is a read me file to explain the changes that we've made to code and how to use it.

# Feature Distribution

This script provides utilities to generate graph data using specified parameters and visualize the relationships between features and their chordality status.

---

## Overview

This script includes two main functionalities:
1. **Data Generation:** Generate features and chordality labels for graphs based on the Erdős-Rényi model.
2. **Visualization:** Create histograms to compare the distribution of selected graph features for chordal and non-chordal graphs.
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

### 2. LocalAnjos script
The local Anjos script that you wrote now becomes

```python
import feature_distribution
import matplotlib.pyplot as plt

selected_features = ["global_clustering"]
for i in range(5,11):
    data = feature_distribution.generate_graph_data(V = i, samples = 1000)
    fig = feature_distribution.visualise(data, selected_features, V = i)
    plt.show()
```