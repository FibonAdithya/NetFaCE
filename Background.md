# NetFaCE
Investigating what network-based features correspond to chordal extensions

## Background
In many areas of semidefinite optimisation, the aim is to quickly solve a large system of equations. We can represented the matrix as a graph, making the graph chordal means that we can quickly then solve this system by picking a good decomposition. This project aims to invesigate the former property of chordality and what features may correspond to chordality of the graph. The network based features of a graph are as follows:
- Number of verticies
- Number of Edges
- The maximum degree
- The minimum degree
- The mean degree
- The global clustering coefficient
- The density
- The diameter
- The radius
Later in the project we may also use Clique Features:
- Number of cliques
- Maximal size of cliques
- Minimum size of cliques
- Mean size of cliques
- Varience of the size of cliques

## Solution Overview
We seek to implement a series of ML methods and models to gain insight into possible relations with a series of different variables such as:
- Minimum Chordal Extension
- Change in features
- Decomposition time of Cholesky Factorisation

## Classification
Our first experiment is to test if a model can classify whether a graph is chordal or not based on it's features.