# Node2Vec 
[![Downloads](http://pepy.tech/badge/node2vec)](http://pepy.tech/project/node2vec)

Python3 implementation of the node2vec algorithm Aditya Grover, Jure Leskovec and Vid Kocijan.
[node2vec: Scalable Feature Learning for Networks. A. Grover, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2016.](https://snap.stanford.edu/node2vec/)

## Changes:

New in `0.2.2`:

Added edge embedding functionality. Module `node2vec.edges`.
(Fixed error upon installation)

## Installation

`pip install node2vec`

## Usage
```python
import networkx as nx
from node2vec import Node2Vec

# Create a graph
graph = nx.fast_gnp_random_graph(n=100, p=0.5)

# Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4) 

# Embed nodes
model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)

# Look for most similar nodes
model.wv.most_similar('2')  # Output node names are always strings

# Save embeddings for later use
model.wv.save_word2vec_format(EMBEDDING_FILENAME)

# Save model for later use
model.save(EMBEDDING_MODEL_FILENAME)

# Embed edges using Hadamard method
from node2vec.edges import HadamardEmbedder

edges_embs = HadamardEmbedder(keyed_vectors=model.wv)

# Look for embeddings on the fly - here we pass normal tuples
edges_embs[('1', '2')]
''' OUTPUT
array([ 5.75068220e-03, -1.10937878e-02,  3.76693785e-01,  2.69105062e-02,
       ... ... ....
       ..................................................................],
      dtype=float32)
'''

# Get all edges in a separate KeyedVectors instance - use with caution could be huge for big networks
edges_kv = edges_embs.as_keyed_vectors()

# Look for most similar edges - this time tuples must be sorted and as str
edges_kv.most_similar(str(('1', '2')))

# Save embeddings for later use
edges_kv.save_word2vec_format(EDGES_EMBEDDING_FILENAME)

```

### Parameters

#### `node2vec.Node2vec`

- `Node2Vec` constructor:
    1. `graph`: The first positional argument has to be a networkx graph. Node names must be all integers or all strings. On the output model they will always be strings.
    2. `dimensions`: Embedding dimensions (default: 128)
    3. `walk_length`: Number of nodes in each walk (default: 80)
    4. `num_walks`: Number of walks per node (default: 10)
    5. `p`: Return hyper parameter (default: 1)
    6. `q`: Inout parameter (default: 1)
    7. `weight_key`: On weighted graphs, this is the key for the weight attribute (default: 'weight')
    8. `workers`: Number of workers for parallel execution (default: 1)
    9. `sampling_strategy`: Node specific sampling strategies, supports setting node specific 'q', 'p', 'num_walks' and 'walk_length'.
        Use these keys exactly. If not set, will use the global ones which were passed on the object initialization`
    10. `quiet`: Boolean controlling the verbosity. (default: False)
    
- `Node2Vec.fit` method:
    Accepts any key word argument acceptable by gensim.Word2Vec

#### `node2vec.EdgeEmbedder`

`EdgeEmbedder` is an abstract class which all the concrete edge embeddings class inherit from.
The classes are `AverageEmbedder`, `HadamardEmbedder`, `WeightedL1Embedder` and `WeightedL2Embedder` which their practical definition could be found in the [paper](https://arxiv.org/pdf/1607.00653.pdf) on table 1
Notice that edge embeddings are defined for any pair of nodes, connected or not and even node with itself.

- Constructor:
    1. `keyed_vectors`: A gensim.models.KeyedVectors instance containing the node embeddings
    2. `quiet`: Boolean controlling the verbosity. (default: False)

- `EdgeEmbedder.__getitem__(item)` method, better known as `EdgeEmbedder[item]`:
    1. `item` - A tuple consisting of 2 nodes from the `keyed_vectors` passed in the constructor. Will return the embedding of the edge.

- `EdgeEmbedder.as_keyed_vectors` method: Returns a `gensim.models.KeyedVectors` instance with all possible node pairs in a *sorted* manner as string.
  For example, for nodes ['1', '2', '3'] we will have as keys "('1', '1')", "('1', '2')", "('1', '3')", "('2', '2')", "('2', '3')" and "('3', '3')".

## Caveats
- Node names in the input graph must be all strings, or all ints
- Parallel execution not working on Windows (`joblib` known issue). To run non-parallel on Windows pass `workers=1` on the `Node2Vec`'s constructor

## TODO
- [x] Parallel implementation for walk generation
- [ ] Parallel implementation for probability precomputation

## Contributing
I will probably not be maintaining this package actively, if someone wants to contribute and maintain, please contact me.
