# Node2Vec

Python3 implementation of the node2vec algorithm Aditya Grover, Jure Leskovec and Vid Kocijan.
[node2vec: Scalable Feature Learning for Networks. A. Grover, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2016.](https://snap.stanford.edu/node2vec/)

## Installation

`pip install node2vec`

## Usage
```python
import networkx as nx
from node2vec import Node2Vec

# Create a graph
graph = nx.fast_gnp_random_graph(n=100, p=0.5)

# Precompute probabilities and generate walks
node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4) 

# Embed
model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)

# Look for most similar nodes
model.wv.most_similar('2')  # Output node names are always strings

# Save embeddings for later use
model.wv.save_word2vec_format(EMBEDDING_FILENAME)

# Save model for later use
model.save(EMBEDDING_MODEL_FILENAME)

```

### Parameters
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
    
- `Node2Vec.fit` method:
    Accepts any key word argument acceptable by gensim.Word2Vec
    
## Caveats
- Node names in the input graph must be all strings, or all ints
- Does not work on Anaconda + Windows

## TODO
- [x] Parallel implementation for walk generation
- [ ] Parallel implementation for probability precomputation

## Contributing
I will probably not be maintaining this package actively, if someone wants to contribute and maintain, please contact me.
