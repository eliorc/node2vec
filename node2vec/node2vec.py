import random
import numpy as np
import gensim
from tqdm import tqdm


class Node2Vec:

    PROBABILITIES_KEY = 'probabilities'
    NUM_WALKS_KEY = 'num_walks'
    WALK_LENGTH_KEY = 'walk_length'
    P_KEY = 'p'
    Q_KEY = 'q'

    def __init__(self, graph, dimensions=128, walk_length=80, num_walks=10, p=1, q=1, weight_key='weight',
                 sampling_strategy={}):
        """
        Initiates the Node2Vec object, precomputes walking probabilities and generates the walks.
        :param graph: Input graph
        :type graph: Networkx Graph
        :param dimensions: Embedding dimensions (default: 128)
        :type dimensions: int
        :param walk_length: Number of nodes in each walk (default: 80)
        :type walk_length: int
        :param num_walks: Number of walks per node (default: 10)
        :type num_walks: int
        :param p: Return hyper parameter (default: 1)
        :type p: float
        :param q: Inout parameter (default: 1)
        :type q: float
        :param weight_key: On weighted graphs, this is the key for the weight attribute (default: 'weight')
        :type weight_key: str
        :param sampling_strategy: Node specific sampling strategies, supports setting node specific 'q', 'p', 'num_walks' and 'walk_length'.
        Use these keys exactly. If not set, will use the global ones which were passed on the object initialization
        """
        self.graph = graph
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.weight_key = weight_key
        self.sampling_strategy = sampling_strategy

        self._precompute_probabilities()
        self.walks = self._generate_walks()

    def _precompute_probabilities(self):
        """
        Precomputes transition probabilities for each node.
        """

        for source in self.graph.nodes():

            for current_node in self.graph.neighbors(source):

                # Init probabilities dict
                if self.PROBABILITIES_KEY not in self.graph.node[current_node]:
                    self.graph.node[current_node][self.PROBABILITIES_KEY] = dict()

                if current_node == 3 and source == 1:
                    asd = 123

                unnormalized_weights = list()

                # Calculate unnormalized weights
                for destination in self.graph.neighbors(current_node):

                    # Retrieve p and q
                    p = self.sampling_strategy[current_node].get(self.P_KEY,
                                                                 self.p) if current_node in self.sampling_strategy else self.p
                    q = self.sampling_strategy[current_node].get(self.Q_KEY,
                                                                 self.q) if current_node in self.sampling_strategy else self.q

                    if destination == source:  # Backwards probability
                        ss_weight = self.graph[current_node][destination].get(self.weight_key, 1) * 1/p
                    elif destination in self.graph[source]:  # If the neighbor is connected to the source
                        ss_weight = self.graph[current_node][destination].get(self.weight_key, 1)
                    else:
                        ss_weight = self.graph[current_node][destination].get(self.weight_key, 1) * 1/q

                    # Assign the unnormalized sampling strategy weight, normalize during random walk
                    unnormalized_weights.append(ss_weight)

                # Normalize
                unnormalized_weights = np.array(unnormalized_weights)
                self.graph.node[current_node][self.PROBABILITIES_KEY][source] = unnormalized_weights / unnormalized_weights.sum()

    def _generate_walks(self):
        """
        Generates the random walks which will be used as the skip-gram input.
        :return: List of walks. Each walk is a list of nodes.
        """

        walks = list()

        with tqdm(total=self.num_walks) as pbar:
            pbar.set_description('Generating walks')

            for n_walk in range(self.num_walks):

                # Update progress bar
                pbar.update(1)

                # Shuffle the nodes
                shuffled_nodes = list(self.graph.nodes())
                random.shuffle(shuffled_nodes)

                # Start a random walk from every node
                for source in shuffled_nodes:

                    # Skip nodes with specific num_walks
                    if source in self.sampling_strategy and \
                            self.NUM_WALKS_KEY in self.sampling_strategy[source] and \
                            self.sampling_strategy[source][self.NUM_WALKS_KEY] <= n_walk:
                        continue

                    # Start walk
                    walk = [source]

                    # Calculate walk length
                    if source in self.sampling_strategy:
                        walk_length = self.sampling_strategy[source].get(self.WALK_LENGTH_KEY, self.walk_length)
                    else:
                        walk_length = self.walk_length

                    while len(walk) < walk_length:
                        walk_options = list(self.graph.neighbors(walk[-1]))

                        if len(walk) == 1:  # For the first step
                            walk_to = np.random.choice(walk_options, size=1)[0]
                        else:
                            probabilities = self.graph.node[walk[-1]][self.PROBABILITIES_KEY][walk[-2]]
                            walk_to = np.random.choice(walk_options, size=1, p=probabilities)[0]

                        walk.append(walk_to)

                    walk = list(map(str, walk))

                    walks.append(walk)

        return walks

    def fit(self, **skip_gram_params):
        """
        Creates the embeddings using gensim's Word2Vec.
        :param skip_gram_params: Parameteres for gensim.models.Word2Vec - do not supply 'size' it is taken from the Node2Vec 'dimensions' parameter
        :type skip_gram_params: dict
        :return: A gensim word2vec model
        """
        return gensim.models.Word2Vec(self.walks, size=self.dimensions, **skip_gram_params)
