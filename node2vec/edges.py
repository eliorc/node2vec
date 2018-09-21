import numpy as np
from abc import ABC, abstractmethod
from functools import reduce
from itertools import combinations_with_replacement
from gensim.models import KeyedVectors
from tqdm import tqdm


class EdgeEmbedder(ABC):

    def __init__(self, keyed_vectors, quiet=False):
        """
        :param keyed_vectors: KeyedVectors containing nodes and embeddings to calculate edges for
        :type keyed_vectors: gensim.models.KeyedVectors
        """

        self.kv = keyed_vectors
        self.quiet = quiet

    @abstractmethod
    def _embed(self, edge):
        """
        Abstract method for implementing the embedding method
        :param edge: tuple of two nodes
        :type edge: tuple
        :return: Edge embedding
        :rtype: numpy.array
        """
        pass

    def __getitem__(self, edge):
        if not isinstance(edge, tuple) or not len(edge) == 2:
            raise ValueError('edge must be a tuple of two nodes')

        if edge[0] not in self.kv.index2word:
            raise KeyError('node {} does not exist in given KeyedVectors'.format(edge[0]))

        if edge[1] not in self.kv.index2word:
            raise KeyError('node {} does not exist in given KeyedVectors'.format(edge[1]))

        return self._embed(edge)

    def as_keyed_vectors(self):
        """
        Generated a KeyedVectors instance with all
        :return:
        """

        edge_generator = combinations_with_replacement(self.kv.index2word, r=2)

        if not self.quiet:
            vocab_size = len(self.kv.vocab)
            total_size = reduce(lambda x, y: x * y, range(1, vocab_size + 2)) / \
                         (2 * reduce(lambda x, y: x * y, range(1, vocab_size)))

            edge_generator = tqdm(edge_generator, desc='Generating edge features', total=total_size)

        # Generate features
        tokens = []
        features = []
        for edge in edge_generator:
            token = str(tuple(sorted(edge)))
            embedding = self._embed(edge)

            tokens.append(token)
            features.append(embedding)

        # Build KV instance
        edge_kv = KeyedVectors(vector_size=self.kv.vector_size)
        edge_kv.add(
            entities=tokens,
            weights=features)

        return edge_kv


class AverageEmbedder(EdgeEmbedder):
    """
    Average node features
    """

    def _embed(self, edge):
        return (self.kv[edge[0]] + self.kv[edge[1]]) / 2


class HadamardEmbedder(EdgeEmbedder):
    """
    Hadamard product node features
    """

    def _embed(self, edge):
        return self.kv[edge[0]] * self.kv[edge[1]]


class WeightedL1Embedder(EdgeEmbedder):
    """
    Weighted L1 node features
    """

    def _embed(self, edge):
        return np.abs(self.kv[edge[0]] - self.kv[edge[1]])


class WeightedL2Embedder(EdgeEmbedder):
    """
    Weighted L2 node features
    """

    def _embed(self, edge):
        return (self.kv[edge[0]] - self.kv[edge[1]]) ** 2
