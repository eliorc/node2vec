from abc import ABC, abstractmethod
from functools import reduce
from itertools import combinations_with_replacement

import numpy as np
import pkg_resources
from gensim.models import KeyedVectors
from tqdm import tqdm


class EdgeEmbedder(ABC):
    INDEX_MAPPING_KEY = 'index2word' if pkg_resources.get_distribution("gensim").version < '4.0.0' else 'index_to_key'

    def __init__(self, keyed_vectors: KeyedVectors, quiet: bool = False):
        """
        :param keyed_vectors: KeyedVectors containing nodes and embeddings to calculate edges for
        """

        self.kv = keyed_vectors
        self.quiet = quiet

    @abstractmethod
    def _embed(self, edge: tuple) -> np.ndarray:
        """
        Abstract method for implementing the embedding method
        :param edge: tuple of two nodes
        :return: Edge embedding
        """
        pass

    def __getitem__(self, edge) -> np.ndarray:
        if not isinstance(edge, tuple) or not len(edge) == 2:
            raise ValueError('edge must be a tuple of two nodes')

        if edge[0] not in getattr(self.kv, self.INDEX_MAPPING_KEY):
            raise KeyError('node {} does not exist in given KeyedVectors'.format(edge[0]))

        if edge[1] not in getattr(self.kv, self.INDEX_MAPPING_KEY):
            raise KeyError('node {} does not exist in given KeyedVectors'.format(edge[1]))

        return self._embed(edge)

    def as_keyed_vectors(self) -> KeyedVectors:
        """
        Generated a KeyedVectors instance with all the possible edge embeddings
        :return: Edge embeddings
        """

        edge_generator = combinations_with_replacement(getattr(self.kv, self.INDEX_MAPPING_KEY), r=2)

        if not self.quiet:
            vocab_size = len(getattr(self.kv, self.INDEX_MAPPING_KEY))
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
        if pkg_resources.get_distribution("gensim").version < '4.0.0':
            edge_kv.add(
                entities=tokens,
                weights=features)
        else:
            edge_kv.add_vectors(
                keys=tokens,
                weights=features)

        return edge_kv


class AverageEmbedder(EdgeEmbedder):
    """
    Average node features
    """

    def _embed(self, edge: tuple):
        return (self.kv[edge[0]] + self.kv[edge[1]]) / 2


class HadamardEmbedder(EdgeEmbedder):
    """
    Hadamard product node features
    """

    def _embed(self, edge: tuple):
        return self.kv[edge[0]] * self.kv[edge[1]]


class WeightedL1Embedder(EdgeEmbedder):
    """
    Weighted L1 node features
    """

    def _embed(self, edge: tuple):
        return np.abs(self.kv[edge[0]] - self.kv[edge[1]])


class WeightedL2Embedder(EdgeEmbedder):
    """
    Weighted L2 node features
    """

    def _embed(self, edge: tuple):
        return (self.kv[edge[0]] - self.kv[edge[1]]) ** 2
