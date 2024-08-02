import random
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import logging
from tqdm.auto import tqdm

def parallel_generate_walks(d_graph: dict, global_walk_length: int, num_walks: int, cpu_num: int,
                            sampling_strategy: dict = None, num_walks_key: str = None, walk_length_key: str = None,
                            neighbors_key: str = None, probabilities_key: str = None, first_travel_key: str = None,
                            quiet: bool = False) -> list:
    """
    Generates the random walks which will be used as the skip-gram input.

    :return: List of walks. Each walk is a list of nodes.
    """

    walks = list()

    if not quiet:
        pbar = tqdm(total=num_walks, desc='Generating walks (CPU: {})'.format(cpu_num))

    for n_walk in range(num_walks):

        # Update progress bar
        if not quiet:
            pbar.update(1)

        # Shuffle the nodes
        shuffled_nodes = list(d_graph.keys())
        random.shuffle(shuffled_nodes)

        # Start a random walk from every node
        for source in shuffled_nodes:

            # Skip nodes with specific num_walks
            if source in sampling_strategy and \
                    num_walks_key in sampling_strategy[source] and \
                    sampling_strategy[source][num_walks_key] <= n_walk:
                continue

            # Start walk
            walk = [source]

            # Calculate walk length
            if source in sampling_strategy:
                walk_length = sampling_strategy[source].get(walk_length_key, global_walk_length)
            else:
                walk_length = global_walk_length

            # Perform walk
            while len(walk) < walk_length:

                walk_options = d_graph[walk[-1]].get(neighbors_key, None)

                # Skip dead end nodes
                if not walk_options:
                    break

                if len(walk) == 1:  # For the first step
                    probabilities = d_graph[walk[-1]][first_travel_key]
                    walk_to = random.choices(walk_options, weights=probabilities)[0]
                else:
                    probabilities = d_graph[walk[-1]][probabilities_key][walk[-2]]
                    walk_to = random.choices(walk_options, weights=probabilities)[0]

                walk.append(walk_to)

            walk = list(map(str, walk))  # Convert all to strings

            walks.append(walk)

    if not quiet:
        pbar.close()

    return walks



class NodeTransitionProbabilities:
    def __init__(self, graph, sampling_strategy, p, q, weight_key,probabilities_key,first_travel_key,neighbor_key):
        self.graph = graph
        self.sampling_strategy = sampling_strategy
        self.p = p
        self.q = q
        self.weight_key = weight_key
        self.PROBABILITIES_KEY = probabilities_key
        self.FIRST_TRAVEL_KEY = first_travel_key
        self.NEIGHBORS_KEY = neighbor_key
        self.d_graph = {node: {} for node in graph.nodes}
        self.locks = {node: Lock() for node in graph.nodes}

    def _compute_node_probabilities(self, source):
        '''
        Helper function for parallel computation of transition probabilities of nodes.
        '''
        d_graph = self.d_graph

        # Acquire lock for the source node to safely initialize its data
        with self.locks[source]:
            if self.PROBABILITIES_KEY not in d_graph[source]:
                d_graph[source][self.PROBABILITIES_KEY] = dict()

        for current_node in self.graph.neighbors(source):
            # Acquire lock for the current node to safely initialize its data
            with self.locks[current_node]:
                if self.PROBABILITIES_KEY not in d_graph[current_node]:
                    d_graph[current_node][self.PROBABILITIES_KEY] = dict()

            unnormalized_weights = list()
            d_neighbors = list()

            for destination in self.graph.neighbors(current_node):
                p = self.sampling_strategy[current_node].get(self.P_KEY, self.p) if current_node in self.sampling_strategy else self.p
                q = self.sampling_strategy[current_node].get(self.Q_KEY, self.q) if current_node in self.sampling_strategy else self.q

                try:
                    if self.graph[current_node][destination].get(self.weight_key):
                        weight = self.graph[current_node][destination].get(self.weight_key, 1)
                    else:
                        edge = list(self.graph[current_node][destination])[-1]
                        weight = self.graph[current_node][destination][edge].get(self.weight_key, 1)
                except:
                    weight = 1

                if destination == source:
                    ss_weight = weight * 1 / p
                elif destination in self.graph[source]:
                    ss_weight = weight
                else:
                    ss_weight = weight * 1 / q

                unnormalized_weights.append(ss_weight)
                d_neighbors.append(destination)

            unnormalized_weights = np.array(unnormalized_weights)
            with self.locks[current_node]:
                if source in d_graph[current_node][self.PROBABILITIES_KEY]:
                    raise Exception(f'Overwriting probabilities for node {current_node} from source {source}')
                d_graph[current_node][self.PROBABILITIES_KEY][source] = unnormalized_weights / unnormalized_weights.sum()

        first_travel_weights = []

        for destination in self.graph.neighbors(source):
            first_travel_weights.append(self.graph[source][destination].get(self.weight_key, 1))

        first_travel_weights = np.array(first_travel_weights)
        with self.locks[source]:
            d_graph[source][self.FIRST_TRAVEL_KEY] = first_travel_weights / first_travel_weights.sum()
            d_graph[source][self.NEIGHBORS_KEY] = list(self.graph.neighbors(source))

    def compute_probabilities_parallel(self,quiet,n_workers):
        nodes_generator = self.graph.nodes() if quiet \
            else tqdm(self.graph.nodes(), desc='Computing transition probabilities')
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(self._compute_node_probabilities, node): node for node in nodes_generator}
            for future in as_completed(futures):
                node = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f'Node {node} generated an exception: {exc}')
