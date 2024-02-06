import random
from tqdm import tqdm

import numpy as np


def parallel_generate_walks(d_graph: dict, global_walk_length: int, num_walks: int, cpu_num: int, column_key: str,
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
                
            if column_key not in d_graph[source]:
                continue
                
            # Start walk
            walk = [source]
            walk_columns = [d_graph[source][column_key]]

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
                
                del_idx = []
                for i in range(len(walk_options)):
                    if d_graph[walk_options[i]][column_key] in walk_columns:
                        del_idx.append(i)
                walk_options = [x for x in walk_options if d_graph[x][column_key] not in walk_columns]
        
                if len(walk) == 1:  # For the first step
                    if len(walk_options) > 0:
                        probabilities = d_graph[walk[-1]][first_travel_key]
                        probabilities = np.delete(probabilities, del_idx)
                        walk_to = random.choices(walk_options, weights=probabilities)[0]
                else:
                    probabilities = d_graph[walk[-1]][probabilities_key][walk[-2]]
                    walk_to = random.choices(walk_options, weights=probabilities)[0]

                walk.append(walk_to)
                walk.append(d_graph[walk_to][column_key])

            walk = list(map(str, walk))  # Convert all to strings

            walks.append(walk)

    if not quiet:
        pbar.close()

    return walks
