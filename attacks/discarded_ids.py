"""
This file contains a custom implementation of Multi Krum and a collect weights decorator such as client's ids are also collected. 
This way we obtain also which clients as been discarded by the aggregation algorithm. 
"""

import functools
from typing import Hashable, List, Union

import tensorly as tl
from flex.model import FlexModel
from flex.pool.aggregators import set_tensorly_backend


# Copied from flex-clash
def generalized_percentile_aggregator_f(
    list_of_weights: list, percentile: Union[slice, int]
):
    agg_weights = []
    number_of_layers = len(list_of_weights[0])
    for layer_index in range(number_of_layers):
        weights_per_layer = [weights[layer_index] for weights in list_of_weights]
        weights_per_layer = tl.stack(weights_per_layer)
        agg_layer = tl.sort(weights_per_layer, axis=0)[percentile]
        agg_weights.append(agg_layer)
    return agg_weights


def median_f(list_of_weights: list):
    num_clients = len(list_of_weights)
    median_pos = num_clients // 2
    return generalized_percentile_aggregator_f(list_of_weights, median_pos)


def compute_distance_matrix(list_of_weights: list):
    num_clients = len(list_of_weights)
    distance_matrix = [list(range(num_clients)) for i in range(num_clients)]
    for i in range(num_clients):
        w_i = list_of_weights[i]
        for j in range(i, num_clients):
            w_j = list_of_weights[j]
            tmp_dist = sum([tl.norm(a - b) ** 2 for a, b in zip(w_i, w_j)])
            distance_matrix[i][j] = tmp_dist
            distance_matrix[j][i] = tmp_dist
    return distance_matrix


# Modified for keeping track of ids
def krum_criteria(distance_matrix, list_of_ids: List[Hashable], f, m):
    num_clients = len(distance_matrix)
    assert num_clients == len(list_of_ids)
    # Compute scores
    scores = []
    num_selected = num_clients - f - 2
    for i in range(num_clients):
        completed_scores = distance_matrix[i]
        completed_scores.sort()
        scores.append(
            sum(completed_scores[1 : num_selected + 1])
        )  # distance to oneself is always first
    # We associate each client with her scores and sort them using her scores
    pairs = [(i, scores[i], list_of_ids[i]) for i in range(num_clients)]
    pairs.sort(key=lambda pair: pair[1])
    return pairs[:m], pairs[m + 1 :]


def _multikrum(list_of_weights: list, list_of_ids: List[Hashable], f=1, m=5):
    set_tensorly_backend(list_of_weights)
    distance_matrix = compute_distance_matrix(list_of_weights)
    pairs, discarded = krum_criteria(distance_matrix, list_of_ids, f, m)
    selected_weights = [list_of_weights[i] for i, _, _ in pairs]
    discarded_ids = [id for _, _, id in discarded]
    # Think twice, maybe I can return the distance to aggregation result of those discarded
    return median_f(selected_weights), discarded_ids


# Copied from flex-clash until here


# Krum aggregator wrapped in order to not to use the decorator for custom control
def multikrum(aggregator_flex_model: FlexModel, _dataset, *args, **kwargs):
    aggregated_weights, discarded_ids = _multikrum(
        aggregator_flex_model["weights"],
        aggregator_flex_model["weights_ids"],
        *args,
        **kwargs,
    )
    aggregator_flex_model["aggregated_weights"] = aggregated_weights
    aggregator_flex_model["discarded_ids"] = discarded_ids
    aggregator_flex_model["weights"] = []
    aggregator_flex_model["weights_ids"] = []
    return aggregator_flex_model


krum = functools.partial(multikrum, m=1)


# Decorator for collecting weights from clients and their ids
def collect_clients_weights_with_ids(func):

    @functools.wraps(func)
    def _collect_weights_(
        aggregator_flex_model: FlexModel,
        clients_flex_models: List[FlexModel],
        *args,
        **kwargs,
    ):
        if "weights" not in aggregator_flex_model:
            aggregator_flex_model["weights"] = []
        if "weights_ids" not in aggregator_flex_model:
            aggregator_flex_model["weights_ids"] = []
        for k in clients_flex_models:
            client_weights = func(clients_flex_models[k], *args, **kwargs)
            aggregator_flex_model["weights"].append(client_weights)
            aggregator_flex_model["weights_ids"].append(k.actor_id)

    return _collect_weights_
