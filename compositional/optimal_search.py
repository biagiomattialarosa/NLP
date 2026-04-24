import heapq
import time

import numpy as np

from . import formula as F
from . import optimal_heuristic
from . import utils
from . import heuristic_utils
from .heuristic_utils import INDEX_TUPLE_MAX, INDEX_TUPLE_MIN, INDEX_TUPLE_SAMPLE, INDEX_TUPLE_SUM
from compositional import mask_utils


def estimate_iou_frontier(*, frontier, label_mapping, heuristic, heuristic_info, max_improvement, disjoint_info, num_hits, max_size_mask, length, global_min_threshold):
    """Sort the frontier based on the heuristic info.

    Args:
        frontier (list): A list of candidate labels.
        heuristic_info (dict): A dictionary of heuristic info.
        quantities (tuple): A tuple of quantities.

    Returns:
        sorted_frontier (list): A sorted list of candidate labels.
    """
    # Sort the frontier based on the heuristic score and store both the node of the frontier and the score
    frontier_estimates = []
    minimum_threshold = global_min_threshold

    for node in frontier:

        # Estimate the heuristic score for the node
        max_score, min_score = optimal_heuristic.estimate_label_iou(heuristic,
            node, label_mapping, heuristic_info,  max_improvement=max_improvement, disjoint_info=disjoint_info, num_hits=num_hits, max_size_mask=max_size_mask,
            max_length=length, minimum_threshold=minimum_threshold)

        # If the minimum score is greater than the current minimum threshold, update the minimum threshold
        if min_score > minimum_threshold:
            minimum_threshold = min_score

        # Add nodes and their estimation to the current frontier
        label = node[2]
        for node_path in max_score:
            node_path_max_iou, node_path_next_op, _ , node_path_paths_to_expand = node_path
            if node_path_max_iou > 0:
                # Add the path to the frontier estimates if the IoU is greater than 0.
                # Note: filtering by minimum_threshold is done inside the estimate_label_iou function
                frontier_estimates.append((-node_path_max_iou, node_path_next_op, label, node_path_paths_to_expand, heuristic))
                if node_path_max_iou < minimum_threshold:
                    raise ValueError(f"Node {node} has a max IoU {node_path_max_iou} lower than the minimum threshold {minimum_threshold}. This should not happen.")
    if minimum_threshold > global_min_threshold:
        # Reduce the frontier with the new minimum threshold
        # This case covers the time where the minimum threshold is found later in the search
        frontier_estimates = reduce_frontier(
            frontier_estimates, minimum_threshold, label_mapping=label_mapping, heuristic_info=heuristic_info, max_improvement=max_improvement, disjoint_info=disjoint_info, num_hits=num_hits, max_size_mask=max_size_mask, length=length
        )
    return frontier_estimates, minimum_threshold

def compute_max_improvement(heuristic_info, label_mapping, max_size_mask, num_hits, length):
    # Decompose quantities
    seg_quantities, neuron_quantities, concepts_quantities = heuristic_info
    neuron_unique_tuple, neuron_common_tuple, neuron_coverable_tuple, neuron_sum_tuple, _, _   = neuron_quantities
    neuron_unique = neuron_unique_tuple[0]
    neuron_common = neuron_common_tuple[0]
    neuron_coverable = neuron_coverable_tuple[0]
    neuron_sum = neuron_sum_tuple[0]

    # Convert to numpy arrays for efficient computation

    common_intersection = []
    unique_intersection = []
    common_extras = []
    unique_extras = []
    common_uncovered = []
    unique_uncovered = []
    for index_concept in range(len(concepts_quantities)):
        label = F.Leaf(index_concept)
        common_intersection.append(heuristic_utils.get_quantity(label=label,
            concepts_quantities=concepts_quantities,
            label_mapping=label_mapping,
            quantity_name='common_intersection',
            quantity_type='max',
            quantity_scope='sample'
        ))
        unique_intersection.append(heuristic_utils.get_quantity(label=label,
            concepts_quantities=concepts_quantities,
            label_mapping=label_mapping,
            quantity_name='unique_intersection',
            quantity_type='max',
            quantity_scope='sample'
        ))
        common_extras.append(heuristic_utils.get_quantity(label=label,
            concepts_quantities=concepts_quantities,
            label_mapping=label_mapping,
            quantity_name='common_extras',
            quantity_type='max',
            quantity_scope='sample'
        ))
        unique_extras.append(heuristic_utils.get_quantity(label=label,
            concepts_quantities=concepts_quantities,
            label_mapping=label_mapping,
            quantity_name='unique_extras',
            quantity_type='max',
            quantity_scope='sample'
        ))
        common_uncovered.append(heuristic_utils.get_quantity(label=label,
            concepts_quantities=concepts_quantities,
            label_mapping=label_mapping,
            quantity_name='common_uncovered',
            quantity_type='max',
            quantity_scope='sample'
        ))
        unique_uncovered.append(heuristic_utils.get_quantity(label=label,
            concepts_quantities=concepts_quantities,
            label_mapping=label_mapping,
            quantity_name='unique_uncovered',
            quantity_type='max',
            quantity_scope='sample'
        ))
    common_intersection = np.stack(common_intersection, axis=0)
    unique_intersection = np.stack(unique_intersection, axis=0)
    common_extras = np.stack(common_extras, axis=0)
    unique_extras = np.stack(unique_extras, axis=0)
    common_uncovered = np.stack(common_uncovered, axis=0)
    unique_uncovered = np.stack(unique_uncovered, axis=0)

    sum_common_intersection = common_intersection.sum(axis=1)
    sum_unique_intersection = unique_intersection.sum(axis=1)
    sum_unique_extras = unique_extras.sum(axis=1)
    sum_common_extras = common_extras.sum(axis=1)
    sum_common_uncovered = common_uncovered.sum(axis=1)
    sum_unique_uncovered = unique_uncovered.sum(axis=1)

    # Cumulaative sums
    # sample_cum_sum_unique_intersection = np.cumsum(unique_intersection, axis=0)
    # sample_cum_sum_unique_extras = np.cumsum(unique_extras, axis=0)
    # cum_sum_unique_intersection = np.cumsum(sum_unique_intersection)
    # cum_sum_unique_extras = np.cumsum(sum_unique_extras)

    # For each sample, extract the top-k and bottom-k values for each quantity (per sample)
    topk_common_intersection = np.partition(common_intersection, -length, axis=0)[-length:]
    bottom_common_intersection = np.partition(common_intersection, length-1, axis=0)[:length]
    topk_unique_intersection = np.partition(unique_intersection, -length, axis=0)[-length:]
    bottom_unique_intersection = np.partition(unique_intersection, length-1, axis=0)[:length]
    topk_common_extras = np.partition(common_extras, -length, axis=0)[-length:]
    bottomk_common_extras = np.partition(common_extras, length-1, axis=0)[:length]
    topk_unique_extras = np.partition(unique_extras, -length, axis=0)[-length:]
    bottomk_unique_extras = np.partition(unique_extras, length-1, axis=0)[:length]
    topk_common_uncovered = np.partition(common_uncovered, -length, axis=0)[-length:]
    bottom_common_uncovered = np.partition(common_uncovered, length-1, axis=0)[:length]
    topk_unique_uncovered = np.partition(unique_uncovered, -length, axis=0)[-length:]
    bottom_unique_uncovered = np.partition(unique_uncovered, length-1, axis=0)[:length]
    # Sort each quantity by the sum of the values across samples (per concept)
    sorted_common_intersection = -np.sort(-sum_common_intersection)
    sorted_unique_intersection = -np.sort(-sum_unique_intersection)
    sorted_common_extras = -np.sort(-sum_common_extras)
    sorted_unique_extras = -np.sort(-sum_unique_extras)
    sorted_common_uncovered = -np.sort(-sum_common_uncovered)
    sorted_unique_uncovered = -np.sort(-sum_unique_uncovered)

    # Build max improvement up to the maximum length of explanation
    max_improvement_len = []
    for explanation_len in range(1, length+2):
        len_max_improvement = (
            (topk_common_intersection[:explanation_len].sum(axis=0),
            bottom_common_intersection[:explanation_len].sum(axis=0)),

            (topk_unique_intersection[:explanation_len].sum(axis=0),
            bottom_unique_intersection[:explanation_len].sum(axis=0)),

            (topk_common_extras[:explanation_len].sum(axis=0),
            bottomk_common_extras[:explanation_len].sum(axis=0)),

            (topk_unique_extras[:explanation_len].sum(axis=0),
            bottomk_unique_extras[:explanation_len].sum(axis=0)),

            (topk_common_uncovered[:explanation_len].sum(axis=0),
            bottom_common_uncovered[:explanation_len].sum(axis=0)),

            (topk_unique_uncovered[:explanation_len].sum(axis=0),
            bottom_unique_uncovered[:explanation_len].sum(axis=0)),

            # Sums
            ( sum(sorted_common_intersection[:explanation_len]),
                sum(sorted_common_intersection[-explanation_len:])),
            ( sum(sorted_unique_intersection[:explanation_len]),
                sum(sorted_unique_intersection[-explanation_len:])),
            ( sum(sorted_common_extras[:explanation_len]),
                sum(sorted_common_extras[-explanation_len:])),
            ( sum(sorted_unique_extras[:explanation_len]),
                sum(sorted_unique_extras[-explanation_len:])),
            ( sum(sorted_common_uncovered[:explanation_len]),
                sum(sorted_common_uncovered[-explanation_len:])),
            ( sum(sorted_unique_uncovered[:explanation_len]),
                sum(sorted_unique_uncovered[-explanation_len:])),
        )
        max_improvement_len.append(len_max_improvement)

    # Ensure the improvement respects the limits
    for explanation_len in range(len(max_improvement_len)):
        (t_common_intersection, b_common_intersection), (t_unique_intersection, b_unique_intersection), \
            (t_common_extras, b_common_extras), (t_unique_extras, b_unique_extras), (t_common_uncovered, b_common_uncovered), \
            (t_unique_uncovered, b_unique_uncovered), \
            (sum_t_common_intersection, sum_b_common_intersection), (sum_t_unique_intersection, sum_b_unique_intersection), \
            (sum_t_common_extras, sum_b_common_extras), (sum_t_unique_extras, sum_b_unique_extras), \
            (sum_t_common_uncovered, sum_b_common_uncovered), (sum_t_unique_uncovered, sum_b_unique_uncovered) = max_improvement_len[explanation_len] = max_improvement_len[explanation_len]

        # Intersection cannot exceed the number of hits
        t_common_intersection = np.minimum(t_common_intersection, neuron_common)
        t_common_intersection = np.minimum(t_common_intersection, neuron_coverable)
        t_unique_intersection = np.minimum(t_unique_intersection, neuron_unique)
        t_unique_intersection = np.minimum(t_unique_intersection, neuron_coverable)
        b_common_intersection = np.minimum(b_common_intersection, neuron_common)
        b_common_intersection = np.minimum(b_common_intersection, neuron_coverable)
        b_unique_intersection = np.minimum(b_unique_intersection, neuron_unique)
        b_unique_intersection = np.minimum(b_unique_intersection, neuron_coverable)
        # Extras cannot exceed the space left in the mask
        t_common_extras = np.minimum(t_common_extras, max_size_mask - neuron_sum)
        t_unique_extras = np.minimum(t_unique_extras, max_size_mask - neuron_sum)
        b_common_extras = np.minimum(b_common_extras, max_size_mask- neuron_sum)
        b_unique_extras = np.minimum(b_unique_extras, max_size_mask- neuron_sum)
        # Uncovered cannot exceed the number of hits    
        t_common_uncovered = np.minimum(t_common_uncovered, neuron_coverable)
        b_common_uncovered = np.minimum(b_common_uncovered, neuron_coverable)
        t_unique_uncovered = np.minimum(t_unique_uncovered, neuron_coverable)
        b_unique_uncovered = np.minimum(b_unique_uncovered, neuron_coverable)

        # Sums cannot exceed the number of hits or the maximum size of the mask
        tot_size = max_size_mask* common_intersection.shape[1]

        sum_t_common_intersection = min(sum_t_common_intersection, neuron_common.sum())
        sum_t_common_intersection = min(sum_t_common_intersection, neuron_coverable.sum())
        sum_b_common_intersection = min(sum_b_common_intersection, neuron_common.sum())
        sum_b_common_intersection = min(sum_b_common_intersection, neuron_coverable.sum())
        sum_t_unique_intersection = min(sum_t_unique_intersection, neuron_unique.sum())
        sum_t_unique_intersection = min(sum_t_unique_intersection, neuron_coverable.sum())
        sum_b_unique_intersection = min(sum_b_unique_intersection, neuron_unique.sum())
        sum_b_unique_intersection = min(sum_b_unique_intersection, neuron_coverable.sum())
        sum_t_common_extras = min(sum_t_common_extras, tot_size - num_hits)
        sum_b_common_extras = min(sum_b_common_extras, tot_size - num_hits)
        sum_t_unique_extras = min(sum_t_unique_extras, tot_size - num_hits)
        sum_b_unique_extras = min(sum_b_unique_extras, tot_size - num_hits)
        sum_t_common_uncovered = min(sum_t_common_uncovered, neuron_coverable.sum())
        sum_b_common_uncovered = min(sum_b_common_uncovered, neuron_coverable.sum())
        sum_t_unique_uncovered = min(sum_t_unique_uncovered, neuron_coverable.sum())
        sum_b_unique_uncovered = min(sum_b_unique_uncovered, neuron_coverable.sum())

        # Update the max improvement length
        max_improvement_len[explanation_len] = (
            (t_common_intersection, b_common_intersection, sum_t_common_intersection, sum_b_common_intersection),
            (t_unique_intersection, b_unique_intersection, sum_t_unique_intersection, sum_b_unique_intersection),
            (t_common_extras, b_common_extras, sum_t_common_extras, sum_b_common_extras),
            (t_unique_extras, b_unique_extras, sum_t_unique_extras, sum_b_unique_extras),
            (t_common_uncovered, b_common_uncovered, sum_t_common_uncovered, sum_b_common_uncovered),
            (t_unique_uncovered, b_unique_uncovered, sum_t_unique_uncovered, sum_b_unique_uncovered)
        )
    # Reshape the max improvement so that for each quantity we have a list of lenghts
    reshaped_max_improvement_len = []
    for quantity in range(len(max_improvement_len[0])):
        reshaped_quantity = []
        for explanation_len in range(len(max_improvement_len)):
            t, b, sum_t, sum_b = max_improvement_len[explanation_len][quantity]
            reshaped_quantity.append((t, b, sum_t, sum_b))
        reshaped_max_improvement_len.append(reshaped_quantity)
    #return reshaped_max_improvement_len, (cum_sum_unique_intersection, cum_sum_unique_extras)
    return reshaped_max_improvement_len, (None, None)

def check_if_present(index_op, list_to_check, list_to_add):
    for items in list_to_check[index_op]:
        if len(items) == len(list_to_add) and set(items) == set(list_to_add):
            # If the items are the same, we do not add them
            return True
    return False

def formula_all_disjoint(label, disjoint_info):
    """Check if the formula contains all disjoint labels."""
    vals = label.get_vals()
    assert len(vals) == len(label)
    for i in range(len(vals)):
        for j in range(i+1, len(vals)):
            if not optimal_heuristic.are_disjoint(F.Leaf(vals[i]), F.Leaf(vals[j]), disjoint_info):
                return False
    return True

def expand_node(frontier_node, *, candidate_labels, max_length):
    _, next_op, label, paths_to_expand, _ = frontier_node
    next_frontier = []
    # Info useful to avoid logical equivalence
    if len(label) > 1:
        last_val = label.get_vals()[0]
        last_op = label.get_ops()[0]
    for candidate_term in candidate_labels:
        # Skip the candidate term if it is already in the label
        if candidate_term.val in label.get_vals():
            continue
        # Impose order to avoid logical equivalence
        if next_op != 'NOT' and isinstance(label, F.Leaf) and candidate_term.val < label.val:
            continue
        elif len(label) > 1:
            if next_op == 'NOT' and last_op == 'AND' and isinstance(label.right, F.Not):
                if candidate_term.val < last_val:
                    continue
            elif last_op == next_op:
                if candidate_term.val < last_val:
                    continue
        
        # Build the candidate formula based on the next operation
        if next_op == "OR":
            candidate_formula = F.Or(label, candidate_term)
        elif next_op == "AND":
            candidate_formula = F.And(label, candidate_term)
        elif next_op == "NOT":
            candidate_formula = F.And(label, F.Not(candidate_term))
        else:
            raise ValueError(f"Unknown operation {next_op}")

        # Add here constraints about the ops, we need the maximum length of the formula
        # if candidate_formula not in expanded_nodes:
        if len(candidate_formula) == max_length:
            next_frontier.append((None, 'INDIVIDUAL' , candidate_formula, [], None))
        else:
            current_ops = set(candidate_formula.get_ops())
            available_spots = max_length - len(candidate_formula)
            feasibles_paths = [[],[],[],[]]
            
            assert len(paths_to_expand) == 4

            # Given the current paths, mantain only the feasible ones (i.e., the ones that can be fully explored given the steps left)
            for path in paths_to_expand:
                for path_ops in path: 
                    path_missing_ops = set(path_ops) - current_ops

                    assert available_spots >= len(path_missing_ops), f"Available spots: {available_spots}, Missing ops: {path_missing_ops}, Current ops: {current_ops} Formula: {candidate_formula}, Node: {frontier_node}"
                    
                    if len(path_missing_ops) == available_spots:
                        # In this case we have the same number of missing operations and available spaces
                        # The next operator to add needs to be one of the missing operations
                        ops_to_add = list(path_missing_ops)
                    else:
                        # In this case we have more spaces available then missing operations
                        # We can freely choose the next operator
                        ops_to_add = path_ops

                    for op_to_add in ops_to_add:
                        if op_to_add == 'OR':
                            index_op = optimal_heuristic.INDEX_OR
                        elif op_to_add == 'AND':
                            index_op = optimal_heuristic.INDEX_AND
                        elif op_to_add == 'NOT':
                            index_op = optimal_heuristic.INDEX_NOT
                        else:
                            raise ValueError(f"Unknown operation {ops_to_add}")

                        # Add this path if it is not already present in the feasible ops
                        if not check_if_present(index_op, feasibles_paths, ops_to_add):
                            feasibles_paths[index_op].append(list(ops_to_add))
            next_frontier.append((None, None, candidate_formula, feasibles_paths, None))
    return next_frontier

def reduce_frontier(frontier, global_minimum_threshold, label_mapping, heuristic_info, max_improvement, disjoint_info, num_hits, max_size_mask, length):
    """Reduce the frontier based on the global minimum threshold.

    Args:
        frontier (list): A list of candidate labels.
        global_minimum_threshold (float): The global minimum threshold.

    Returns:
        reduced_frontier (list): A reduced list of candidate labels.
    """
    reduced_frontier = []
    for node in frontier:
        iou = node[0]
        if -iou >= global_minimum_threshold:
            reduced_frontier.append(node)
    heapq.heapify(reduced_frontier)
    return reduced_frontier


def propagate_information(label, threshold, masks, bitmaps, frontier, heuristic, heuristic_info, label_mapping, max_improvement, disjoint_info, num_hits, max_size_mask, length):
    # No need to propagate information for single labels and disjoint formulas
    if len(label) <= 1 or formula_all_disjoint(label.left, disjoint_info):
        return frontier, 0
    ancestors_info = compute_ancestors_quantities(label, masks, heuristic_info, bitmaps)
    new_heuristic_info, new_label_mapping = update_heuristic_info(ancestors_info, heuristic_info, label_mapping, max_length=length)
    new_frontier, tot_updated = update_frontier_by_ancestors(frontier, ancestors_info.keys(), threshold, heuristic_name=heuristic, label_mapping=new_label_mapping, heuristic_info=new_heuristic_info, max_improvement=max_improvement, disjoint_info=disjoint_info, num_hits=num_hits, max_size_mask=max_size_mask, max_length=length)
    return new_frontier, tot_updated

def update_frontier(past_frontier, new_nodes, label_mapping, heuristic, heuristic_info, max_improvement, disjoint_info, num_hits, max_size_mask, length, global_min_threshold):
   
    # Estimate the IoU of the new frontier
    new_frontier, local_minimum_threshold = estimate_iou_frontier(
        frontier=new_nodes, label_mapping=label_mapping,
        heuristic=heuristic, heuristic_info=heuristic_info,
        max_improvement=max_improvement,
        num_hits=num_hits, max_size_mask=max_size_mask,
        length=length, global_min_threshold=global_min_threshold,
        disjoint_info=disjoint_info
    )

    # The new frontier computed an higher minimum threshold
    if local_minimum_threshold > global_min_threshold:
        # Update the global minimum threshold
        global_min_threshold = local_minimum_threshold
        # Reduce the past frontier based on the new global minimum threshold
        if past_frontier is not None:
            past_frontier = reduce_frontier(
                past_frontier, global_min_threshold, label_mapping=label_mapping, heuristic_info=heuristic_info, max_improvement=max_improvement, disjoint_info=disjoint_info, num_hits=num_hits, max_size_mask=max_size_mask, length=length
            )
    # Merge the new nodes with the past frontier
    if past_frontier is not None and len(past_frontier) > 0:
        # Merge assumes that the input iterables are sorted
        for new_node in new_frontier:
            heapq.heappush(past_frontier, new_node)
        sorted_frontier = past_frontier
    else:
        # Initialize frontier as a queue
        heapq.heapify(new_frontier)
        sorted_frontier = new_frontier
    return sorted_frontier, global_min_threshold

def compute_exact_iou(label_mask, bitmaps, heuristic_info):
    seg_quantities, neuron_quantities, concepts_quantities = heuristic_info
    (neuron_unique, _), (neuron_common, _), _, _, _ , _ = neuron_quantities
    common_elements, unique_elements, uncoverable_elements = seg_quantities
    unique_elements = unique_elements.to(bitmaps.device)
    common_elements = common_elements.to(bitmaps.device)
    uncoverable_elements = uncoverable_elements.to(bitmaps.device)
    concept_quantities = heuristic_utils.compute_quantities_vector(label_mask, bitmaps, common_elements, unique_elements, neuron_common, neuron_unique)
    c_common_intersection, c_unique_intersection, c_common_extras, c_unique_extras, _, _ = concept_quantities
    label_iou = (c_common_intersection.sum() + c_unique_intersection.sum()) / (bitmaps.sum() + c_common_extras.sum() + c_unique_extras.sum())
    return label_iou.item()

def compute_iou_from_concept_info(concept_quantities, num_hits):
    common_intersection_sum = heuristic_utils.get_quantity(
        label=None, concepts_quantities=concept_quantities,
        quantity_name='common_intersection', quantity_type='max',
        quantity_scope='sum'
    )
    unique_intersection_sum = heuristic_utils.get_quantity(
        label=None, concepts_quantities=concept_quantities,
        quantity_name='unique_intersection', quantity_type='max',
        quantity_scope='sum'
    )
    common_extras_sum = heuristic_utils.get_quantity(
        label=None, concepts_quantities=concept_quantities,
        quantity_name='common_extras', quantity_type='max',
        quantity_scope='sum'
    )
    unique_extras_sum = heuristic_utils.get_quantity(
        label=None, concepts_quantities=concept_quantities,
        quantity_name='unique_extras', quantity_type='max',
        quantity_scope='sum'
    )
    #print(f"Common Intersection: {common_intersection_sum}, Unique Intersection: {unique_intersection_sum}, Common Extras: {common_extras_sum}, Unique Extras: {unique_extras_sum}")
    label_iou = (common_intersection_sum + unique_intersection_sum) / (num_hits + common_extras_sum + unique_extras_sum)
    return label_iou


def update_heuristic_info(ancestors_info, heuristic_info, label_mapping, max_length):
    seg_quantities, neuron_quantities, concepts_quantities = heuristic_info

    for ancestor, (ancestor_quantities, _) in ancestors_info.items():
        if len(ancestor) >= max_length:
            # In this case the information is not useful, we skip it to save memory
            continue
        if isinstance(ancestor, F.Not) and ancestor.val in label_mapping:
            # We can already compute its quantities
            continue
        elif ancestor not in label_mapping:
            label_mapping[ancestor] = len(label_mapping)
            concepts_quantities.append(ancestor_quantities)
        else:
            index_node = label_mapping[ancestor]
            # If the ancestor is already present, we update the concept quantities
            concepts_quantities[index_node] = ancestor_quantities
    
    updated_heuristic_info = (
        seg_quantities,
        neuron_quantities,
        concepts_quantities
    )
    return updated_heuristic_info, label_mapping

def apply_distributive_property(label):
    """Apply the Distributive Property to the label if possible."""
    if len(label) < 3:
        return label
    label_left = label.left
    num_left_op = len(set(label_left.get_ops()))
    if num_left_op == 1:
        external_op = label.__class__
        internal_op = label_left.__class__   
        if external_op == internal_op:
            return label
       # We can apply the Distributive Property
        vals_left = label_left.get_vals()
        chain = []
        for val in vals_left:
            inner_formula = external_op(F.Leaf(val), label.right)
            chain.append(inner_formula)
        for i in range(len(chain)-1):
            final_formula = internal_op(chain[i], chain[i+1])
        return final_formula
    else:
        return label

def compute_ancestors_quantities(label, masks, heuristic_info, bitmaps):
    ancestors_dict = mask_utils.get_ancestor_masks(
            label, masks
        )
    seg_quantities, neuron_quantities, concepts_quantities = heuristic_info
    (neuron_unique, _), (neuron_common, _),  _, neuron_sum, _, _   = neuron_quantities
    common_elements, unique_elements, uncoverable_elements = seg_quantities
    unique_elements = unique_elements.to(bitmaps.device)
    common_elements = common_elements.to(bitmaps.device)
    uncoverable_elements = uncoverable_elements.to(bitmaps.device)
    ancestors_info = {}
    for ancestor in ancestors_dict.keys():
        label_mask = utils.sparse_to_torch(mask_utils.get_formula_mask(
                    ancestor, masks
                )).to(bitmaps.device)
        concept_quantities = heuristic_utils.compute_quantities_vector(label_mask, bitmaps, common_elements, unique_elements, neuron_common, neuron_unique)
        concept_info = heuristic_utils.get_concept_info(concept_quantities)
        ancestor_iou = compute_iou_from_concept_info(concept_info, num_hits=neuron_sum[1])
        ancestors_info[ancestor] = (concept_info, ancestor_iou)
    return ancestors_info

def update_frontier_by_ancestors(frontier, ancestors, threshold, heuristic_name, label_mapping, heuristic_info, max_improvement, disjoint_info, num_hits, max_size_mask, max_length):
    sorted_ancestors = sorted(ancestors, key=lambda x: len(x))
    sorted_ancestors = [a for a in sorted_ancestors if len(a) > 1]
    new_frontier = []
    found_ancestor = False
    tot_updated = 0
    for frontier_node in frontier:
        len_node = len(frontier_node[2])
        old_iou = -frontier_node[0]
        found_ancestor = False
        for ancestor_label in sorted_ancestors:
            if found_ancestor:
                # If we already found an ancestor, we can stop checking the others since the other matches will be with shorter common ancestors
                break
            if len(ancestor_label) > len_node:
                # If the ancestor is longer than the node, there cannot be common ancestors
                continue
            elif len(ancestor_label) == len_node and ancestor_label == frontier_node[2]:
                new_max, new_min = optimal_heuristic.update_optimal_label_iou(heuristic_name='sample', node=frontier_node, label_mapping=label_mapping, heuristic_info=heuristic_info, max_improvement=max_improvement, disjoint_info=disjoint_info, num_hits=num_hits, max_size_mask=max_size_mask, max_length=max_length)
                if new_max != old_iou:
                    tot_updated += 1
                if new_max >= threshold:
                    new_frontier.append((-new_max, frontier_node[1], frontier_node[2], frontier_node[3], heuristic_name ))
                # We can exit because the other matches will be with shorter common ancestors
                found_ancestor = True
            elif len(ancestor_label) == len_node and ancestor_label != frontier_node[2]:
                # Similarly to the previous case, there cannot be common ancestors 
                continue
            else:
                # Case where the ancestor is shorter than the node
                if ancestor_label in frontier_node[2].tree_path():
                    new_max, new_min = optimal_heuristic.update_optimal_label_iou(heuristic_name='sample', node=frontier_node, label_mapping=label_mapping, heuristic_info=heuristic_info, max_improvement=max_improvement, disjoint_info=disjoint_info, num_hits=num_hits, max_size_mask=max_size_mask, max_length=max_length)
                    if new_max != old_iou:
                        tot_updated += 1
                    if new_max >= threshold:
                        new_frontier.append((-new_max, frontier_node[1], frontier_node[2], frontier_node[3], heuristic_name))
                    # We can exit because the other matches will be with shorter common ancestors
                    found_ancestor = True
        if not found_ancestor:
            # If no ancestor was found, we add the node to the past frontier
            new_frontier.append(frontier_node)
    heapq.heapify(new_frontier)
    return new_frontier, tot_updated

def perform_exhaustive_heuristic_search(
    heuristicA,
    heuristic_info,
    disjoint_info,
    masks,
    bitmaps,
    num_hits,
    *,
    max_size_mask,
    length=3,
):
    """Compute the heuristic score for each concept in the candidate_concepts
    list for the given bitmaps.

    Args:
        bitmaps (torch.Tensor): A tensor of shape (N, H, W) where N is the
            number of sample.
        masks (dict): A dictionary of concept masks. Each mask is a tensor of
            shape (H, W).
        candidate_concepts (list): A list of candidate concepts.

    Returns:
        heuristic_rank (dict): A dictionary of concept scores. Each score is
            a float.
    """
    
    num_hits = num_hits.item()

    # Candidate concepts
    candidate_labels = [F.Leaf(c) for c in range(len(masks))]
    label_mapping = {F.Leaf(c): c for c in range(len(masks))}
    
    # Max improvement for this neuron
    max_improvement = compute_max_improvement(heuristic_info, label_mapping, max_size_mask, num_hits, length)

    # Initialize the frontier with all the candidate labels
    current_frontier = [(0.0, None, k, None, None) for k in label_mapping.keys()]
    if len(current_frontier) == 0:
        return None, 0, 0, 0

    best_label, best_iou, tot_visited, tot_expanded, tot_estimated = explore_frontier(
                    current_frontier,
                    'sum',
                    candidate_labels,
                    heuristic_info,
                    disjoint_info,
                    0.0,
                    label_mapping,
                    max_improvement,
                    masks,
                    bitmaps,
                    num_hits,
                    max_size_mask=max_size_mask,
                    length=length,
    )
    
    return best_label, best_iou, tot_visited, tot_expanded, tot_estimated


def explore_frontier(
    initial_frontier,
    heuristic_name,
    candidate_concepts,
    heuristic_info,
    disjoint_info,
    minimum_threshold,
    label_mapping,
    max_improvement,
    masks,
    bitmaps,
    num_hits,
    *,
    max_size_mask,
    length=3,
):
    """Compute the heuristic score for each concept in the candidate_concepts
    list for the given bitmaps.

    Args:
        bitmaps (torch.Tensor): A tensor of shape (N, H, W) where N is the
            number of sample.
        masks (dict): A dictionary of concept masks. Each mask is a tensor of
            shape (H, W).
        candidate_concepts (list): A list of candidate concepts.

    Returns:
        heuristic_rank (dict): A dictionary of concept scores. Each score is
            a float.
    """
      
    
    # statistics
    expanded_nodes = 0
    estimated_nodes = 0
    visited_nodes = 0

    # Aux
    best_results = (0.0, None) # (IoU, label)
    recent_nodes = []
    visited  = []
    recent_e_iou = 2

    #DEBUG INFO
    updated_because_ancestor = 0
    skipped_because_recent = 0
    update_distributive = 0

    # Initialize frontiers
    current_frontier, minimum_threshold = update_frontier(
        past_frontier=None, new_nodes=initial_frontier, 
        label_mapping=label_mapping, heuristic=heuristic_name, 
        heuristic_info=heuristic_info, max_improvement=max_improvement,
        num_hits=num_hits, max_size_mask=max_size_mask,
        length=length, global_min_threshold=minimum_threshold, disjoint_info=disjoint_info
    )
    done = len(current_frontier) == 0
    while not done:
        node = heapq.heappop(current_frontier)
        e_node, next_op_node, label_node, _, node_heuristic = node
        
        if -e_node < minimum_threshold:
            done = len(current_frontier) == 0
            continue

        print(f"Current node: {label_node}, Heuristic: {-round(e_node, 2)}, Minimum Threshold: {round(minimum_threshold, 2)}, Frontier Size: {len(current_frontier)}, Expanded Nodes: {expanded_nodes}, Estimated Nodes: {estimated_nodes}, Visited Nodes: {visited_nodes}", end='\r')
        # Compute Sample Estimate
        if node_heuristic != 'sample':
            new_max, _ = optimal_heuristic.update_optimal_label_iou(heuristic_name='sample', node=node, label_mapping=label_mapping, heuristic_info=heuristic_info, max_improvement=max_improvement, disjoint_info=disjoint_info, num_hits=num_hits, max_size_mask=max_size_mask, max_length=length)
            if new_max < -e_node:
                # If the estimate is lower than the previous one
                if new_max > minimum_threshold:
                    # we update it and reinsert the node in the frontier
                    heapq.heappush(current_frontier, (-new_max, next_op_node, label_node, node[3], 'sample'))
                done = len(current_frontier) == 0
                continue
        
        # For label of size >=3 we can try to apply the Distributive Property
        transformed_label = apply_distributive_property(label_node)
        if transformed_label != label_node:
            node_after_distr = (node[0], node[1], transformed_label, node[3])
            new_max, _ = optimal_heuristic.update_optimal_label_iou(heuristic_name='sample', node=node_after_distr, label_mapping=label_mapping, heuristic_info=heuristic_info, max_improvement=max_improvement, disjoint_info=disjoint_info, num_hits=num_hits, max_size_mask=max_size_mask, max_length=length)
            if new_max < -e_node:
                update_distributive += 1
                if new_max > minimum_threshold:
                    # If the estimate is lower than the previous one we update it and reinsert the node in the frontier
                    heapq.heappush(current_frontier, (-new_max, next_op_node, label_node, node[3], 'sample'))
                done = len(current_frontier) == 0
                continue

        # Recent node mechanism to avoid expanding the same node multiple times
        if -e_node >= recent_e_iou:
            if node in recent_nodes:
                done = len(current_frontier) == 0
                skipped_because_recent += 1
                continue
            else:
                recent_nodes.append(node)                          
        else:
            recent_nodes = [node]
            recent_e_iou = -e_node

        if next_op_node == 'INDIVIDUAL':
            if label_node not in visited:
                iou, current_frontier, minimum_threshold, prop_upda = analyze_final_node(
                    label_node, masks, bitmaps, heuristic_info, minimum_threshold,
                        current_frontier, heuristic_name, label_mapping, max_improvement,
                        disjoint_info, num_hits, max_size_mask, length)
                updated_because_ancestor += prop_upda
                # Add the node to the visited nodes
                visited_nodes += 1
                #visited.append(label_node)

                if iou > best_results[0]:
                    best_results = (iou, label_node)
                elif iou == best_results[0]:
                    if len(label_node) < len(best_results[1]):
                        # We prefer shorter labels in case of equal IoU
                        best_results = (iou, label_node)

            done = len(current_frontier) == 0
            continue
        
        # Expand the node to get the next frontier
        next_frontier = expand_node(node, candidate_labels=candidate_concepts, max_length=length)
        expanded_nodes += 1
        estimated_nodes += len(next_frontier)

        # Compute the estimation for the next frontier
        current_frontier, minimum_threshold = update_frontier(
            past_frontier=current_frontier, new_nodes=next_frontier, 
            label_mapping=label_mapping, heuristic=heuristic_name, 
            heuristic_info=heuristic_info, max_improvement=max_improvement,
            num_hits=num_hits, max_size_mask=max_size_mask,
            length=length, global_min_threshold=minimum_threshold, disjoint_info=disjoint_info
        )

        done = len(current_frontier) == 0
    best_iou = best_results[0]
    best_label = best_results[1]
    return best_label, best_iou, visited_nodes, expanded_nodes, estimated_nodes

def analyze_final_node(label_node, masks, bitmaps, heuristic_info, minimum_threshold,
                       current_frontier, heuristic_name, label_mapping, max_improvement,
                       disjoint_info, num_hits, max_size_mask, length):
    # This is the case where the formula has no operations to expand
    # Compute Label Mask
    label_mask = utils.sparse_to_torch(mask_utils.get_formula_mask(
            label_node, masks
        )).to(bitmaps.device)
    # Compute the exact IoU for the label node
    iou = compute_exact_iou(label_mask, bitmaps, heuristic_info)

    current_frontier, updated = propagate_information(
                label_node, minimum_threshold, masks, bitmaps, current_frontier, heuristic_name, heuristic_info, label_mapping,
                max_improvement, disjoint_info, num_hits, max_size_mask, length
            )
    # If the IoU is greater than the minimum threshold, we update the minimum threshold
    if iou > minimum_threshold:
        minimum_threshold = iou

        if len(current_frontier) > 0:
            # Reduce the current frontier based on the new minimum threshold
            current_frontier = reduce_frontier(
                current_frontier, minimum_threshold, label_mapping=label_mapping, heuristic_info=heuristic_info, max_improvement=max_improvement, disjoint_info=disjoint_info, num_hits=num_hits, max_size_mask=max_size_mask, length=length
            )
    return iou, current_frontier, minimum_threshold, updated