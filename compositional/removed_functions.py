import heapq
import time

import torch
import numpy as np

from . import formula as F
from . import optimal_heuristic
from . import utils


def estimate_iou_frontier(*, frontier, label_mapping, heuristic, heuristic_info, max_improvement, disjoint_info, num_hits, max_size_mask, length, global_min_threshold, debug=False):
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
    start_time = time.time()
    frontier_nodes_times = []
    added_to_frontier = 0
    reduced_frontier = False

    # Debugging Info
    if debug:
        debug_info = {
            'esti_quantities_time': [],
            'individual_estimation_time': [],
            'or_chain_estimation_time': [],
            'and_chain_estimation_time': [],
            'and_not_chain_estimation_time': [],
            'and_or_chain_estimation_time': [],
            'comb_or_andnot_chain_estimation_time': []
        }
    
    for node in frontier:
        if debug:
            before_time = time.time()
            max_score, min_score, d_info = optimal_heuristic.get_heuristic_score(
            node, label_mapping, heuristic, heuristic_info, max_improvement, disjoint_info, num_hits=num_hits, max_size_mask=max_size_mask,
            max_length=length, minimum_threshold=minimum_threshold, debug=debug)
            after_time = time.time()
            frontier_nodes_times.append(after_time - before_time)
            # Debugging info
            for k,v in d_info.items():
                debug_info[k].append(v)
        else:
            # Estimate the heuristic score for the node
            max_score, min_score = optimal_heuristic.get_heuristic_score(
                node, label_mapping, heuristic, heuristic_info, max_improvement, disjoint_info, num_hits=num_hits, max_size_mask=max_size_mask,
                max_length=length, minimum_threshold=minimum_threshold, debug=debug)

        # If the minimum score is greater than the current minimum threshold, update the minimum threshold
        if min_score > minimum_threshold:
            minimum_threshold = min_score

        # Add nodes and their estimation to the current frontier
        label = node[2]
        for node_path in max_score:
            node_path_max_iou, node_path_next_op, _ , node_path_paths_to_expand = node_path
            if node_path_max_iou > 0:
                # Add the path to the frontier estimates if the IoU is greater than 0.
                # Note: filtering by minimum_threshold is done inside the get_heuristic_score function
                frontier_estimates.append((-node_path_max_iou, node_path_next_op, label, node_path_paths_to_expand))
                added_to_frontier += 1
                if node_path_max_iou < minimum_threshold:
                    raise ValueError(f"Node {node} has a max IoU {node_path_max_iou} lower than the minimum threshold {minimum_threshold}. This should not happen.")
    parsing_frontier_time = time.time()
    if minimum_threshold > global_min_threshold:
        # Reduce the frontier with the new minimum threshold
        # This case covers the time where the minimum threshold is found later in the search
        frontier_estimates = reduce_frontier(
            frontier_estimates, minimum_threshold)
        reduced_frontier = True
    reduce_frontier_time = time.time()
    end_time = time.time()

    if debug:
        # Add debug info
        debug_info["added_to_frontier"] = added_to_frontier
        debug_info["parsing_frontier_time"] = parsing_frontier_time - start_time
        debug_info["estimate_reduced_frontier"] = reduced_frontier
        debug_info["estimate_reduce_frontier_time"] = reduce_frontier_time - parsing_frontier_time
        debug_info["time_estimate_iou_frontier"] = end_time - start_time
        debug_info["avg_time_frontier_node"] = sum(frontier_nodes_times) / len(frontier_nodes_times) if len(frontier_nodes_times) > 0 else 0
        debug_info["median_time_frontier_node"] = np.median(frontier_nodes_times)
        return frontier_estimates, minimum_threshold, debug_info
    else:
        return frontier_estimates, minimum_threshold


def compute_max_improvement(heuristic_info, max_size_mask, num_hits, length):
    # Decompose quantities
    seg_quantities, neuron_quantities, concepts_quantities = heuristic_info
    neuron_unique_tuple, neuron_common_tuple, neuron_coverable_mask, neuron_coverable_tuple, neuron_sum_tuple  = neuron_quantities
    common_intersection_tuple, unique_intersection_tuple, common_extras_tuple, unique_extras_tuple, uncovered_tuple = concepts_quantities
    neuron_unique = neuron_unique_tuple[0]
    neuron_common = neuron_common_tuple[0]
    neuron_coverable = neuron_coverable_tuple[0]
    neuron_sum = neuron_sum_tuple[0]
    common_intersection = common_intersection_tuple[0]
    unique_intersection = unique_intersection_tuple[0]
    common_extras = common_extras_tuple[0]
    unique_extras = unique_extras_tuple[0]
    uncovered = uncovered_tuple[0]

    # Convert to numpy arrays for efficient computation
    common_intersection = np.stack(common_intersection, axis=0)
    unique_intersection = np.stack(unique_intersection, axis=0)
    common_extras = np.stack(common_extras, axis=0)
    unique_extras = np.stack(unique_extras, axis=0)
    uncovered = np.stack(uncovered, axis=0)

    # For each sample, extract the top-k and bottom-k values for each quantity (per sample)
    topk_common_intersection = np.partition(common_intersection, -length, axis=0)[-length:]
    bottom_common_intersection = np.partition(common_intersection, length-1, axis=0)[:length]
    topk_unique_intersection = np.partition(unique_intersection, -length, axis=0)[-length:]
    bottom_unique_intersection = np.partition(unique_intersection, length-1, axis=0)[:length]
    topk_common_extras = np.partition(common_extras, -length, axis=0)[-length:]
    bottomk_common_extras = np.partition(common_extras, length-1, axis=0)[:length]
    topk_unique_extras = np.partition(unique_extras, -length, axis=0)[-length:]
    bottomk_unique_extras = np.partition(unique_extras, length-1, axis=0)[:length]
    topk_uncovered = np.partition(uncovered, -length, axis=0)[-length:]
    bottom_uncovered = np.partition(uncovered, length-1, axis=0)[:length]
    # Sort each quantity by the sum of the values across samples (per concept)
    sorted_common_intersection = -np.sort(-common_intersection.sum(axis=1))
    sorted_unique_intersection = -np.sort(-unique_intersection.sum(axis=1))
    sorted_common_extras = -np.sort(-common_extras.sum(axis=1))
    sorted_unique_extras = -np.sort(-unique_extras.sum(axis=1))
    sorted_uncovered = -np.sort(-uncovered.sum(axis=1))

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

            (topk_uncovered[:explanation_len].sum(axis=0),
            bottom_uncovered[:explanation_len].sum(axis=0)),

            # Sums
            ( sum(sorted_common_intersection[:explanation_len]),
                sum(sorted_common_intersection[-explanation_len:])),
            ( sum(sorted_unique_intersection[:explanation_len]),
                sum(sorted_unique_intersection[-explanation_len:])),
            ( sum(sorted_common_extras[:explanation_len]),
                sum(sorted_common_extras[-explanation_len:])),
            ( sum(sorted_unique_extras[:explanation_len]),
                sum(sorted_unique_extras[-explanation_len:])),
            ( sum(sorted_uncovered[:explanation_len]),
                sum(sorted_uncovered[-explanation_len:])),
        )
        max_improvement_len.append(len_max_improvement)

    # Ensure the improvement respects the limits
    for explanation_len in range(len(max_improvement_len)):
        (t_common_intersection, b_common_intersection), (t_unique_intersection, b_unique_intersection), \
            (t_common_extras, b_common_extras), (t_unique_extras, b_unique_extras), (t_uncovered,b_uncovered), \
            (sum_t_common_intersection, sum_b_common_intersection), (sum_t_unique_intersection, sum_b_unique_intersection), \
            (sum_t_common_extras, sum_b_common_extras), (sum_t_unique_extras, sum_b_unique_extras), \
            (sum_t_uncovered, sum_b_uncovered) = max_improvement_len[explanation_len] = max_improvement_len[explanation_len]
        
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
        t_uncovered = np.minimum(t_uncovered, neuron_coverable)
        b_uncovered = np.minimum(b_uncovered, neuron_coverable)

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
        sum_t_uncovered = min(sum_t_uncovered, neuron_coverable.sum())
        sum_b_uncovered = min(sum_b_uncovered, neuron_coverable.sum())
        
        # Update the max improvement length
        max_improvement_len[explanation_len] = (
            (t_common_intersection, b_common_intersection, sum_t_common_intersection, sum_b_common_intersection),
            (t_unique_intersection, b_unique_intersection, sum_t_unique_intersection, sum_b_unique_intersection),
            (t_common_extras, b_common_extras, sum_t_common_extras, sum_b_common_extras),
            (t_unique_extras, b_unique_extras, sum_t_unique_extras, sum_b_unique_extras),
            (t_uncovered, b_uncovered, sum_t_uncovered, sum_b_uncovered)
        )
    # Reshape the max improvement so that for each quantity we have a list of lenghts
    reshaped_max_improvement_len = []
    for quantity in range(len(max_improvement_len[0])):
        reshaped_quantity = []
        for explanation_len in range(len(max_improvement_len)):
            t, b, sum_t, sum_b = max_improvement_len[explanation_len][quantity]
            reshaped_quantity.append((t, b, sum_t, sum_b))
        reshaped_max_improvement_len.append(reshaped_quantity)
    return reshaped_max_improvement_len

def check_if_present(index_op, list_to_check, list_to_add):
    for items in list_to_check[index_op]:
        if len(items) == len(list_to_add) and set(items) == set(list_to_add):
            # If the items are the same, we do not add them
            return True
    return False

def expand_node(frontier_node, candidate_labels, max_length):
    _, next_op, label, paths_to_expand = frontier_node
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
            if last_op == next_op:
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
            next_frontier.append((None, 'INDIVIDUAL' , candidate_formula, []))
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
            next_frontier.append((None, None, candidate_formula, feasibles_paths))
    return next_frontier

def reduce_frontier(frontier, global_minimum_threshold):
    """Reduce the frontier based on the global minimum threshold.

    Args:
        frontier (list): A list of candidate labels.
        global_minimum_threshold (float): The global minimum threshold.

    Returns:
        reduced_frontier (list): A reduced list of candidate labels.
    """
    reduced_frontier = []
    for (iou, next_op, label, paths) in frontier:
        if -iou >= global_minimum_threshold:
            reduced_frontier.append((iou, next_op, label, paths))
    return reduced_frontier

def update_frontier(*, past_frontier, new_nodes, label_mapping, heuristic, heuristic_info, max_improvement, disjoint_info, num_hits, max_size_mask, length, global_min_threshold, debug=False):
    outer_reduced = False
    if debug:
        new_frontier, local_minimum_threshold, debug_info = estimate_iou_frontier(
            frontier=new_nodes, label_mapping=label_mapping, 
            heuristic=heuristic, heuristic_info=heuristic_info, 
            max_improvement=max_improvement, 
            num_hits=num_hits, max_size_mask=max_size_mask, 
            length=length, global_min_threshold=global_min_threshold,
            disjoint_info=disjoint_info, debug=debug
        )
        estimate_iou_frontier_time = time.time()

    else:
        # Estimate the IoU of the new frontier
        new_frontier, local_minimum_threshold = estimate_iou_frontier(
            frontier=new_nodes, label_mapping=label_mapping, 
            heuristic=heuristic, heuristic_info=heuristic_info, 
            max_improvement=max_improvement, 
            num_hits=num_hits, max_size_mask=max_size_mask, 
            length=length, global_min_threshold=global_min_threshold,
            disjoint_info=disjoint_info, debug=debug
        )


    # The new frontier computed an higher minimum threshold
    if local_minimum_threshold > global_min_threshold:
        # Update the global minimum threshold
        global_min_threshold = local_minimum_threshold
        # Reduce the past frontier based on the new global minimum threshold
        if past_frontier is not None:
            outer_reduced = True
            past_frontier = reduce_frontier(
                past_frontier, global_min_threshold)
            heapq.heapify(past_frontier)
    first_reduced_frontier_time = time.time()


    # Merge the new nodes with the past frontier
    merged = False
    if past_frontier is not None and len(past_frontier) > 0:
        merged = True
        # Merge assumes that the input iterables are sorted
        for new_node in new_frontier:
            heapq.heappush(past_frontier, new_node)
        sorted_frontier = past_frontier
    else:
        # Initialize frontier as a queue
        heapq.heapify(new_frontier)
        sorted_frontier = new_frontier
    sorted_frontier_time = time.time()
    if debug:
        debug_info["update_reduced_frontier"] = outer_reduced
        debug_info["update_reduce_frontier_time"] = first_reduced_frontier_time - estimate_iou_frontier_time
        debug_info["merged"] = merged
        debug_info["merge_time"] = sorted_frontier_time - first_reduced_frontier_time
        return sorted_frontier, global_min_threshold, debug_info
    else:
        return sorted_frontier, global_min_threshold

def get_ancestor_masks(f, masks, path_masks=None):
    """
    Function to return a mask for a given formula.
    Args:
        f (src.formula.Formula): formula.
        masks (list): list of masks.
        optional_masks (dict): dictionary of additional masks (beam masks).
    Returns:
        Formula's Mask.
    """
    if path_masks is not None and f in path_masks.keys():
        return path_masks
    elif path_masks is None:
        path_masks = {}
    if isinstance(f, F.Leaf):
        mask = masks[f.val]
        path_masks[f] = utils.sparse_to_torch(mask)
        return path_masks
    elif isinstance(f, F.Or):
        l_ancestors_masks = get_ancestor_masks(f.left, masks, path_masks)
        r_ancestors_masks = get_ancestor_masks(f.right, masks, path_masks)
        mask = l_ancestors_masks[f.left] | r_ancestors_masks[f.right]
        path_masks[f] = mask
        path_masks.update(l_ancestors_masks)
        path_masks.update(r_ancestors_masks)
        return path_masks
    elif isinstance(f, F.And):
        l_ancestors_masks = get_ancestor_masks(f.left, masks, path_masks)
        r_ancestors_masks = get_ancestor_masks(f.right, masks, path_masks)
        mask = l_ancestors_masks[f.left] & r_ancestors_masks[f.right]
        path_masks[f] = mask
        path_masks.update(l_ancestors_masks)
        path_masks.update(r_ancestors_masks)
        return path_masks
    elif isinstance(f, F.Not):
        l_ancestors_masks = get_ancestor_masks(f.val, masks, path_masks)
        not_mask = ~l_ancestors_masks[f.val]
        path_masks[f] = not_mask
        return path_masks
    elif isinstance(f, int):
        mask = masks[f]
        path_masks[F.Leaf(f)] = utils.sparse_to_torch(mask)
        return path_masks
    else:
        raise ValueError(f"Unknown formula type {type(f)}")


# We don't update the heuristic info because exact IoU is computed rarely and only at the end.
# If wew do this, we should also check for label_mapping inside get_esti
def compute_exact_iou_and_update_heuristic_info(
    label, label_mapping, label_mask, bitmaps, heuristic_info):

    seg_quantities, neuron_quantities, concepts_quantities = heuristic_info
    neuron_unique, neuron_common, neuron_coverable_mask, neuron_coverable, neuron_sum  = neuron_quantities
    common_intersection_tuple, unique_intersection_tuple, common_extras_tuple, unique_extras_tuple, uncovered_tuple = concepts_quantities
    common_intersection = common_intersection_tuple[0]
    common_intersection_sum = common_intersection_tuple[1]
    unique_intersection = unique_intersection_tuple[0]
    unique_intersection_sum = unique_intersection_tuple[1]
    common_extras = common_extras_tuple[0]
    common_extras_sum = common_extras_tuple[1]
    unique_extras = unique_extras_tuple[0]
    unique_extras_sum = unique_extras_tuple[1]
    uncovered = uncovered_tuple[0]
    uncovered_sum = uncovered_tuple[1]

    common_elements, unique_elements, uncoverable_elements = seg_quantities

    # Update Label
    unique_elements = unique_elements.to(bitmaps.device)
    common_elements = common_elements.to(bitmaps.device)
    uncoverable_elements = uncoverable_elements.to(bitmaps.device)
    concept_quantities = optimal_heuristic.compute_quantities_vector(label_mask, bitmaps, common_elements, unique_elements, neuron_coverable_mask)
    c_common_intersection, c_unique_intersection, c_common_extras, c_unique_extras, c_uncovered = concept_quantities
    label_iou = (c_common_intersection.sum() + c_unique_intersection.sum()) / (bitmaps.sum() + c_common_extras.sum() + c_unique_extras.sum())
    if label not in label_mapping:
        # Assign first free available index to the node
        index_node = len(common_intersection)
        if index_node in label_mapping.values():
            raise ValueError(f"Index {index_node} already exists in label mapping.")
        label_mapping[label] = index_node
        # Update heuristic info
        common_intersection.append(c_common_intersection)
        common_intersection_sum.append(c_common_intersection.sum())
        unique_intersection.append(c_unique_intersection)
        unique_intersection_sum.append(c_unique_intersection.sum())
        common_extras.append(c_common_extras)
        common_extras_sum.append(c_common_extras.sum())
        unique_extras.append(c_unique_extras)
        unique_extras_sum.append(c_unique_extras.sum())
        uncovered.append(c_uncovered)
        uncovered_sum.append(c_uncovered.sum())

    else:
        index_node = label_mapping[label]
        common_intersection[index_node] = c_common_intersection
        common_intersection_sum[index_node] = c_common_intersection.sum()
        unique_intersection[index_node] = c_unique_intersection
        unique_intersection_sum[index_node] = c_unique_intersection.sum()
        common_extras[index_node] = c_common_extras
        common_extras_sum[index_node] = c_common_extras.sum()
        unique_extras[index_node] = c_unique_extras
        unique_extras_sum[index_node] = c_unique_extras.sum()
        uncovered[index_node] = c_uncovered
        uncovered_sum[index_node] = c_uncovered.sum()
    
    concepts_quantities = (
        (common_intersection, common_intersection_sum), (unique_intersection, unique_intersection_sum), (common_extras, common_extras_sum), (unique_extras, unique_extras_sum), (uncovered, uncovered_sum))
    heuristic_info = (seg_quantities, neuron_quantities, concepts_quantities)
    return label_iou.item(), label_mapping, heuristic_info

def compute_exact_iou(label_mask, bitmaps, heuristic_info):
    seg_quantities, neuron_quantities, concepts_quantities = heuristic_info
    neuron_unique, neuron_common, neuron_coverable_mask, neuron_coverable, neuron_sum  = neuron_quantities
    common_elements, unique_elements, uncoverable_elements = seg_quantities
    unique_elements = unique_elements.to(bitmaps.device)
    common_elements = common_elements.to(bitmaps.device)
    uncoverable_elements = uncoverable_elements.to(bitmaps.device)
    concept_quantities = optimal_heuristic.compute_quantities_vector(label_mask, bitmaps, common_elements, unique_elements, neuron_coverable_mask)
    c_common_intersection, c_unique_intersection, c_common_extras, c_unique_extras, c_uncovered = concept_quantities
    label_iou = (c_common_intersection.sum() + c_unique_intersection.sum()) / (bitmaps.sum() + c_common_extras.sum() + c_unique_extras.sum())
    return label_iou.item()

def explore_disjoint_frontier(
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
      
    
    # Search parameters
    done = False

    # # Debugging information
    debug = False
    total_added_frontier = 0
    list_estimate_reduce_frontier_time = []
    list_update_reduced_frontier_time = []
    list_merge_time = []
    list_time_estimate_iou_frontier = []
    list_expanded_time = []
    list_avg_frontier_node_time = []
    list_median_frontier_node_time = []
    list_parsing_frontier_time = []
    popped_nodes = 0
    expanded_nodes = 0
    full_iou_nodes = 0
    nodes_updated = 0
    non_final_nodes = 0
    final_nodes = 0
    skipped_because_full_iou = 0
    skipped_because_recent = 0

    # Node times 
    list_esti_quantities_time = []
    list_individual_estimation_time = []
    list_or_chain_estimation_time = []
    list_and_chain_estimation_time = []
    list_and_not_chain_estimation_time = []
    list_and_or_chain_estimation_time = []
    list_comb_or_andnot_chain_estimation_time = []


    # # Initialize the frontier with all the candidate labels

    done = len(initial_frontier) == 0
    i = 0
    best_results = (0.0, None) # (IoU, label)
    visited = []
    recent_nodes = []
    recent_e_iou = 2

    # Initialize frontiers
    overlapping_frontier = []
    current_frontier = initial_frontier

    while not done:
        #node = heapq.heapreplace(current_frontier, node_1)
        node = heapq.heappop(current_frontier)
        popped_nodes += 1
        e_node = node[0]
        label_node = node[2]
        next_op_node = node[1]
        
        if debug:
            if len(label_node) < length:
                non_final_nodes += 1
            else:
                final_nodes += 1

        # If node visited, skip it, this should not happen
        if debug and next_op_node=='INDIVIDUAL' and label_node in visited:
            done = len(current_frontier) == 0
            skipped_because_full_iou += 1
            #print(f"Skipping node {node} because it was already visited.")
            #raise ValueError(f"Node {node} already visited. This should not happen.")
            continue


        # if debug and -e_node > recent_e_iou:
        #     done = len(current_frontier) == 0
        #     print(f"Skipping node {node} because its estimated IoU is too low: {-e_node} > {recent_e_iou}.")
        #     exit()
        #     continue           
        
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
        
        if debug:
            if i % 500 == 0:
                # print(f"Iteration {i} Time: {(time.time() - init_time)/60:.2f} Stats: \t Nodes: Added to frontier {total_added_frontier} \t popped {popped_nodes} \t expanded: {expanded_nodes} \t Full IoU:{full_iou_nodes}  \t Updated: {nodes_updated} \t Skipped bc full IoU: {skipped_because_full_iou} \t Skipped bc recent: {skipped_because_recent}")
                # #print(f"Times Avg: estimate_reduce {np.mean(list_estimate_reduce_frontier_time):.4f} \t update_reduced {np.mean(list_update_reduced_frontier_time):.4f} \t merge {np.mean(list_merge_time):.4f} \t estimate_iou {np.mean(list_time_estimate_iou_frontier):.4f} \t expanded {np.mean(list_expanded_time):.4f} \t avg_frontier_node {np.mean(list_avg_frontier_node_time):.4f} \t median_frontier_node {np.median(list_median_frontier_node_time):.4f} \t parsing_frontier {np.mean(list_parsing_frontier_time):.4f}")
                # #print(f"Times Tot: estimate_reduce {sum(list_estimate_reduce_frontier_time):.4f} \t update_reduced {sum(list_update_reduced_frontier_time):.4f} \t merge {sum(list_merge_time):.4f} \t estimate_iou {sum(list_time_estimate_iou_frontier):.4f} \t expanded {sum(list_expanded_time):.4f} \t avg_frontier_node {sum(list_avg_frontier_node_time):.4f} \t median_frontier_node {sum(list_median_frontier_node_time):.4f} \t parsing_frontier {sum(list_parsing_frontier_time):.4f}")
                # print(f"Times Avg:  expanded {np.mean(list_expanded_time):.4f} \t avg_frontier_node {np.mean(list_avg_frontier_node_time):.4f} \t median_frontier_node {np.median(list_median_frontier_node_time):.4f} \t parsing_frontier {np.mean(list_parsing_frontier_time):.4f}")
                # print(f"Times Tot:  expanded {sum(list_expanded_time):.4f} \t avg_frontier_node {sum(list_avg_frontier_node_time):.4f} \t median_frontier_node {sum(list_median_frontier_node_time):.4f} \t parsing_frontier {sum(list_parsing_frontier_time):.4f}")
                # print(f"Times Node Avg: esti_quantities {np.mean(list_esti_quantities_time):.4f}/{len(list_esti_quantities_time)} \t individual_estimation {np.mean(list_individual_estimation_time):.4f}/{len(list_individual_estimation_time)} \t or_chain {np.mean(list_or_chain_estimation_time):.4f}/{len(list_or_chain_estimation_time)} \t and_chain {np.mean(list_and_chain_estimation_time):.4f}/{len(list_and_chain_estimation_time)} \t and_not_chain {np.mean(list_and_not_chain_estimation_time):.4f}/{len(list_and_not_chain_estimation_time)} \t and_or_chain {np.mean(list_and_or_chain_estimation_time):.4f}/{len(list_and_or_chain_estimation_time)} \t comb_or_andnot_chain {np.mean(list_comb_or_andnot_chain_estimation_time):.4f}/{len(list_comb_or_andnot_chain_estimation_time)}")
                # print(f"Times Node Tot: esti_quantities {sum(list_esti_quantities_time):.4f} \t individual_estimation {sum(list_individual_estimation_time):.4f} \t or_chain {sum(list_or_chain_estimation_time):.4f} \t and_chain {sum(list_and_chain_estimation_time):.4f} \t and_not_chain {sum(list_and_not_chain_estimation_time):.4f} \t and_or_chain {sum(list_and_or_chain_estimation_time):.4f} \t comb_or_andnot_chain {sum(list_comb_or_andnot_chain_estimation_time):.4f}")
                # print("--------------------------------------------------------------------------------------------")
                print(f"Iteration {i} \t Nodes: Added to frontier {total_added_frontier} \t popped {popped_nodes} \t expanded: {expanded_nodes} \t non-final nodes: {non_final_nodes} \t final nodes: {final_nodes}")
                print("--------------------------------------------------------------------------------------------") 
        else:
            print(f"Iteration {i} \t Disjoint frontier size: {len(current_frontier)} \t Overlap frontier size: {len(overlapping_frontier)} \t Node Esti: {round(e_node,4)} Threshold: {round(minimum_threshold, 4)} Node label: {label_node}", end='\r') 

        if next_op_node == 'INDIVIDUAL':
            iou, current_frontier, heuristic_info, minimum_threshold, label_mapping = analyze_final_node(
                label_node, masks, bitmaps, heuristic_info, minimum_threshold,
                       current_frontier, heuristic_name, label_mapping, max_improvement,
                       disjoint_info, num_hits, max_size_mask, length)

            # Add the node to the visited nodes
            visited.append(label_node)
            full_iou_nodes += 1

            if iou > best_results[0]:
                best_results = (iou, label_node)
            elif iou == best_results[0]:
                if len(label_node) < len(best_results[1]):
                    # We prefer shorter labels in case of equal IoU
                    best_results = (iou, label_node)

            done = len(current_frontier) == 0
            continue
        
        if debug and len(label_node) >= length:
            # compute real IoU
            done = len(current_frontier) == 0
            raise ValueError(
                f"Node {node} has a label of length {len(label_node)} which is greater than the maximum length {length}. This should not happen."
            )


        # Compute the updated frontier based on the new nodes
        if debug:
            # Expand the node to get the next frontier
            before_expand_time = time.time()
            next_frontier, next_overlap_frontier = expand_node(node, candidate_labels=candidate_concepts, max_length=length, disjoint_info=disjoint_info)
            
            expand_time = time.time()

            # Expand the overlapping frontier
            overlapping_frontier.extend(next_overlap_frontier)

            expanded_nodes += len(next_frontier)
            list_expanded_time.append(expand_time - before_expand_time)
            current_frontier, minimum_threshold, debug_info, heuristic_info, label_mapping = update_frontier(
                past_frontier=current_frontier, new_nodes=next_frontier, 
                label_mapping=label_mapping, heuristic=heuristic_name, 
                heuristic_info=heuristic_info, max_improvement=max_improvement,
                num_hits=num_hits, max_size_mask=max_size_mask,
                length=length, global_min_threshold=minimum_threshold, disjoint_info=disjoint_info,
                debug=True
            )

            # Update debug info
            total_added_frontier += debug_info['added_to_frontier']
            if debug_info['estimate_reduced_frontier']:
                list_estimate_reduce_frontier_time.append(debug_info['estimate_reduce_frontier_time'])
            if debug_info['update_reduced_frontier']:
                list_update_reduced_frontier_time.append(debug_info['update_reduce_frontier_time'])
            if debug_info['merged']:
                list_merge_time.append(debug_info['merge_time'])
            list_time_estimate_iou_frontier.append(debug_info['time_estimate_iou_frontier'])
            list_avg_frontier_node_time.append(debug_info['avg_time_frontier_node'])
            list_median_frontier_node_time.append(debug_info['median_time_frontier_node'])
            list_parsing_frontier_time.append(debug_info['parsing_frontier_time'])
            list_esti_quantities_time.extend(debug_info['esti_quantities_time'])
            list_individual_estimation_time.extend(debug_info['individual_estimation_time'])
            list_or_chain_estimation_time.extend(debug_info['or_chain_estimation_time'])
            list_and_chain_estimation_time.extend(debug_info['and_chain_estimation_time'])
            list_and_not_chain_estimation_time.extend(debug_info['and_not_chain_estimation_time'])
            list_and_or_chain_estimation_time.extend(debug_info['and_or_chain_estimation_time'])
            list_comb_or_andnot_chain_estimation_time.extend(debug_info['comb_or_andnot_chain_estimation_time'])
        else:
            # Expand the node to get the next frontier
            next_frontier, next_overlap_frontier = expand_node(node, candidate_labels=candidate_concepts, max_length=length, disjoint_info=disjoint_info)
            
            # Expand the overlapping frontier
            overlapping_frontier.extend(next_overlap_frontier)


            # Compute the estimation for the next frontier
            current_frontier, minimum_threshold, heuristic_info, label_mapping = update_frontier(
                past_frontier=current_frontier, new_nodes=next_frontier, 
                label_mapping=label_mapping, heuristic=heuristic_name, 
                heuristic_info=heuristic_info, max_improvement=max_improvement,
                num_hits=num_hits, max_size_mask=max_size_mask,
                length=length, global_min_threshold=minimum_threshold, disjoint_info=disjoint_info
            )

        i += 1
        done = len(current_frontier) == 0
    best_iou = best_results[0]
    best_label = best_results[1]
    
    return best_label, best_iou, visited, popped_nodes, overlapping_frontier, minimum_threshold, label_mapping

def explore_overlapping_frontier(
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
    visited,
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
      
    
    # Search parameters

    # # Debugging information
    debug = False
    total_added_frontier = 0
    list_estimate_reduce_frontier_time = []
    list_update_reduced_frontier_time = []
    list_merge_time = []
    list_time_estimate_iou_frontier = []
    list_expanded_time = []
    list_avg_frontier_node_time = []
    list_median_frontier_node_time = []
    list_parsing_frontier_time = []
    popped_nodes = 0
    expanded_nodes = 0
    full_iou_nodes = 0
    non_final_nodes = 0
    final_nodes = 0
    nodes_updated = 0
    skipped_because_full_iou = 0
    skipped_because_recent = 0

    # Node times 
    list_esti_quantities_time = []
    list_individual_estimation_time = []
    list_or_chain_estimation_time = []
    list_and_chain_estimation_time = []
    list_and_not_chain_estimation_time = []
    list_and_or_chain_estimation_time = []
    list_comb_or_andnot_chain_estimation_time = []


    # Initialize the frontier with all the candidate labels

    done = len(initial_frontier) == 0
    i = 0
    best_results = (0.0, None) # (IoU, label)
    recent_nodes = []
    recent_e_iou = 2

    # Initialize frontiers
    current_frontier, minimum_threshold, heuristic_info, label_mapping = update_frontier(
        past_frontier=None, new_nodes=initial_frontier, 
        label_mapping=label_mapping, heuristic=heuristic_name, 
        heuristic_info=heuristic_info, max_improvement=max_improvement,
        num_hits=num_hits, max_size_mask=max_size_mask,
        length=length, global_min_threshold=minimum_threshold, disjoint_info=disjoint_info
    )
    while not done:
        #node = heapq.heapreplace(current_frontier, node_1)
        node = heapq.heappop(current_frontier)
        popped_nodes += 1
        e_node = node[0]
        label_node = node[2]
        next_op_node = node[1]
        
        if debug:
            if len(label_node) < length:
                non_final_nodes += 1
            else:
                final_nodes += 1
        # If node visited, skip it, this should not happen
        if debug and next_op_node=='INDIVIDUAL' and label_node in visited:
            done = len(current_frontier) == 0
            skipped_because_full_iou += 1
            continue


        # if debug and -e_node > recent_e_iou:
        #     done = len(current_frontier) == 0
        #     print(f"Skipping node {node} because its estimated IoU is too low: {-e_node} > {recent_e_iou}.")
        #     print("Disjoint")
        #     exit()
        #     continue           
        
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
        
        if debug:
            if i % 500 == 0:
                # print(f"Iteration {i} Time: {(time.time() - init_time)/60:.2f} Stats: \t Nodes: Added to frontier {total_added_frontier} \t popped {popped_nodes} \t expanded: {expanded_nodes} \t Full IoU:{full_iou_nodes}  \t Updated: {nodes_updated} \t Skipped bc full IoU: {skipped_because_full_iou} \t Skipped bc recent: {skipped_because_recent}")
                # #print(f"Times Avg: estimate_reduce {np.mean(list_estimate_reduce_frontier_time):.4f} \t update_reduced {np.mean(list_update_reduced_frontier_time):.4f} \t merge {np.mean(list_merge_time):.4f} \t estimate_iou {np.mean(list_time_estimate_iou_frontier):.4f} \t expanded {np.mean(list_expanded_time):.4f} \t avg_frontier_node {np.mean(list_avg_frontier_node_time):.4f} \t median_frontier_node {np.median(list_median_frontier_node_time):.4f} \t parsing_frontier {np.mean(list_parsing_frontier_time):.4f}")
                # #print(f"Times Tot: estimate_reduce {sum(list_estimate_reduce_frontier_time):.4f} \t update_reduced {sum(list_update_reduced_frontier_time):.4f} \t merge {sum(list_merge_time):.4f} \t estimate_iou {sum(list_time_estimate_iou_frontier):.4f} \t expanded {sum(list_expanded_time):.4f} \t avg_frontier_node {sum(list_avg_frontier_node_time):.4f} \t median_frontier_node {sum(list_median_frontier_node_time):.4f} \t parsing_frontier {sum(list_parsing_frontier_time):.4f}")
                # print(f"Times Avg:  expanded {np.mean(list_expanded_time):.4f} \t avg_frontier_node {np.mean(list_avg_frontier_node_time):.4f} \t median_frontier_node {np.median(list_median_frontier_node_time):.4f} \t parsing_frontier {np.mean(list_parsing_frontier_time):.4f}")
                # print(f"Times Tot:  expanded {sum(list_expanded_time):.4f} \t avg_frontier_node {sum(list_avg_frontier_node_time):.4f} \t median_frontier_node {sum(list_median_frontier_node_time):.4f} \t parsing_frontier {sum(list_parsing_frontier_time):.4f}")
                # print(f"Times Node Avg: esti_quantities {np.mean(list_esti_quantities_time):.4f}/{len(list_esti_quantities_time)} \t individual_estimation {np.mean(list_individual_estimation_time):.4f}/{len(list_individual_estimation_time)} \t or_chain {np.mean(list_or_chain_estimation_time):.4f}/{len(list_or_chain_estimation_time)} \t and_chain {np.mean(list_and_chain_estimation_time):.4f}/{len(list_and_chain_estimation_time)} \t and_not_chain {np.mean(list_and_not_chain_estimation_time):.4f}/{len(list_and_not_chain_estimation_time)} \t and_or_chain {np.mean(list_and_or_chain_estimation_time):.4f}/{len(list_and_or_chain_estimation_time)} \t comb_or_andnot_chain {np.mean(list_comb_or_andnot_chain_estimation_time):.4f}/{len(list_comb_or_andnot_chain_estimation_time)}")
                # print(f"Times Node Tot: esti_quantities {sum(list_esti_quantities_time):.4f} \t individual_estimation {sum(list_individual_estimation_time):.4f} \t or_chain {sum(list_or_chain_estimation_time):.4f} \t and_chain {sum(list_and_chain_estimation_time):.4f} \t and_not_chain {sum(list_and_not_chain_estimation_time):.4f} \t and_or_chain {sum(list_and_or_chain_estimation_time):.4f} \t comb_or_andnot_chain {sum(list_comb_or_andnot_chain_estimation_time):.4f}")
                # print("--------------------------------------------------------------------------------------------")
                print(f"Iteration {i} \t Nodes: Added to frontier {total_added_frontier} \t popped {popped_nodes} \t expanded: {expanded_nodes} \t non-final nodes: {non_final_nodes} \t final nodes: {final_nodes}")
                print("--------------------------------------------------------------------------------------------")           
        else:
            print(f"Iteration {i} \t Overlap frontier size: {len(current_frontier)} \t Node Esti: {round(e_node,4)} Threshold: {round(minimum_threshold, 4)} Node label: {label_node}", end='\r') 

        if next_op_node == 'INDIVIDUAL':
            #if len(label_node) >= length and isinstance(label_node, F.And) and isinstance(label_node.right, F.Not):
                #print(f"Final node {label_node} with estimated IoU {-e_node:.4f} and threshold {minimum_threshold:.4f}")
            new_max, new_min = optimal_heuristic.update_optimal_label_iou(heuristic_name='sample', node=node, label_mapping=label_mapping, heuristic_info=heuristic_info, max_improvement=max_improvement, disjoint_info=disjoint_info, num_hits=num_hits, max_size_mask=max_size_mask, max_length=length)
            if new_max < -e_node:
                #print(f"Node {node} has a new max IoU {new_max:.4f} which is lower than the previous estimated IoU {-e_node:.4f}")
                if new_max > minimum_threshold:
                    heapq.heappush(current_frontier, (-new_max, next_op_node, label_node, node[3]))
                done = len(current_frontier) == 0
                continue
                # else:
                #     print(f"Node {node} has a new max IoU {new_max:.4f} which is greater than the previous estimated IoU {-e_node:.4f}")
           # print(node, new_max, minimum_threshold)
                
            iou, current_frontier, heuristic_info, minimum_threshold, label_mapping = analyze_final_node(
                label_node, masks, bitmaps, heuristic_info, minimum_threshold,
                       current_frontier, heuristic_name, label_mapping, max_improvement,
                       disjoint_info, num_hits, max_size_mask, length)

            # Add the node to the visited nodes
            visited.append(label_node)
            full_iou_nodes += 1


            if iou > best_results[0]:
                best_results = (iou, label_node)
            elif iou == best_results[0]:
                if len(label_node) < len(best_results[1]):
                    # We prefer shorter labels in case of equal IoU
                    best_results = (iou, label_node)

            done = len(current_frontier) == 0
            continue
        
        if debug and len(label_node) >= length:
            # compute real IoU
            done = len(current_frontier) == 0
            raise ValueError(
                f"Node {node} has a label of length {len(label_node)} which is greater than the maximum length {length}. This should not happen."
            )


        # Compute the updated frontier based on the new nodes
        if debug:
            # Expand the node to get the next frontier
            before_expand_time = time.time()
            void_frontier, next_frontier = expand_node(node, candidate_labels=candidate_concepts, max_length=length, disjoint_info=disjoint_info)
            expand_time = time.time()

            expanded_nodes += len(next_frontier)
            list_expanded_time.append(expand_time - before_expand_time)
            current_frontier, minimum_threshold, debug_info, heuristic_info, label_mapping = update_frontier(
                past_frontier=current_frontier, new_nodes=next_frontier, 
                label_mapping=label_mapping, heuristic=heuristic_name, 
                heuristic_info=heuristic_info, max_improvement=max_improvement,
                num_hits=num_hits, max_size_mask=max_size_mask,
                length=length, global_min_threshold=minimum_threshold, disjoint_info=disjoint_info,
                debug=True
            )

            # Update debug info
            total_added_frontier += debug_info['added_to_frontier']
            if debug_info['estimate_reduced_frontier']:
                list_estimate_reduce_frontier_time.append(debug_info['estimate_reduce_frontier_time'])
            if debug_info['update_reduced_frontier']:
                list_update_reduced_frontier_time.append(debug_info['update_reduce_frontier_time'])
            if debug_info['merged']:
                list_merge_time.append(debug_info['merge_time'])
            list_time_estimate_iou_frontier.append(debug_info['time_estimate_iou_frontier'])
            list_avg_frontier_node_time.append(debug_info['avg_time_frontier_node'])
            list_median_frontier_node_time.append(debug_info['median_time_frontier_node'])
            list_parsing_frontier_time.append(debug_info['parsing_frontier_time'])
            list_esti_quantities_time.extend(debug_info['esti_quantities_time'])
            list_individual_estimation_time.extend(debug_info['individual_estimation_time'])
            list_or_chain_estimation_time.extend(debug_info['or_chain_estimation_time'])
            list_and_chain_estimation_time.extend(debug_info['and_chain_estimation_time'])
            list_and_not_chain_estimation_time.extend(debug_info['and_not_chain_estimation_time'])
            list_and_or_chain_estimation_time.extend(debug_info['and_or_chain_estimation_time'])
            list_comb_or_andnot_chain_estimation_time.extend(debug_info['comb_or_andnot_chain_estimation_time'])
        else:
            # Expand the node to get the next frontier
            void_frontier, next_frontier = expand_node(node, candidate_labels=candidate_concepts, max_length=length, disjoint_info=disjoint_info)
            assert len(void_frontier) == 0, "The void frontier should be empty in the overlap frontier exploration."

        # Compute the estimation for the next frontier
        current_frontier, minimum_threshold, heuristic_info, label_mapping = update_frontier(
            past_frontier=current_frontier, new_nodes=next_frontier, 
            label_mapping=label_mapping, heuristic=heuristic_name, 
            heuristic_info=heuristic_info, max_improvement=max_improvement,
            num_hits=num_hits, max_size_mask=max_size_mask,
            length=length, global_min_threshold=minimum_threshold, disjoint_info=disjoint_info
        )

        i += 1
        done = len(current_frontier) == 0
    best_iou = best_results[0]
    best_label = best_results[1]
    
    return best_label, best_iou, len(visited), popped_nodes


def perform_exhaustive_heuristic_searchUNIFIED(
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
      
    # Candidate concepts
    candidate_labels = [F.Leaf(c) for c in range(len(masks))]
    label_mapping = {F.Leaf(c): c for c in range(len(masks))}
    
    # Max improvement for this neuron
    max_improvement = compute_max_improvement(heuristic_info, label_mapping, max_size_mask, num_hits, length)

    # Search parameters
    done = False
    heuristic_name ='sum'
    #heuristic_name ='sample'
    #heuristic_name ='hybrid'
    # Debugging information
    num_hits = num_hits.item()
    debug = False
    total_added_frontier = 0
    list_estimate_reduce_frontier_time = []
    list_update_reduced_frontier_time = []
    list_merge_time = []
    list_time_estimate_iou_frontier = []
    list_expanded_time = []
    list_avg_frontier_node_time = []
    list_median_frontier_node_time = []
    list_parsing_frontier_time = []
    popped_nodes = 0
    expanded_nodes = 0
    full_iou_nodes = 0
    nodes_updated = 0
    skipped_because_full_iou = 0
    skipped_because_recent = 0

    # Node times 
    list_esti_quantities_time = []
    list_individual_estimation_time = []
    list_or_chain_estimation_time = []
    list_and_chain_estimation_time = []
    list_and_not_chain_estimation_time = []
    list_and_or_chain_estimation_time = []
    list_comb_or_andnot_chain_estimation_time = []


    # label_a = F.And(F.Leaf(12), F.Not(F.Leaf(634)))
    # next_op = 'NOT'
    # last_val = label_a.get_vals()[0]
    # last_op = label_a.get_ops()[0]
    # print(last_op, last_val)
    # yes=[]
    # no =[]
    # for candidate_term in candidate_labels:
    #     # Skip the candidate term if it is already in the label
    #     if candidate_term.val in label_a.get_vals():
    #         continue
    #     # Impose order to avoid logical equivalence
    #     if next_op != 'NOT' and isinstance(label_a, F.Leaf) and candidate_term.val < label_a.val:
    #         continue
    #     elif len(label_a) > 1:
    #         if next_op == 'NOT' and last_op == 'AND' and isinstance(label_a.right, F.Not):
    #             if candidate_term.val < last_val:
    #                 continue
    #         if last_op == next_op:
    #             if candidate_term.val < last_val:
    #                 continue
    #     yes.append(candidate_term)
    # print(yes)
    # exit()
    # print(compute_ratio(
    #         utils.sparse_to_torch(mask_utils.get_formula_mask(F.Leaf(139), masks)).to(bitmaps.device),
    #         bitmaps, heuristic_info, debug=True
    #     ))
    # print(compute_ratio(
    #         utils.sparse_to_torch(mask_utils.get_formula_mask(F.Leaf(254), masks)).to(bitmaps.device),
    #         bitmaps, heuristic_info, debug=True
    #     ))
    # common_stem = F.Or(F.Leaf(139), F.Leaf(254))
    # ratio = compute_ratio(
    #         utils.sparse_to_torch(mask_utils.get_formula_mask(common_stem, masks)).to(bitmaps.device),
    #         bitmaps, heuristic_info, debug=True
    #     )
    # print(448)
    # print(compute_ratio(
    #         utils.sparse_to_torch(mask_utils.get_formula_mask(F.Leaf(448), masks)).to(bitmaps.device),
    #         bitmaps, heuristic_info, debug=True
    #     ))
    # formula_a = F.Or(common_stem, F.Leaf(448))
    # print(compute_ratio(
    #         utils.sparse_to_torch(mask_utils.get_formula_mask(formula_a, masks)).to(bitmaps.device),
    #         bitmaps, heuristic_info, debug=True
    #     ))
    # print(511)
    # print(compute_ratio(
    #         utils.sparse_to_torch(mask_utils.get_formula_mask(F.Leaf(511), masks)).to(bitmaps.device),
    #         bitmaps, heuristic_info, debug=True
    #     ))
    # formula_b = F.Or(common_stem, F.Leaf(511))
    # print(compute_ratio(
    #         utils.sparse_to_torch(mask_utils.get_formula_mask(formula_b, masks)).to(bitmaps.device),
    #         bitmaps, heuristic_info, debug=True
    #     ))
    # exit()
    # Initialize the frontier with all the candidate labels
    frontier = [(0.0, None, k, None) for k in label_mapping.keys()]
    if debug:
        current_frontier, minimum_info, debug_info, heuristic_info, label_mapping = update_frontier(
            bitmaps=bitmaps, masks=masks,
            past_frontier=None, new_nodes=frontier, label_mapping=label_mapping,
            heuristic=heuristic_name, heuristic_info=heuristic_info, 
            max_improvement=max_improvement,
            num_hits=num_hits, max_size_mask=max_size_mask,
            length=length, global_min_threshold=0.0, disjoint_info=disjoint_info, debug=debug
        )
        # Update debug info
        total_added_frontier += debug_info['added_to_frontier']
        if debug_info['estimate_reduced_frontier']:
            list_estimate_reduce_frontier_time.append(debug_info['estimate_reduce_frontier_time'])
        if debug_info['update_reduced_frontier']:
            list_update_reduced_frontier_time.append(debug_info['update_reduce_frontier_time'])
        if debug_info['merged']:
            list_merge_time.append(debug_info['merge_time'])
        list_time_estimate_iou_frontier.append(debug_info['time_estimate_iou_frontier'])
        list_avg_frontier_node_time.append(debug_info['avg_time_frontier_node'])
        list_median_frontier_node_time.append(debug_info['median_time_frontier_node'])
        list_parsing_frontier_time.append(debug_info['parsing_frontier_time'])
        list_esti_quantities_time.extend(debug_info['esti_quantities_time'])
        list_individual_estimation_time.extend(debug_info['individual_estimation_time'])
        list_or_chain_estimation_time.extend(debug_info['or_chain_estimation_time'])
        list_and_chain_estimation_time.extend(debug_info['and_chain_estimation_time'])
        list_and_not_chain_estimation_time.extend(debug_info['and_not_chain_estimation_time'])
        list_and_or_chain_estimation_time.extend(debug_info['and_or_chain_estimation_time'])
        list_comb_or_andnot_chain_estimation_time.extend(debug_info['comb_or_andnot_chain_estimation_time'])
    else:
        current_frontier, minimum_info, heuristic_info, label_mapping = update_frontier(
            bitmaps=bitmaps, masks=masks,
            past_frontier=None, new_nodes=frontier, label_mapping=label_mapping,
            heuristic=heuristic_name, heuristic_info=heuristic_info, 
            max_improvement=max_improvement,
            num_hits=num_hits, max_size_mask=max_size_mask,
            length=length, global_min_threshold=0.0, disjoint_info=disjoint_info, debug=debug
        )
    minimum_threshold, minimum_node = minimum_info

    done = len(current_frontier) == 0
    i = 0
    best_results = (0.0, None) # (IoU, label)
    visited = []
    recent_nodes = []
    recent_e_iou = 2
    # init_time = time.time()
    # concept_a = 12
    # concept_b_1 = 83
    # concept_b_2 = 315
    # concept_c_1 = 1
    # concept_c_2 = 2
    # formula_1 = F.And(F.Or(F.Leaf(concept_a), F.Leaf(concept_b_1)), F.Not(F.Leaf(concept_c_1)))
    # formula_2 = F.And(F.Or(F.Leaf(concept_a), F.Leaf(concept_b_1)), F.Not(F.Leaf(concept_c_2)))
    # formula_3 = F.And(F.Or(F.Leaf(concept_a), F.Leaf(concept_b_2)), F.Not(F.Leaf(concept_c_1)))
    # node_1 = (-2.0, 'INDIVIDUAL', formula_1, [[], [], [], []])
    # node_2 = (-2.0, 'INDIVIDUAL', formula_2, [[], [], [], []])
    # node_3 = (-2.0, 'INDIVIDUAL', formula_3, [[], [], [], []])
    # current_frontier.append(node_1)
    # current_frontier.append(node_2)
    # current_frontier.append(node_3)
    # heapq.heapify(current_frontier)

    # Initialize overlapping frontier
    overlapping_frontier = []
    while not done:
        #node = heapq.heapreplace(current_frontier, node_1)
        node = heapq.heappop(current_frontier)
        popped_nodes += 1
        e_node = node[0]
        label_node = node[2]
        next_op_node = node[1]
        # from . import sum_heuristic
        # #((83 AND (NOT 5)) OR 97)
        # concept_a = 83
        # concept_b = 3
        # concept_c = 97
        # formula_1 = F.Or(F.And(F.Leaf(concept_a), F.Not(F.Leaf(1))), F.Leaf(concept_c))
        # formula_2 = F.Or(F.And(F.Leaf(concept_a), F.Not(F.Leaf(1))), F.Leaf(657))
        # formula_3 = F.Or(F.And(F.Leaf(concept_b), F.Not(F.Leaf(3))), F.Leaf(concept_c))
        # formula_4 = F.Or(F.And(F.Leaf(concept_b), F.Not(F.Leaf(4))), F.Leaf(concept_c))
        # formulas = [formula_1, formula_2, formula_3, formula_4]
        # frontier = [(1.0, 'INDIVIDUAL', formula,  [[], [], [], []]) for formula in formulas]
        # #a = optimal_heuristic.get_esti_quantities(sum_heuristic, dummy_node, label_mapping, heuristic_info, max_size_mask, disjoint_info, debug=False, use_leaf=False)
        # heuristic_info, label_mapping, _, _  = propagate_information(
        #             formula_1, masks, bitmaps, frontier, heuristic_name, heuristic_info, label_mapping,
        #             max_improvement, disjoint_info, num_hits, max_size_mask, length
        #         )
        # exit()
        # if len(label_node) == 2:
        #     ancestors_quantities = compute_ancestors_quantities(label_node, masks, heuristic_info, bitmaps)
        #     heuristic_info, label_mapping = update_heuristic_info(ancestors_quantities, heuristic_info, label_mapping)
        #     local_min_threshold = 0

        #     # Update stats considering the ancestors
        #     for ancestor, (_, ancestor_iou) in ancestors_quantities.items():
        #         if ancestor in visited:
        #             continue
        #         visited.append(ancestor)
        #         if ancestor_iou > best_results[0]:
        #             best_results = (ancestor_iou, ancestor)
        #         if ancestor_iou >= local_min_threshold:
        #             local_min_threshold = ancestor_iou
        #     if local_min_threshold > minimum_threshold:
        #         # Update the minimum threshold if the local minimum is higher
        #         minimum_threshold = local_min_threshold
        #         current_frontier = reduce_frontier(
        #             current_frontier, minimum_threshold
        #         )
        #         heapq.heapify(current_frontier)
        #         done = len(current_frontier) == 0
        #     print("Done")
        #     exit()
        # seg_quantities, neuron_quantities, concepts_quantities = heuristic_info
        # index_node = label_mapping[label_node]
        # print(label_node, compute_iou_from_quantities(concepts_quantities[index_node], num_hits=num_hits))
        # label_mask = utils.sparse_to_torch(mask_utils.get_formula_mask(
        #             label_node, masks
        #         )).to(bitmaps.device)
        # print(compute_exact_iou(label_mask, bitmaps, heuristic_info))
        # exit()
        # If node visited, skip it, this should not happen
        if debug and next_op_node=='INDIVIDUAL' and label_node in visited:
            done = len(current_frontier) == 0
            skipped_because_full_iou += 1
            raise ValueError(f"Node {node} already visited. This should not happen.")
            continue


        if debug and -e_node > recent_e_iou:
            done = len(current_frontier) == 0
            print(f"Skipping node {node} because its estimated IoU is too low: {-e_node} > {recent_e_iou}.")
            #exit()
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
        
        if debug:
            if i % 100 == 0:
                print(f"Iteration {i} Time: {(time.time() - init_time)/60:.2f} Stats: \t Nodes: Added to frontier {total_added_frontier} \t popped {popped_nodes} \t expanded: {expanded_nodes} \t Full IoU:{full_iou_nodes}  \t Updated: {nodes_updated} \t Skipped bc full IoU: {skipped_because_full_iou} \t Skipped bc recent: {skipped_because_recent}")
                #print(f"Times Avg: estimate_reduce {np.mean(list_estimate_reduce_frontier_time):.4f} \t update_reduced {np.mean(list_update_reduced_frontier_time):.4f} \t merge {np.mean(list_merge_time):.4f} \t estimate_iou {np.mean(list_time_estimate_iou_frontier):.4f} \t expanded {np.mean(list_expanded_time):.4f} \t avg_frontier_node {np.mean(list_avg_frontier_node_time):.4f} \t median_frontier_node {np.median(list_median_frontier_node_time):.4f} \t parsing_frontier {np.mean(list_parsing_frontier_time):.4f}")
                #print(f"Times Tot: estimate_reduce {sum(list_estimate_reduce_frontier_time):.4f} \t update_reduced {sum(list_update_reduced_frontier_time):.4f} \t merge {sum(list_merge_time):.4f} \t estimate_iou {sum(list_time_estimate_iou_frontier):.4f} \t expanded {sum(list_expanded_time):.4f} \t avg_frontier_node {sum(list_avg_frontier_node_time):.4f} \t median_frontier_node {sum(list_median_frontier_node_time):.4f} \t parsing_frontier {sum(list_parsing_frontier_time):.4f}")
                print(f"Times Avg:  expanded {np.mean(list_expanded_time):.4f} \t avg_frontier_node {np.mean(list_avg_frontier_node_time):.4f} \t median_frontier_node {np.median(list_median_frontier_node_time):.4f} \t parsing_frontier {np.mean(list_parsing_frontier_time):.4f}")
                print(f"Times Tot:  expanded {sum(list_expanded_time):.4f} \t avg_frontier_node {sum(list_avg_frontier_node_time):.4f} \t median_frontier_node {sum(list_median_frontier_node_time):.4f} \t parsing_frontier {sum(list_parsing_frontier_time):.4f}")
                print(f"Times Node Avg: esti_quantities {np.mean(list_esti_quantities_time):.4f}/{len(list_esti_quantities_time)} \t individual_estimation {np.mean(list_individual_estimation_time):.4f}/{len(list_individual_estimation_time)} \t or_chain {np.mean(list_or_chain_estimation_time):.4f}/{len(list_or_chain_estimation_time)} \t and_chain {np.mean(list_and_chain_estimation_time):.4f}/{len(list_and_chain_estimation_time)} \t and_not_chain {np.mean(list_and_not_chain_estimation_time):.4f}/{len(list_and_not_chain_estimation_time)} \t and_or_chain {np.mean(list_and_or_chain_estimation_time):.4f}/{len(list_and_or_chain_estimation_time)} \t comb_or_andnot_chain {np.mean(list_comb_or_andnot_chain_estimation_time):.4f}/{len(list_comb_or_andnot_chain_estimation_time)}")
                print(f"Times Node Tot: esti_quantities {sum(list_esti_quantities_time):.4f} \t individual_estimation {sum(list_individual_estimation_time):.4f} \t or_chain {sum(list_or_chain_estimation_time):.4f} \t and_chain {sum(list_and_chain_estimation_time):.4f} \t and_not_chain {sum(list_and_not_chain_estimation_time):.4f} \t and_or_chain {sum(list_and_or_chain_estimation_time):.4f} \t comb_or_andnot_chain {sum(list_comb_or_andnot_chain_estimation_time):.4f}")
                print("--------------------------------------------------------------------------------------------")
        else:
            print(f"Iteration {i} \t current frontier size: {len(current_frontier)} \t Node Esti: {round(e_node,4)} Threshold: {round(minimum_threshold, 4)} Node label: {label_node}", end='\r') 

        if next_op_node == 'INDIVIDUAL':
            # This is the case where the formula has no operations to expand
            # Compute Label Mask
            label_mask = utils.sparse_to_torch(mask_utils.get_formula_mask(
                    label_node, masks
                )).to(bitmaps.device)
            # Compute the exact IoU for the label node
            iou = compute_exact_iou(label_mask, bitmaps, heuristic_info)
            # print()
            # print(f"Computed IoU for node {label_node}: {iou:.4f} at iteration {i}")
            # print()
            # max_score, min_score, _ = optimal_heuristic.estimate_label_iou(heuristic_name,
            #     node, label_mapping, heuristic_info,  max_improvement=max_improvement, disjoint_info=disjoint_info, num_hits=num_hits, max_size_mask=max_size_mask,
            #     max_length=length, minimum_threshold=minimum_threshold, debug=True)
            # print(f"Estimated IoU for node {label_node}: max {max_score:} min {min_score} at iteration {i}")
            # if len(label_node) > 1:
            #     print(f"Old len Label Mapping: {len(label_mapping)}")
            #     heuristic_info, label_mapping, current_frontier , _ = propagate_information(
            #         label_node, minimum_threshold, masks, bitmaps, current_frontier, heuristic_name, heuristic_info, label_mapping,
            #         max_improvement, disjoint_info, num_hits, max_size_mask, length
            #     )
            #     print(f"New len Label Mapping: {len(label_mapping)}")

            # Add the node to the visited nodes
            visited.append(label_node)
            full_iou_nodes += 1


            if iou > best_results[0]:
                best_results = (iou, label_node)
            elif iou == best_results[0]:
                if len(label_node) < len(best_results[1]):
                    # We prefer shorter labels in case of equal IoU
                    best_results = (iou, label_node)

            # If the IoU is greater than the minimum threshold, we update the minimum threshold
            if iou > minimum_threshold:
                minimum_threshold = iou
                label_node = node[2]

                heuristic_info, label_mapping, current_frontier, _ = propagate_information(
                    label_node, minimum_threshold, masks, bitmaps, current_frontier, heuristic_name, heuristic_info, label_mapping,
                    max_improvement, disjoint_info, num_hits, max_size_mask, length
                )
                if len(current_frontier) > 0:
                    # Reduce the current frontier based on the new minimum threshold
                    current_frontier = reduce_frontier(
                        current_frontier, minimum_threshold)
                    heapq.heapify(current_frontier)
            done = len(current_frontier) == 0
            continue
        
        if debug and len(label_node) >= length:
            # compute real IoU
            done = len(current_frontier) == 0
            raise ValueError(
                f"Node {node} has a label of length {len(label_node)} which is greater than the maximum length {length}. This should not happen."
            )


        # Compute the updated frontier based on the new nodes
        if debug:
            # Expand the node to get the next frontier
            before_expand_time = time.time()
            next_frontier = expand_node(node, candidate_labels=candidate_labels, max_length=length)
            expand_time = time.time()

            expanded_nodes += len(next_frontier)
            list_expanded_time.append(expand_time - before_expand_time)
            current_frontier, minimum_info, debug_info, heuristic_info, label_mapping = update_frontier(
                bitmaps=bitmaps, masks=masks,
                past_frontier=current_frontier, new_nodes=next_frontier, 
                label_mapping=label_mapping, heuristic=heuristic_name, 
                heuristic_info=heuristic_info, max_improvement=max_improvement,
                num_hits=num_hits, max_size_mask=max_size_mask,
                length=length, global_min_threshold=minimum_threshold, disjoint_info=disjoint_info,
                debug=True
            )
            minimum_threshold, minimum_node = minimum_info

            # Update debug info
            total_added_frontier += debug_info['added_to_frontier']
            if debug_info['estimate_reduced_frontier']:
                list_estimate_reduce_frontier_time.append(debug_info['estimate_reduce_frontier_time'])
            if debug_info['update_reduced_frontier']:
                list_update_reduced_frontier_time.append(debug_info['update_reduce_frontier_time'])
            if debug_info['merged']:
                list_merge_time.append(debug_info['merge_time'])
            list_time_estimate_iou_frontier.append(debug_info['time_estimate_iou_frontier'])
            list_avg_frontier_node_time.append(debug_info['avg_time_frontier_node'])
            list_median_frontier_node_time.append(debug_info['median_time_frontier_node'])
            list_parsing_frontier_time.append(debug_info['parsing_frontier_time'])
            list_esti_quantities_time.extend(debug_info['esti_quantities_time'])
            list_individual_estimation_time.extend(debug_info['individual_estimation_time'])
            list_or_chain_estimation_time.extend(debug_info['or_chain_estimation_time'])
            list_and_chain_estimation_time.extend(debug_info['and_chain_estimation_time'])
            list_and_not_chain_estimation_time.extend(debug_info['and_not_chain_estimation_time'])
            list_and_or_chain_estimation_time.extend(debug_info['and_or_chain_estimation_time'])
            list_comb_or_andnot_chain_estimation_time.extend(debug_info['comb_or_andnot_chain_estimation_time'])
        else:
            # Expand the node to get the next frontier
            next_frontier, next_overlap_frontier = expand_node(node, candidate_labels=candidate_labels, max_length=length, disjoint_info=disjoint_info)
            
            # Expand the overlapping frontier
            overlapping_frontier.extend(next_overlap_frontier)

            # Compute the estimation for the next frontier
        current_frontier, minimum_info, heuristic_info, label_mapping = update_frontier(
            bitmaps=bitmaps, masks=masks,
            past_frontier=current_frontier, new_nodes=next_frontier, 
            label_mapping=label_mapping, heuristic=heuristic_name, 
            heuristic_info=heuristic_info, max_improvement=max_improvement,
            num_hits=num_hits, max_size_mask=max_size_mask,
            length=length, global_min_threshold=minimum_threshold, disjoint_info=disjoint_info
        )
        minimum_threshold, minimum_node = minimum_info

        i += 1
        done = len(current_frontier) == 0
    best_iou = best_results[0]
    best_label = best_results[1]

    # # Compute linear time 
    # ratio_single = []
    # iou_single = []
    # for c in candidate_labels:
    #     ratio = compute_ratio(
    #         utils.sparse_to_torch(mask_utils.get_formula_mask(c, masks)).to(bitmaps.device),
    #         bitmaps, heuristic_info
    #     )
    #     ratio_single.append((c, ratio))
    #     iou = compute_exact_iou(
    #         utils.sparse_to_torch(mask_utils.get_formula_mask(c, masks)).to(bitmaps.device),
    #         bitmaps, heuristic_info
    #     )
    #     iou_single.append((c, iou))
    # # sort iou single by iou
    # iou_single = sorted(iou_single, key=lambda x: x[1], reverse=True)
    # top_3_iou = iou_single[:3]

    # # sort ratio single by ratio
    # ratio_single = sorted(ratio_single, key=lambda x: x[1], reverse=True)
    # top_3 = ratio_single[:3]
    # # Build best linear
    # best_linear = None
    # for c, ratio in top_3:
    #     if best_linear is None and ratio > 0:
    #         iou_candidate = compute_exact_iou(
    #             utils.sparse_to_torch(mask_utils.get_formula_mask(c, masks)).to(bitmaps.device),
    #             bitmaps, heuristic_info
    #         )
    #         best_linear = (c, iou_candidate)
    #     elif best_linear is not None:
    #         candidate = F.Or(best_linear[0], c)
    #         iou_candidate = compute_exact_iou(
    #             utils.sparse_to_torch(mask_utils.get_formula_mask(candidate, masks)).to(bitmaps.device),
    #             bitmaps, heuristic_info
    #         )
    #         if iou_candidate > best_linear[1]:
    #             best_linear = (candidate, iou_candidate)
    # common_stem = F.Or(F.Leaf(139), F.Leaf(254))
    # ratio = compute_ratio(
    #         utils.sparse_to_torch(mask_utils.get_formula_mask(common_stem, masks)).to(bitmaps.device),
    #         bitmaps, heuristic_info, debug=True
    #     )
    # formula_a = F.Or(common_stem, F.Leaf(448))
    # formula_b = F.Or(common_stem, F.Leaf(511))

    # # Build best iou linear
    # best_iou_linear = None
    # for c, iou in top_3_iou:
    #     if best_iou_linear is None or iou > 0:
    #         best_iou_linear = (c, iou)
    #     elif best_iou_linear is not None:
    #         candidate = F.Or(best_iou_linear[0], c)
    #         iou_candidate = compute_exact_iou(
    #             utils.sparse_to_torch(mask_utils.get_formula_mask(candidate, masks)).to(bitmaps.device),
    #             bitmaps, heuristic_info
    #         )
    #         if iou_candidate > best_iou_linear[1]:
    #             best_iou_linear = (candidate, iou_candidate)
    
    # print(f"Best linear: {best_linear[0]} with IoU {best_linear[1]}")
    # print(f"Best label: {best_label} with IoU {best_iou}")
    # assert best_label == best_linear[0], f"Best label {best_label} does not match best linear {best_linear[0]} {ratio_single}"
    
    
    return best_label, best_iou, len(visited), popped_nodes


def perform_exhaustive_heuristic_search(
    heuristic,
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
    
    # Max improvement for this neuron
    max_improvement = compute_max_improvement(heuristic_info, max_size_mask, num_hits, length)
    
    # Candidate concepts
    candidate_labels = [F.Leaf(c) for c in range(len(masks))]
    label_mapping = {F.Leaf(c): c for c in range(len(masks))}
    
    # Search parameters
    done = False

    # Debugging information
    num_hits = num_hits.item()
    debug = True
    total_added_frontier = 0
    list_estimate_reduce_frontier_time = []
    list_update_reduced_frontier_time = []
    list_merge_time = []
    list_time_estimate_iou_frontier = []
    list_expanded_time = []
    list_avg_frontier_node_time = []
    list_median_frontier_node_time = []
    list_parsing_frontier_time = []
    popped_nodes = 0
    expanded_nodes = 0
    full_iou_nodes = 0
    nodes_updated = 0
    skipped_because_full_iou = 0
    skipped_because_recent = 0

    # Node times 
    list_esti_quantities_time = []
    list_individual_estimation_time = []
    list_or_chain_estimation_time = []
    list_and_chain_estimation_time = []
    list_and_not_chain_estimation_time = []
    list_and_or_chain_estimation_time = []
    list_comb_or_andnot_chain_estimation_time = []


    # Initialize the frontier with all the candidate labels
    frontier = [(0.0, None, k, None) for k in label_mapping.keys()]
    if debug:
        current_frontier, minimum_threshold, debug_info = update_frontier(
            past_frontier=None, new_nodes=frontier, label_mapping=label_mapping,
            heuristic=heuristic, heuristic_info=heuristic_info, 
            max_improvement=max_improvement,
            num_hits=num_hits, max_size_mask=max_size_mask,
            length=length, global_min_threshold=0.0, disjoint_info=disjoint_info, debug=debug
        )
        # Update debug info
        total_added_frontier += debug_info['added_to_frontier']
        if debug_info['estimate_reduced_frontier']:
            list_estimate_reduce_frontier_time.append(debug_info['estimate_reduce_frontier_time'])
        if debug_info['update_reduced_frontier']:
            list_update_reduced_frontier_time.append(debug_info['update_reduce_frontier_time'])
        if debug_info['merged']:
            list_merge_time.append(debug_info['merge_time'])
        list_time_estimate_iou_frontier.append(debug_info['time_estimate_iou_frontier'])
        list_avg_frontier_node_time.append(debug_info['avg_time_frontier_node'])
        list_median_frontier_node_time.append(debug_info['median_time_frontier_node'])
        list_parsing_frontier_time.append(debug_info['parsing_frontier_time'])
        list_esti_quantities_time.extend(debug_info['esti_quantities_time'])
        list_individual_estimation_time.extend(debug_info['individual_estimation_time'])
        list_or_chain_estimation_time.extend(debug_info['or_chain_estimation_time'])
        list_and_chain_estimation_time.extend(debug_info['and_chain_estimation_time'])
        list_and_not_chain_estimation_time.extend(debug_info['and_not_chain_estimation_time'])
        list_and_or_chain_estimation_time.extend(debug_info['and_or_chain_estimation_time'])
        list_comb_or_andnot_chain_estimation_time.extend(debug_info['comb_or_andnot_chain_estimation_time'])
    else:
        current_frontier, minimum_threshold = update_frontier(
            past_frontier=None, new_nodes=frontier, label_mapping=label_mapping,
            heuristic=heuristic, heuristic_info=heuristic_info, 
            max_improvement=max_improvement,
            num_hits=num_hits, max_size_mask=max_size_mask,
            length=length, global_min_threshold=0.0, disjoint_info=disjoint_info, debug=debug
        )


    done = len(current_frontier) == 0
    i = 0
    best_results = (0.0, None) # (IoU, label)
    visited = []
    recent_nodes = []
    recent_e_iou = 2
    init_time = time.time()

    while not done:
        node = heapq.heappop(current_frontier)
        popped_nodes += 1
        e_node = node[0]
        label_node = node[2]
        next_op_node = node[1]
        
        # If node visited, skip it, this should not happen
        if debug and next_op_node=='INDIVIDUAL' and label_node in visited:
            done = len(current_frontier) == 0
            skipped_because_full_iou += 1
            raise ValueError(f"Node {node} already visited. This should not happen.")
            continue


        if debug and -e_node > recent_e_iou:
            done = len(current_frontier) == 0
            print(f"Skipping node {node} because its estimated IoU is too low: {-e_node} > {recent_e_iou}.")
            exit()
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
        
        if debug:
            if i % 10 == 0:
                print(f"Iteration {i} Time: {(time.time() - init_time)/60:.2f} Stats: \t Nodes: Added to frontier {total_added_frontier} \t popped {popped_nodes} \t expanded: {expanded_nodes} \t Full IoU:{full_iou_nodes}  \t Updated: {nodes_updated} \t Skipped bc full IoU: {skipped_because_full_iou} \t Skipped bc recent: {skipped_because_recent}")
                #print(f"Times Avg: estimate_reduce {np.mean(list_estimate_reduce_frontier_time):.4f} \t update_reduced {np.mean(list_update_reduced_frontier_time):.4f} \t merge {np.mean(list_merge_time):.4f} \t estimate_iou {np.mean(list_time_estimate_iou_frontier):.4f} \t expanded {np.mean(list_expanded_time):.4f} \t avg_frontier_node {np.mean(list_avg_frontier_node_time):.4f} \t median_frontier_node {np.median(list_median_frontier_node_time):.4f} \t parsing_frontier {np.mean(list_parsing_frontier_time):.4f}")
                #print(f"Times Tot: estimate_reduce {sum(list_estimate_reduce_frontier_time):.4f} \t update_reduced {sum(list_update_reduced_frontier_time):.4f} \t merge {sum(list_merge_time):.4f} \t estimate_iou {sum(list_time_estimate_iou_frontier):.4f} \t expanded {sum(list_expanded_time):.4f} \t avg_frontier_node {sum(list_avg_frontier_node_time):.4f} \t median_frontier_node {sum(list_median_frontier_node_time):.4f} \t parsing_frontier {sum(list_parsing_frontier_time):.4f}")
                print(f"Times Avg:  expanded {np.mean(list_expanded_time):.4f} \t avg_frontier_node {np.mean(list_avg_frontier_node_time):.4f} \t median_frontier_node {np.median(list_median_frontier_node_time):.4f} \t parsing_frontier {np.mean(list_parsing_frontier_time):.4f}")
                print(f"Times Tot:  expanded {sum(list_expanded_time):.4f} \t avg_frontier_node {sum(list_avg_frontier_node_time):.4f} \t median_frontier_node {sum(list_median_frontier_node_time):.4f} \t parsing_frontier {sum(list_parsing_frontier_time):.4f}")
                print(f"Times Node Avg: esti_quantities {np.mean(list_esti_quantities_time):.4f}/{len(list_esti_quantities_time)} \t individual_estimation {np.mean(list_individual_estimation_time):.4f}/{len(list_individual_estimation_time)} \t or_chain {np.mean(list_or_chain_estimation_time):.4f}/{len(list_or_chain_estimation_time)} \t and_chain {np.mean(list_and_chain_estimation_time):.4f}/{len(list_and_chain_estimation_time)} \t and_not_chain {np.mean(list_and_not_chain_estimation_time):.4f}/{len(list_and_not_chain_estimation_time)} \t and_or_chain {np.mean(list_and_or_chain_estimation_time):.4f}/{len(list_and_or_chain_estimation_time)} \t comb_or_andnot_chain {np.mean(list_comb_or_andnot_chain_estimation_time):.4f}/{len(list_comb_or_andnot_chain_estimation_time)}")
                print(f"Times Node Tot: esti_quantities {sum(list_esti_quantities_time):.4f} \t individual_estimation {sum(list_individual_estimation_time):.4f} \t or_chain {sum(list_or_chain_estimation_time):.4f} \t and_chain {sum(list_and_chain_estimation_time):.4f} \t and_not_chain {sum(list_and_not_chain_estimation_time):.4f} \t and_or_chain {sum(list_and_or_chain_estimation_time):.4f} \t comb_or_andnot_chain {sum(list_comb_or_andnot_chain_estimation_time):.4f}")
                print("--------------------------------------------------------------------------------------------")

        else:
            print(f"Iteration {i} \t current frontier size: {len(current_frontier)} \t Node Esti: {round(e_node,4)} Threshold: {round(minimum_threshold, 4)} Node label: {label_node}", end='\r') 

        if next_op_node == 'INDIVIDUAL':
            # This is the case where the formula has no operations to expand

            # Compute the masks of all the label ancestors including the label itself
            # Number of ancestors = len(label)
            ancestors_masks = get_ancestor_masks(
                    label_node, masks
                )
            
            # Compute the exact IoU for the label node and update the heuristic info
            label_mask = utils.sparse_to_torch(ancestors_masks[label_node]).to(bitmaps.device)

            # Old way to compute the IoU and update the heuristic info
            # iou, label_mapping, heuristic_info = compute_exact_iou_and_update_heuristic_info(
            #     label_node, label_mapping, label_mask, bitmaps, heuristic_info)
            
            # Compute the exact IoU for the label node
            iou = compute_exact_iou(label_mask, bitmaps, heuristic_info)

            # Add the node to the visited nodes
            visited.append(label_node)
            full_iou_nodes += 1


            if iou > best_results[0]:
                best_results = (iou, label_node)
            elif iou == best_results[0]:
                if len(label_node) < len(best_results[1]):
                    # We prefer shorter labels in case of equal IoU
                    best_results = (iou, label_node)
                else:
                    raise ValueError(
                        f"Found a label with the same IoU {iou} but longer than the best one: {best_results[1]} vs {label_node}"
                    )
        
            local_minimum_threshold = iou 
            
            # Remove from ancestors leaf, negative, and already visited labels since we have already their information

            ancestors = [a for a in ancestors_masks.keys() if not isinstance(a, F.Leaf) and not isinstance(a, F.Not) and a not in visited]

            # Update heuristic info based on the ancestors
            if len(ancestors) > 0:
                # Compute exact IoU for the ancestors and update the heuristic info
                for ancestor in ancestors:
                    ancestor_mask = utils.sparse_to_torch(ancestors_masks[ancestor]).to(bitmaps.device)
                    ancestor_iou, label_mapping, heuristic_info = compute_exact_iou_and_update_heuristic_info(
                        ancestor, label_mapping, ancestor_mask, bitmaps, heuristic_info)
                    visited.append((ancestor, []))
                    full_iou_nodes += 1
                    if ancestor_iou > best_results[0]:
                        best_results = (ancestor_iou, ancestor)
                    elif ancestor_iou == best_results[0]:
                        if len(ancestor) < len(best_results[1]):
                            # We prefer shorter labels in case of equal IoU
                            best_results = (ancestor_iou, ancestor)
                        else:
                            raise ValueError(
                                f"Found a label with the same IoU {ancestor_iou} but longer than the best one: {best_results[1]} vs {ancestor}"
                            )
                    if ancestor_iou > local_minimum_threshold:
                        local_minimum_threshold = ancestor_iou

                # Update only the nodes in the frontier that have common ancestors with the current node
                frontier_to_update = []
                for ancestor in ancestors:
                    for frontier_node in current_frontier:
                        if ancestor in frontier_node[2].tree_path() and frontier_node not in frontier_to_update:
                            frontier_to_update.append(frontier_node)
                nodes_updated += len(frontier_to_update)

                # Sort frontier based on the length of the node
                frontier_to_update.sort(key=lambda x: len(x[2]), reverse=True)
                for node_to_update in frontier_to_update:
                    # This operation is slow and needs to be optimized
                    index_node = current_frontier.index(node_to_update)
                    paths_to_update = node_to_update[3]
                    old_esti = current_frontier[index_node][0]
                    label = node_to_update[2]
                    new_max_score, new_min_score = optimal_heuristic.update_optimal_label_iou(
                        label, paths_to_update,  label_mapping, heuristic_info, max_improvement, disjoint_info, num_hits, max_size_mask,
                    length)

                    if -new_max_score == old_esti and new_min_score <= minimum_threshold:
                        # If the max score is the same as the old estimate and there are no updates on the threshold, we can skip this node
                        continue

                    # Replace the node in the current frontier with the new estimate
                    current_frontier.remove(node_to_update)

                    if new_min_score > local_minimum_threshold:
                        local_minimum_threshold = new_min_score
         
                    if new_max_score >= minimum_threshold:
                        current_frontier.append((-new_max_score, paths_to_update, label))
                heapq.heapify(current_frontier)
            
            if local_minimum_threshold > minimum_threshold:
                minimum_threshold = local_minimum_threshold
                if len(current_frontier) > 0:
                    # Reduce the current frontier based on the new minimum threshold
                    current_frontier = reduce_frontier(
                        current_frontier, minimum_threshold)
                    heapq.heapify(current_frontier)
            done = len(current_frontier) == 0
            #print(f"Time to compute exact IoU for  {time.time() - time_no_ops:.2f} seconds")
            continue
        
        if debug and len(label_node) >= length:
            # compute real IoU
            done = len(current_frontier) == 0
            raise ValueError(
                f"Node {node} has a label of length {len(label_node)} which is greater than the maximum length {length}. This should not happen."
            )




        # Compute the updated frontier based on the new nodes
        if debug:
            # Expand the node to get the next frontier
            before_expand_time = time.time()
            next_frontier = expand_node(node, candidate_labels, max_length=length)
            expand_time = time.time()

            expanded_nodes += len(next_frontier)
            list_expanded_time.append(expand_time - before_expand_time)
            current_frontier, minimum_threshold, debug_info = update_frontier(
                past_frontier=current_frontier, new_nodes=next_frontier, 
                label_mapping=label_mapping, heuristic=heuristic, 
                heuristic_info=heuristic_info, max_improvement=max_improvement,
                num_hits=num_hits, max_size_mask=max_size_mask,
                length=length, global_min_threshold=minimum_threshold, disjoint_info=disjoint_info,
                debug=True
            )
            # Update debug info
            total_added_frontier += debug_info['added_to_frontier']
            if debug_info['estimate_reduced_frontier']:
                list_estimate_reduce_frontier_time.append(debug_info['estimate_reduce_frontier_time'])
            if debug_info['update_reduced_frontier']:
                list_update_reduced_frontier_time.append(debug_info['update_reduce_frontier_time'])
            if debug_info['merged']:
                list_merge_time.append(debug_info['merge_time'])
            list_time_estimate_iou_frontier.append(debug_info['time_estimate_iou_frontier'])
            list_avg_frontier_node_time.append(debug_info['avg_time_frontier_node'])
            list_median_frontier_node_time.append(debug_info['median_time_frontier_node'])
            list_parsing_frontier_time.append(debug_info['parsing_frontier_time'])
            list_esti_quantities_time.extend(debug_info['esti_quantities_time'])
            list_individual_estimation_time.extend(debug_info['individual_estimation_time'])
            list_or_chain_estimation_time.extend(debug_info['or_chain_estimation_time'])
            list_and_chain_estimation_time.extend(debug_info['and_chain_estimation_time'])
            list_and_not_chain_estimation_time.extend(debug_info['and_not_chain_estimation_time'])
            list_and_or_chain_estimation_time.extend(debug_info['and_or_chain_estimation_time'])
            list_comb_or_andnot_chain_estimation_time.extend(debug_info['comb_or_andnot_chain_estimation_time'])
        else:
            # Expand the node to get the next frontier
            next_frontier = expand_node(node, candidate_labels, max_length=length)
            
            # Compute the estimation for the next frontier
            current_frontier, minimum_threshold = update_frontier(
                past_frontier=current_frontier, new_nodes=next_frontier, 
                label_mapping=label_mapping, heuristic=heuristic, 
                heuristic_info=heuristic_info, max_improvement=max_improvement,
                num_hits=num_hits, max_size_mask=max_size_mask,
                length=length, global_min_threshold=minimum_threshold, disjoint_info=disjoint_info
            )

        i += 1
        done = len(current_frontier) == 0
    best_iou = best_results[0]
    best_label = best_results[1]
    return best_label, best_iou, len(visited), popped_nodes

def get_quantity_by_index(concepts_quantities, index):
    common_intersection_tuple, unique_intersection_tuple, common_extras_tuple, unique_extras_tuple, common_uncovered_tuple, unique_uncovered_tuple = concepts_quantities
    common_intersection = common_intersection_tuple[0]
    common_intersection_sum = common_intersection_tuple[1]
    unique_intersection = unique_intersection_tuple[0]
    unique_intersection_sum = unique_intersection_tuple[1]
    common_extras = common_extras_tuple[0]
    common_extras_sum = common_extras_tuple[1]
    unique_extras = unique_extras_tuple[0]
    unique_extras_sum = unique_extras_tuple[1]
    common_uncovered = common_uncovered_tuple[0]
    common_uncovered_sum = common_uncovered_tuple[1]
    unique_uncovered = unique_uncovered_tuple[0]
    unique_uncovered_sum = unique_uncovered_tuple[1]
    return (common_intersection[index], common_intersection_sum[index]), \
            (unique_intersection[index], unique_intersection_sum[index]), \
            (common_extras[index], common_extras_sum[index]), \
            (unique_extras[index], unique_extras_sum[index]), \
            (common_uncovered[index], common_uncovered_sum[index]), \
            (unique_uncovered[index], unique_uncovered_sum[index])

def add_quantity(concept_quantities, label_quantities):
    """
    Add the quantities of a label to the existing concept quantities.
    
    Args:
        concept_quantities (tuple): The existing concept quantities.
        label_quantities (tuple): The quantities of the label to add.
        
    Returns:
        tuple: Updated concept quantities.
    """

    common_intersection_tuple, unique_intersection_tuple, common_extras_tuple, unique_extras_tuple, common_uncovered_tuple, unique_uncovered_tuple = concept_quantities
    common_intersection = common_intersection_tuple[0]
    common_intersection_sum = common_intersection_tuple[1]
    unique_intersection = unique_intersection_tuple[0]
    unique_intersection_sum = unique_intersection_tuple[1]
    common_extras = common_extras_tuple[0]
    common_extras_sum = common_extras_tuple[1]
    unique_extras = unique_extras_tuple[0]
    unique_extras_sum = unique_extras_tuple[1]
    common_uncovered = common_uncovered_tuple[0]
    common_uncovered_sum = common_uncovered_tuple[1]
    unique_uncovered = unique_uncovered_tuple[0]
    unique_uncovered_sum = unique_uncovered_tuple[1]
    label_common_intersection_tuple, label_unique_intersection_tuple, label_common_extras_tuple, label_unique_extras, label_uncovered = label_quantities
    label_common_intersection, label_common_intersection_sum = label_common_intersection_tuple
    label_unique_intersection, label_unique_intersection_sum = label_unique_intersection_tuple
    label_common_extras, label_common_extras_sum = label_common_extras_tuple
    label_unique_extras, label_unique_extras_sum = label_unique_extras
    label_uncovered, label_uncovered_sum = label_uncovered
    # Update heuristic info
    common_intersection.append(label_common_intersection)
    common_intersection_sum.append(label_common_intersection_sum)
    unique_intersection.append(label_unique_intersection)
    unique_intersection_sum.append(label_unique_intersection_sum)
    common_extras.append(label_common_extras)
    common_extras_sum.append(label_common_extras_sum)
    unique_extras.append(label_unique_extras)
    unique_extras_sum.append(label_unique_extras_sum)
    common_uncovered.append(label_uncovered)
    common_uncovered_sum.append(label_uncovered_sum)
    unique_uncovered.append(label_uncovered)
    unique_uncovered_sum.append(label_uncovered_sum)
    return (common_intersection, common_intersection_sum), \
           (unique_intersection, unique_intersection_sum), \
           (common_extras, common_extras_sum), \
           (unique_extras, unique_extras_sum), \
           (common_uncovered, common_uncovered_sum), \
           (unique_uncovered, unique_uncovered_sum)

## OLD just for reference, not used in the current implementation
def estimate_disjoint_label_info(label, left_quantities, right_quantities, neuron_unique, neuron_common, neuron_coverable, neuron_sum, max_size_mask, disjoint_masks):

    """
    Estimate the label information for a given label based on its left and right quantities.

    Args:
        label (F.Formula): The label for which to estimate the label information.
        left_quantities (tuple): Quantities from the left child of the label.
        right_quantities (tuple): Quantities from the right child of the label.

    Returns:
        tuple: Estimated quantities for the label.
    """
    left_quantities_sample = left_quantities
    right_quantities_sample = right_quantities
    left_common_intersection, left_unique_intersection, left_common_extras, left_unique_extras, left_uncovered = left_quantities_sample
    
    max_left_common_intersection_tuple, min_left_common_intersection_tuple = get_max_min_quantity(left_common_intersection)
    max_left_unique_intersection_tuple, min_left_unique_intersection_tuple = get_max_min_quantity(left_unique_intersection)
    max_left_common_extras_tuple, min_left_common_extras_tuple = get_max_min_quantity(left_common_extras)
    max_left_unique_extras_tuple, min_left_unique_extras_tuple = get_max_min_quantity(left_unique_extras)
    max_left_uncovered_tuple, min_left_uncovered_tuple = get_max_min_quantity(left_uncovered)
    max_left_common_intersection = max_left_common_intersection_tuple[0]
    min_left_common_intersection = min_left_common_intersection_tuple[0]
    max_left_unique_intersection = max_left_unique_intersection_tuple[0]
    min_left_unique_intersection = min_left_unique_intersection_tuple[0]
    max_left_common_extras = max_left_common_extras_tuple[0]
    min_left_common_extras = min_left_common_extras_tuple[0]
    max_left_unique_extras = max_left_unique_extras_tuple[0]
    min_left_unique_extras = min_left_unique_extras_tuple[0]
    max_left_uncovered = max_left_uncovered_tuple[0]
    min_left_uncovered = min_left_uncovered_tuple[0]

    right_common_intersection, right_unique_intersection, right_common_extras, right_unique_extras, right_uncovered = right_quantities_sample
    max_right_common_intersection_tuple, min_right_common_intersection = get_max_min_quantity(right_common_intersection)
    max_right_unique_intersection_tuple, min_right_unique_intersection = get_max_min_quantity(right_unique_intersection)
    max_right_common_extras_tuple, min_right_common_extras = get_max_min_quantity(right_common_extras)
    max_right_unique_extras_tuple, min_right_unique_extras = get_max_min_quantity(right_unique_extras)
    max_right_uncovered_tuple, min_right_uncovered = get_max_min_quantity(right_uncovered)
    max_right_common_intersection = max_right_common_intersection_tuple[0]
    min_right_common_intersection = min_right_common_intersection[0]
    max_right_unique_intersection = max_right_unique_intersection_tuple[0]
    min_right_unique_intersection = min_right_unique_intersection[0]
    max_right_common_extras = max_right_common_extras_tuple[0]
    min_right_common_extras = min_right_common_extras[0]
    max_right_unique_extras = max_right_unique_extras_tuple[0]
    min_right_unique_extras = min_right_unique_extras[0]
    max_right_uncovered = max_right_uncovered_tuple[0]
    min_right_uncovered = min_right_uncovered[0]

    if isinstance(label, F.Or):
        max_unique_intersection = max_left_unique_intersection + max_right_unique_intersection # I_max^u(L) + I_max^u(c)
        min_unique_intersection = min_left_unique_intersection + min_right_unique_intersection # I_min^u(L) + I_min^u(c)

        max_common_intersection = max_left_common_intersection + max_right_common_intersection # I_max^c(L) + I_max^c(c)
        min_common_intersection = min_left_common_intersection + min_right_common_intersection # I_min^c(L) + I_min^c(c)

        if ((np.all(max_unique_intersection <= max_left_unique_intersection) and \
            np.all(max_common_intersection <= max_left_common_intersection) and \
            np.all(min_unique_intersection <= min_left_unique_intersection) and \
            np.all(min_common_intersection <= min_left_common_intersection)) or \
            (np.all(max_unique_intersection <= max_right_unique_intersection) and \
            np.all(max_common_intersection <= max_right_common_intersection) and \
            np.all(min_unique_intersection <= min_right_unique_intersection) and \
            np.all(min_common_intersection <= min_right_common_intersection))):
            # If one of the two side does not change to the intersection, we can discard this formula
            return None, None, None, None, None

        min_uncovered  = np.clip(
            min_left_uncovered - max_right_common_intersection - max_right_unique_intersection, a_min=0, a_max=None) # max(0, U_min^L - I_max^c(c) - I_max^u(c))
        max_uncovered = np.clip(
            max_left_uncovered - min_right_common_intersection - min_right_unique_intersection, a_min=0, a_max=None)

        min_unique_extras = min_left_unique_extras + min_right_unique_extras # E_min^u(L) + E_min^u(c)
        max_unique_extras = max_left_unique_extras + max_right_unique_extras # E_max^u(L) + E_max^u(c)
        
        min_common_extras = min_left_common_extras + min_right_common_extras # E_min^c(L) + E_min^c(c)
        max_common_extras = max_left_common_extras + max_right_common_extras # E_max^c(L) + E_max^c(c)

    elif isinstance(label, F.And) and isinstance(label.right, F.Not):
        # Since they are disjoint, there is not a counter example of their presence together
        # Everything would end up the same of the left label (see commentetd part below)
        return None, None, None, None, None
    
        # #neuron_activation =  neuron_unique + neuron_common
        # max_unique_intersection = max_left_unique_intersection
        # min_unique_intersection = min_left_unique_intersection

        # min_common_intersection = min_left_common_intersection
        # max_common_intersection = max_left_common_intersection

        # min_uncovered = np.maximum(
        #     min_left_uncovered, min_right_common_intersection + min_right_unique_intersection
        # ) # max(U_min^L, I_min^c(c) + I_min^u(c))
        # max_uncovered = neuron_coverable - np.clip(
        #     max_left_common_intersection + max_right_uncovered - neuron_coverable, a_min=0, a_max=None
        # ) # N^u + N^c - max(0, I_max^c(L) + U_max^c - N^u - N^c)

        # min_unique_extras = min_left_unique_extras # E_min^u(L)
        # max_unique_extras = max_left_unique_extras # E_max^u(L)

        # min_common_extras = min_left_common_extras # E_min^c(L)
        # max_common_extras = max_left_common_extras
        # if (np.all(max_unique_extras >= max_left_unique_extras) and \
        #     np.all(max_common_extras >= max_left_common_extras) and \
        #     np.all(min_unique_extras >= min_left_unique_extras) and \
        #     np.all(min_common_extras >= min_left_common_extras) and \
        #     np.all(max_common_intersection <= max_left_common_intersection) and \
        #     np.all(min_common_intersection <= min_left_common_intersection)):
        #     #print("Discarding formula: ", label)
        #     # If one of the two side does not contribute to the intersection, we can discard this formula
        #     return None, None, None, None, None

    elif isinstance(label, F.And):
        # AND of disjoint labels is zero by definition
            return None, None, None, None, None
    else:
        raise ValueError(f"Unknown label type: {type(label)}")

    return ((max_common_intersection, max_common_intersection.sum()), (min_common_intersection, min_common_intersection.sum())), ((max_unique_intersection, max_unique_intersection.sum()), (min_unique_intersection, min_unique_intersection.sum())), \
        ((max_common_extras, max_common_extras.sum()), (min_common_extras, min_common_extras.sum())), \
        ((max_unique_extras, max_unique_extras.sum()), (min_unique_extras, min_unique_extras.sum())), \
        ((max_uncovered, max_uncovered.sum()), (min_uncovered, min_uncovered.sum()))

def estimate_label_info(label, left_quantities, right_quantities, neuron_unique, neuron_common, neuron_coverable, neuron_sum, max_size_mask):

    """
    Estimate the label information for a given label based on its left and right quantities.

    Args:
        label (F.Formula): The label for which to estimate the label information.
        left_quantities (tuple): Quantities from the left child of the label.
        right_quantities (tuple): Quantities from the right child of the label.

    Returns:
        tuple: Estimated quantities for the label.
    """
    left_quantities_sample = left_quantities
    right_quantities_sample = right_quantities
    left_common_intersection, left_unique_intersection, left_common_extras, left_unique_extras, left_uncovered = left_quantities_sample
    
    max_left_common_intersection_tuple, min_left_common_intersection_tuple = get_max_min_quantity(left_common_intersection)
    max_left_unique_intersection_tuple, min_left_unique_intersection_tuple = get_max_min_quantity(left_unique_intersection)
    max_left_common_extras_tuple, min_left_common_extras_tuple = get_max_min_quantity(left_common_extras)
    max_left_unique_extras_tuple, min_left_unique_extras_tuple = get_max_min_quantity(left_unique_extras)
    max_left_uncovered_tuple, min_left_uncovered_tuple = get_max_min_quantity(left_uncovered)
    max_left_common_intersection = max_left_common_intersection_tuple[0]
    min_left_common_intersection = min_left_common_intersection_tuple[0]
    max_left_unique_intersection = max_left_unique_intersection_tuple[0]
    min_left_unique_intersection = min_left_unique_intersection_tuple[0]
    max_left_common_extras = max_left_common_extras_tuple[0]
    min_left_common_extras = min_left_common_extras_tuple[0]
    max_left_unique_extras = max_left_unique_extras_tuple[0]
    min_left_unique_extras = min_left_unique_extras_tuple[0]
    max_left_uncovered = max_left_uncovered_tuple[0]
    min_left_uncovered = min_left_uncovered_tuple[0]

    right_common_intersection, right_unique_intersection, right_common_extras, right_unique_extras, right_uncovered = right_quantities_sample
    max_right_common_intersection_tuple, min_right_common_intersection = get_max_min_quantity(right_common_intersection)
    max_right_unique_intersection_tuple, min_right_unique_intersection = get_max_min_quantity(right_unique_intersection)
    max_right_common_extras_tuple, min_right_common_extras = get_max_min_quantity(right_common_extras)
    max_right_unique_extras_tuple, min_right_unique_extras = get_max_min_quantity(right_unique_extras)
    max_right_uncovered_tuple, min_right_uncovered = get_max_min_quantity(right_uncovered)
    max_right_common_intersection = max_right_common_intersection_tuple[0]
    min_right_common_intersection = min_right_common_intersection[0]
    max_right_unique_intersection = max_right_unique_intersection_tuple[0]
    min_right_unique_intersection = min_right_unique_intersection[0]
    max_right_common_extras = max_right_common_extras_tuple[0]
    min_right_common_extras = min_right_common_extras[0]
    max_right_unique_extras = max_right_unique_extras_tuple[0]
    min_right_unique_extras = min_right_unique_extras[0]
    max_right_uncovered = max_right_uncovered_tuple[0]
    min_right_uncovered = min_right_uncovered[0]

    if isinstance(label, F.Or):
        max_unique_intersection = max_left_unique_intersection + max_right_unique_intersection # I_max^u(L) + I_max^u(c)
        min_unique_intersection = min_left_unique_intersection + min_right_unique_intersection # I_min^u(L) + I_min^u(c)

        min_common_intersection = np.maximum(min_left_common_intersection, min_right_common_intersection) # max(I_min^c(L), I_min^c(c))
        max_common_intersection = np.minimum(
            np.maximum(
                max_left_common_intersection + np.minimum(
                    max_left_uncovered - min_right_unique_intersection,
                    max_right_common_intersection),
                max_right_common_intersection + np.minimum(
                    max_right_uncovered - min_left_unique_intersection,
                    max_left_common_intersection)),
                neuron_coverable - min_left_unique_intersection - min_right_unique_intersection)

        if ((np.all(max_unique_intersection <= max_left_unique_intersection) and \
            np.all(max_common_intersection <= max_left_common_intersection) and \
            np.all(min_unique_intersection <= min_left_unique_intersection) and \
            np.all(min_common_intersection <= min_left_common_intersection)) or \
            (np.all(max_unique_intersection <= max_right_unique_intersection) and \
            np.all(max_common_intersection <= max_right_common_intersection) and \
            np.all(min_unique_intersection <= min_right_unique_intersection) and \
            np.all(min_common_intersection <= min_right_common_intersection))):
            # If one of the two side does not change to the intersection, we can discard this formula
            return None, None, None, None, None

        min_uncovered  = np.clip(
            min_left_uncovered - max_right_common_intersection - max_right_unique_intersection, a_min=0, a_max=None) # max(0, U_min^L - I_max^c(c) - I_max^u(c))
        max_uncovered = np.minimum(
            max_left_uncovered, max_right_uncovered) # min(U_max^L, U_max^c)

        min_unique_extras = min_left_unique_extras + min_right_unique_extras # E_min^u(L) + E_min^u(c)
        max_unique_extras = max_left_unique_extras + max_right_unique_extras # E_max^u(L) + E_max^u(c)

        min_common_extras = np.maximum(min_left_common_extras, min_right_common_extras) # max(E_min^c(L), E_min^c(c))
        max_common_extras = np.minimum(
            max_size_mask - neuron_sum - min_left_unique_extras - min_right_unique_extras,
            max_left_common_extras + max_right_common_extras
        ) # min(max_size_mask - N^u - N^c - E_min^u(L) - E_min^u(c), E_max^c(L) + E_max^c(c))
    elif isinstance(label, F.And) and isinstance(label.right, F.Not):
        #neuron_activation =  neuron_unique + neuron_common
        max_unique_intersection = max_left_unique_intersection
        min_unique_intersection = min_left_unique_intersection

        min_common_intersection = np.clip(
            min_left_common_intersection + min_right_uncovered - neuron_coverable, a_min=0, a_max=None
        ) # max(0, I_min^c(L) + U_min^c - N^u - N^c)
        max_common_intersection = np.minimum(
            max_right_uncovered, max_left_common_intersection
        ) # min(U_max^c, I_max^c(L))

        min_uncovered = np.maximum(
            min_left_uncovered, min_right_common_intersection + min_right_unique_intersection
        ) # max(U_min^L, I_min^c(c) + I_min^u(c))
        max_uncovered = neuron_coverable - np.clip(
            max_left_common_intersection + max_right_uncovered - neuron_coverable, a_min=0, a_max=None
        ) # N^u + N^c - max(0, I_max^c(L) + U_max^c - N^u - N^c)

        min_unique_extras = min_left_unique_extras # E_min^u(L)
        max_unique_extras = max_left_unique_extras # E_max^u(L)

        min_common_extras = np.clip(
            min_left_common_extras + (max_size_mask - neuron_sum - max_right_unique_extras - max_right_common_extras) - \
            (max_size_mask - neuron_sum),
            a_min=0, a_max=None
        )

        max_common_extras = max_left_common_extras
        if (np.all(max_unique_extras >= max_left_unique_extras) and \
            np.all(max_common_extras >= max_left_common_extras) and \
            np.all(min_unique_extras >= min_left_unique_extras) and \
            np.all(min_common_extras >= min_left_common_extras) and \
            np.all(max_common_intersection <= max_left_common_intersection) and \
            np.all(min_common_intersection <= min_left_common_intersection)):
            #print("Discarding formula: ", label)
            # If one of the two side does not contribute to the intersection, we can discard this formula
            return None, None, None, None, None

    elif isinstance(label, F.And):
        min_unique_extras = np.zeros_like(min_left_unique_extras)
        max_unique_extras =  np.zeros_like(min_left_unique_extras)

        min_common_extras = np.clip(
            min_left_common_extras + min_right_common_extras - (max_size_mask - neuron_sum - max_left_unique_extras - max_right_unique_extras)
        , a_min=0, a_max=None
        ) # max(0, E^c(L) + E^c(c) - (max_size_mask - N^u - N^c))
        max_common_extras = np.minimum(
            max_left_common_extras,
            max_right_common_extras
        ) # min(E^c(L), E^c(c))
        if (np.all(max_unique_extras >= max_left_unique_extras) and \
        np.all(max_common_extras >= max_left_common_extras) and \
        np.all(min_common_extras >= min_left_common_extras) and \
        np.all(min_unique_extras >= min_left_unique_extras)) or \
        (np.all(max_unique_extras >= max_right_unique_extras) and \
        np.all(max_common_extras >= max_right_common_extras) and \
        np.all(min_common_extras >= min_right_common_extras) and \
        np.all(min_unique_extras >= min_right_unique_extras)):
            # If one of the two side does not contribute to the intersection, we can discard this formula
            return None, None, None, None, None
        
        max_unique_intersection = np.zeros_like(min_left_unique_intersection)
        min_unique_intersection = np.zeros_like(min_left_unique_intersection)
        
        min_common_intersection = np.clip(
            min_left_common_intersection + min_right_common_intersection - neuron_coverable,
            a_min=0, a_max=None
            ) # max(0, I_min^c(L) + I_min^c(c) - N^u - N^c)
        max_common_intersection = np.minimum(
            max_left_common_intersection, max_right_common_intersection) # min(I_max^c(L), I_max^c(c))
        
        min_uncovered = np.maximum(
            min_left_uncovered, min_right_uncovered) # max(U_min^L, U_min^c)
        max_uncovered = neuron_coverable - np.clip(
            min_left_common_intersection + min_right_common_intersection - neuron_coverable, a_min=0, a_max=None
        ) # N^u + N^c - max(0, I_min^c(L) + I_min^c(c) - N^u - N^c)

    else:
        raise ValueError(f"Unknown label type: {type(label)}")

    # assert np.all(np.less_equal(max_common_intersection+max_unique_intersection, neuron_coverable))
    # assert np.all(np.less_equal(min_common_intersection+min_unique_intersection, neuron_coverable))

    return ((max_common_intersection, max_common_intersection.sum()), (min_common_intersection, min_common_intersection.sum())), ((max_unique_intersection, max_unique_intersection.sum()), (min_unique_intersection, min_unique_intersection.sum())), \
        ((max_common_extras, max_common_extras.sum()), (min_common_extras, min_common_extras.sum())), \
        ((max_unique_extras, max_unique_extras.sum()), (min_unique_extras, min_unique_extras.sum())), \
        ((max_uncovered, max_uncovered.sum()), (min_uncovered, min_uncovered.sum()))


def individual_estimation(label, common_intersection, unique_intersection, common_extras, unique_extras, uncovered, neuron_unique_tuple, neuron_common_tuple, neuron_coverable_tuple, neuron_sum_tuple, max_improvement, *, num_hits, max_size_mask, max_length, minimum_threshold=0, debug=False):
    # Unpack max and min quantities
    max_common_intersection_tuple, min_common_intersection_tuple = get_max_min_quantity(common_intersection)
    max_unique_intersection_tuple, min_unique_intersection_tuple = get_max_min_quantity(unique_intersection)
    max_common_extras_tuple, min_common_extras_tuple = get_max_min_quantity(common_extras)
    max_unique_extras_tuple, min_unique_extras_tuple = get_max_min_quantity(unique_extras)
    #max_common_intersection = max_common_intersection_tuple[0]
    max_common_intersection_sum = max_common_intersection_tuple[1]
    #min_common_intersection = min_common_intersection_tuple[0]
    min_common_intersection_sum = min_common_intersection_tuple[1]
    #max_unique_intersection = max_unique_intersection_tuple[0]
    max_unique_intersection_sum = max_unique_intersection_tuple[1]
    #min_unique_intersection = min_unique_intersection_tuple[0]
    min_unique_intersection_sum = min_unique_intersection_tuple[1]
#    max_common_extras = max_common_extras_tuple[0]
    max_common_extras_sum = max_common_extras_tuple[1]
#    min_common_extras = min_common_extras_tuple[0]
    min_common_extras_sum = min_common_extras_tuple[1]
#    max_unique_extras = max_unique_extras_tuple[0]
    max_unique_extras_sum = max_unique_extras_tuple[1]
#    min_unique_extras = min_unique_extras_tuple[0]
    min_unique_extras_sum = min_unique_extras_tuple[1]
#     neuron_unique = neuron_unique_tuple[0]
# #    neuron_unique_sum = neuron_unique_tuple[1]
#     neuron_common = neuron_common_tuple[0]
#     # neuron_common_sum = neuron_common_tuple[1]
    #neuron_coverable = neuron_coverable_tuple[0]
    # neuron_coverable_sum = neuron_coverable_tuple[1]

    #max_label_intersection = max_common_intersection + max_unique_intersection
    max_label_common_intersection_sum =  max_common_intersection_sum + max_unique_intersection_sum
    if max_label_common_intersection_sum == 0:
        return 0.0, 0.0

    # Max IoU
    min_union = num_hits + min_unique_extras_sum + min_common_extras_sum
    max_iou = max_label_common_intersection_sum / min_union

    # Save computation
    if max_iou == 0 or max_iou < minimum_threshold:
        return 0.0, 0.0
    
    # Min IoU
    min_intersection = min_common_intersection_sum + min_unique_intersection_sum
    max_union = num_hits + max_unique_extras_sum + max_common_extras_sum
    min_iou = min_intersection / max_union

    if debug:
        print(f"Max IoU: {max_iou}, Min IoU: {min_iou}")
        #print(f"Max Intersection: {max_intersection.sum()}, Min Union: {min_union.sum()}")
        print(f"Min Intersection: {min_intersection.sum()}, Max Union: {max_union.sum()}")
    return max_iou, min_iou


def or_chain_estimation(label, common_intersection, unique_intersection, common_extras, unique_extras, uncovered, neuron_unique_tuple, neuron_common_tuple, neuron_coverable_tuple, neuron_sum_tuple, max_improvement, *, num_hits, max_size_mask, max_length, minimum_threshold=0, debug=False, operation_type='sum'):
    # Aux variables
    label_len = len(label)
    k = max_length - label_len # Missing length   

    # Unpack improvement information
    improv_common_intersection, improv_unique_intersection, improv_common_extras, improv_unique_extras, improv_uncovered = max_improvement

    # Unpack max and min quantities
    # max_common_intersection, min_common_intersection = get_max_min_quantity(common_intersection)
    # max_unique_intersection, min_unique_intersection = get_max_min_quantity(unique_intersection)
    # max_common_extras, min_common_extras = get_max_min_quantity(common_extras)
    # max_unique_extras, min_unique_extras = get_max_min_quantity(unique_extras)
    # Unpack max and min quantities
    max_common_intersection_tuple, min_common_intersection_tuple = get_max_min_quantity(common_intersection)
    max_unique_intersection_tuple, min_unique_intersection_tuple = get_max_min_quantity(unique_intersection)
    max_common_extras_tuple, min_common_extras_tuple = get_max_min_quantity(common_extras)
    max_unique_extras_tuple, min_unique_extras_tuple = get_max_min_quantity(unique_extras)
    max_common_intersection = max_common_intersection_tuple[0]
    max_common_intersection_sum = max_common_intersection_tuple[1]
    min_common_intersection = min_common_intersection_tuple[0]
    min_common_intersection_sum = min_common_intersection_tuple[1]
    max_unique_intersection = max_unique_intersection_tuple[0]
    max_unique_intersection_sum = max_unique_intersection_tuple[1]
    min_unique_intersection = min_unique_intersection_tuple[0]
    min_unique_intersection_sum = min_unique_intersection_tuple[1]
    max_common_extras = max_common_extras_tuple[0]
    max_common_extras_sum = max_common_extras_tuple[1]
    min_common_extras = min_common_extras_tuple[0]
    min_common_extras_sum = min_common_extras_tuple[1]
    max_unique_extras = max_unique_extras_tuple[0]
    max_unique_extras_sum = max_unique_extras_tuple[1]
    min_unique_extras = min_unique_extras_tuple[0]
    min_unique_extras_sum = min_unique_extras_tuple[1]
    neuron_unique = neuron_unique_tuple[0]
    neuron_unique_sum = neuron_unique_tuple[1]
    neuron_common = neuron_common_tuple[0]
    neuron_common_sum = neuron_common_tuple[1]
    neuron_coverable = neuron_coverable_tuple[0]
    neuron_coverable_sum = neuron_coverable_tuple[1]
    neuron_sum = neuron_sum_tuple[0]
    neuron_sum_sum = neuron_sum_tuple[1]

    # We discard the labels that cannot increase the IoU
    if max_common_intersection_sum + max_unique_intersection_sum == 0:
        return 0.0, 0.0
    
    #max_label_intersection = max_common_intersection + max_unique_intersection
    top_k_intersection_sum = improv_common_intersection[k][TOP_INDEX_SUM] + improv_unique_intersection[k][TOP_INDEX_SUM]
    top_k_extras_sum = improv_common_extras[k][TOP_INDEX_SUM] + improv_unique_extras[k][TOP_INDEX_SUM]
    bottom_1_intersection_sum = improv_common_intersection[0][BOTTOM_INDEX_SUM] + improv_unique_intersection[0][BOTTOM_INDEX_SUM]
    bottom_1_extras_sum = improv_common_extras[0][BOTTOM_INDEX_SUM] + improv_unique_extras[0][BOTTOM_INDEX_SUM]

    tot_size = max_size_mask*len(neuron_coverable)

    # Max IoU
    if operation_type == 'sum':
        min_union = min(num_hits + min_common_extras_sum + min_unique_extras_sum + max(0, bottom_1_extras_sum - max_common_extras_sum - max_unique_extras_sum), tot_size)
        max_intersection = min(max_common_intersection_sum + max_unique_intersection_sum + top_k_intersection_sum, neuron_coverable_sum)
    elif operation_type == 'sample':
        top_k_intersection = improv_common_intersection[k][TOP_INDEX_SAMPLE] + improv_unique_intersection[k][TOP_INDEX_SAMPLE]
        max_intersection = np.minimum(max_common_intersection + max_unique_intersection + top_k_intersection, neuron_coverable).sum()
        max_label_extras = max_common_extras + max_unique_extras
        min_label_extras = min_common_extras + min_unique_extras
        bottom_1_extras = improv_common_extras[0][BOTTOM_INDEX_SAMPLE] + improv_unique_extras[0][BOTTOM_INDEX_SAMPLE]
        
        min_added_extras = np.clip(bottom_1_extras - max_label_extras, a_min=0, a_max=None)
        min_union = np.clip(neuron_sum + min_label_extras + min_added_extras, a_min=0, a_max=max_size_mask).sum()
    elif operation_type == 'hybrid':
        max_intersection = min(max_common_intersection_sum + max_unique_intersection_sum + top_k_intersection_sum, neuron_coverable_sum)
        if max_intersection == neuron_coverable_sum:
            # Search for an alternative lower overestimation
            top_k_intersection = improv_common_intersection[k][TOP_INDEX_SAMPLE] + improv_unique_intersection[k][TOP_INDEX_SAMPLE]
            max_intersection = np.minimum(max_common_intersection + max_unique_intersection + top_k_intersection, neuron_coverable).sum()

        min_union = min(num_hits + min_common_extras_sum + min_unique_extras_sum + max(0, bottom_1_extras_sum - max_common_extras_sum - max_unique_extras_sum), tot_size)
        if min_union == tot_size:
            max_label_extras = max_common_extras + max_unique_extras
            min_label_extras = min_common_extras + min_unique_extras
            bottom_1_extras = improv_common_extras[0][BOTTOM_INDEX_SAMPLE] + improv_unique_extras[0][BOTTOM_INDEX_SAMPLE]

            min_added_extras = np.clip(bottom_1_extras - max_label_extras, a_min=0, a_max=None)
            min_union = np.clip(neuron_sum + min_label_extras + min_added_extras, a_min=0, a_max=max_size_mask).sum()
    max_iou = max_intersection / min_union

    # Save computation
    if max_iou == 0 or max_iou < minimum_threshold:
        return 0.0, 0.0
    
    # Min IoU
    if operation_type == 'sum':
        min_intersection = max(min_common_intersection_sum + min_unique_intersection_sum, bottom_1_intersection_sum)
        max_union = min(num_hits + min_common_extras_sum + min_unique_extras_sum + top_k_extras_sum, tot_size)
    elif operation_type == 'sample':
        bottom_1_intersection = improv_common_intersection[0][BOTTOM_INDEX_SAMPLE] + improv_unique_intersection[0][BOTTOM_INDEX_SAMPLE]
        min_intersection = np.maximum(min_common_intersection + min_unique_intersection, bottom_1_intersection).sum()
        max_label_extras = max_common_extras + max_unique_extras
        top_k_extras = improv_common_extras[k][TOP_INDEX_SAMPLE] + improv_unique_extras[k][TOP_INDEX_SAMPLE]
        max_union = np.clip(neuron_sum + max_label_extras + top_k_extras, a_min=0, a_max=max_size_mask).sum()
    elif operation_type == 'hybrid':
        max_union = min(num_hits + min_common_extras_sum + min_unique_extras_sum + top_k_extras_sum, tot_size)
        if max_union == tot_size:
            # Search for an alternative lower overestimation
            max_label_extras = max_common_extras + max_unique_extras
            top_k_extras = improv_common_extras[k][TOP_INDEX_SAMPLE] + improv_unique_extras[k][TOP_INDEX_SAMPLE]
            max_union = np.clip(neuron_sum + max_label_extras + top_k_extras, a_min=0, a_max=max_size_mask).sum()
    min_iou = min_intersection / max_union
    return max_iou, min_iou


def and_chain_estimation(label, common_intersection, unique_intersection, common_extras, unique_extras, uncovered, neuron_unique_tuple, neuron_common_tuple, neuron_coverable_tuple, neuron_sum_tuple, max_improvement, *, num_hits, max_size_mask, max_length, minimum_threshold=0, debug=False, operation_type='sum'):
    # Aux variables
    # Unpack improvement information
    improv_common_intersection, improv_unique_intersection, improv_common_extras, improv_unique_extras, improv_uncovered = max_improvement

    # Unpack max and min quantities
    max_common_intersection_tuple, min_common_intersection_tuple = get_max_min_quantity(common_intersection)
    max_common_extras_tuple, min_common_extras_tuple = get_max_min_quantity(common_extras)
    max_unique_extras_tuple, min_unique_extras_tuple = get_max_min_quantity(unique_extras)
    max_common_intersection = max_common_intersection_tuple[0]
    max_common_intersection_sum = max_common_intersection_tuple[1]
    min_common_intersection = min_common_intersection_tuple[0]
    max_common_extras_sum = max_common_extras_tuple[1]
    min_common_extras = min_common_extras_tuple[0]
    min_common_extras_sum = min_common_extras_tuple[1]
    min_unique_extras = min_unique_extras_tuple[0]
    min_unique_extras_sum = min_unique_extras_tuple[1]
    neuron_common = neuron_common_tuple[0]
    neuron_coverable = neuron_coverable_tuple[0]
    neuron_sum = neuron_sum_tuple[0]

    # We discard the labels that cannot increase the IoU
    if max_common_intersection_sum == 0:
        return 0.0, 0.0

    top_1_common_intersection = improv_common_intersection[0][TOP_INDEX_SAMPLE]
    bottom_1_common_extras_sum = improv_common_extras[0][BOTTOM_INDEX_SUM]
    bottom_1_common_intersection_sum = improv_common_intersection[0][BOTTOM_INDEX_SUM]
    tot_size = max_size_mask * len(neuron_coverable)

    # MaxIoU
    if operation_type == 'sum':
        max_intersection = max_common_intersection_sum
        min_union = min(num_hits + max(0, min_common_extras_sum  + bottom_1_common_extras_sum - (tot_size - num_hits - min_unique_extras_sum) ), tot_size)
    elif operation_type == 'sample':
        max_intersection = np.minimum(max_common_intersection, top_1_common_intersection).sum()
        bottom_1_common_extras = improv_common_extras[0][BOTTOM_INDEX_SAMPLE] 
        min_union = np.clip(neuron_sum + np.clip(
            min_common_extras +  bottom_1_common_extras - (max_size_mask - neuron_sum -min_unique_extras),
                a_min=0, a_max=max_size_mask), a_min=0, a_max=max_size_mask).sum()
    elif operation_type == 'hybrid':
        max_intersection = max_common_intersection_sum
        min_union = min(num_hits + max(0, min_common_extras_sum  + bottom_1_common_extras_sum - (tot_size - num_hits - min_unique_extras_sum) ), tot_size)
        if min_union == tot_size:
            # Search for an alternative lower overestimation
            bottom_1_common_extras = improv_common_extras[0][BOTTOM_INDEX_SAMPLE] 
            min_union = np.clip(neuron_sum + np.clip(
                min_common_extras + bottom_1_common_extras - (max_size_mask - neuron_sum - min_unique_extras),
                    a_min=0, a_max=max_size_mask), a_min=0, a_max=max_size_mask).sum()

    max_iou = max_intersection / min_union

    # Save computation
    if max_iou == 0 or max_iou < minimum_threshold:
        return 0.0, 0.0
    if bottom_1_common_intersection_sum == 0:
        return max_iou, 0.0

    return max_iou, 0.0

def and_not_chain_estimation(label, common_intersection, unique_intersection, common_extras, unique_extras, uncovered, neuron_unique_tuple, neuron_common_tuple, neuron_coverable_tuple, neuron_sum_tuple, max_improvement, *, num_hits, max_size_mask, max_length, minimum_threshold=0, debug=False, operation_type='sum'):
    # Aux variables
    # Unpack improvement information
    improv_common_intersection, improv_unique_intersection, improv_common_extras, improv_unique_extras, improv_uncovered = max_improvement

    # Unpack max and min quantities
    max_common_intersection_tuple, min_common_intersection_tuple = get_max_min_quantity(common_intersection)
    max_unique_intersection_tuple, min_unique_intersection_tuple = get_max_min_quantity(unique_intersection)
    max_common_extras_tuple, min_common_extras_tuple = get_max_min_quantity(common_extras)
    max_unique_extras_tuple, min_unique_extras_tuple = get_max_min_quantity(unique_extras)
    max_common_intersection = max_common_intersection_tuple[0]
    max_common_intersection_sum = max_common_intersection_tuple[1]
    min_common_intersection = min_common_intersection_tuple[0]
    min_common_intersection_sum = min_common_intersection_tuple[1]
    max_unique_intersection = max_unique_intersection_tuple[0]
    max_unique_intersection_sum = max_unique_intersection_tuple[1]
    min_unique_intersection = min_unique_intersection_tuple[0]
    min_unique_intersection_sum = min_unique_intersection_tuple[1]
    max_common_extras = max_common_extras_tuple[0]
    max_common_extras_sum = max_common_extras_tuple[1]
    min_common_extras = min_common_extras_tuple[0]
    min_common_extras_sum = min_common_extras_tuple[1]
    max_unique_extras = max_unique_extras_tuple[0]
    max_unique_extras_sum = max_unique_extras_tuple[1]
    min_unique_extras = min_unique_extras_tuple[0]
    min_unique_extras_sum = min_unique_extras_tuple[1]
    neuron_unique = neuron_unique_tuple[0]
    neuron_unique_sum = neuron_unique_tuple[1]
    neuron_common = neuron_common_tuple[0]
    neuron_common_sum = neuron_common_tuple[1]
    neuron_coverable = neuron_coverable_tuple[0]
    neuron_coverable_sum = neuron_coverable_tuple[1]
    neuron_sum = neuron_sum_tuple[0]
    neuron_sum_sum = neuron_sum_tuple[1]


    # We discard the labels that cannot increase the IoU
    if max_common_intersection_sum + max_unique_intersection_sum == 0:
        return 0.0, 0.0
    
    top_1_uncovered = improv_uncovered[0][TOP_INDEX_SAMPLE]
    
    # Max IoU
    if operation_type == 'sum' or operation_type == 'hybrid':
        max_intersection = max_unique_intersection_sum + max_common_intersection_sum
    elif operation_type == 'sample':
        max_intersection = np.minimum(max_unique_intersection + np.minimum(
            max_common_intersection, top_1_uncovered
        ), neuron_coverable).sum()
    min_union = num_hits + min_unique_extras_sum
    max_iou = max_intersection / min_union

    # Save computation
    if max_iou == 0 or max_iou < minimum_threshold:
        return 0.0, 0.0
    
    # Min IoU
    min_intersection = min_unique_intersection_sum
    max_union = num_hits + max_unique_extras_sum + max_common_extras_sum
    min_iou = min_intersection / max_union

    if debug:
        print(f"Max IoU: {max_iou}, Min IoU: {min_iou}")
        print(f"Max Intersection: {max_intersection.sum()}, Min Union: {min_union.sum()}")
        print(f"Min Intersection: {min_intersection.sum()}, Max Union: {max_union.sum()}")
    
    return max_iou, min_iou

def comb_and_or_chain_estimation(label, common_intersection, unique_intersection, common_extras, unique_extras, uncovered, neuron_unique_tuple, neuron_common_tuple, neuron_coverable_tuple, neuron_sum_tuple, max_improvement, *, num_hits, max_size_mask, max_length, minimum_threshold=0, debug=False, operation_type='sum'):
    # Aux variables
    label_len = len(label)
    k = max_length - label_len # Missing length   
    # Unpack improvement information
    improv_common_intersection, improv_unique_intersection, improv_common_extras, improv_unique_extras, improv_uncovered = max_improvement

    # Unpack max and min quantities
    max_common_intersection_tuple, min_common_intersection_tuple = get_max_min_quantity(common_intersection)
    max_unique_intersection_tuple, min_unique_intersection_tuple = get_max_min_quantity(unique_intersection)
    max_common_extras_tuple, min_common_extras_tuple = get_max_min_quantity(common_extras)
    max_unique_extras_tuple, min_unique_extras_tuple = get_max_min_quantity(unique_extras)
    max_common_intersection = max_common_intersection_tuple[0]
    max_common_intersection_sum = max_common_intersection_tuple[1]
    min_common_intersection = min_common_intersection_tuple[0]
    min_common_intersection_sum = min_common_intersection_tuple[1]
    max_unique_intersection = max_unique_intersection_tuple[0]
    max_unique_intersection_sum = max_unique_intersection_tuple[1]
    min_unique_intersection = min_unique_intersection_tuple[0]
    min_unique_intersection_sum = min_unique_intersection_tuple[1]
    max_common_extras = max_common_extras_tuple[0]
    max_common_extras_sum = max_common_extras_tuple[1]
    min_common_extras = min_common_extras_tuple[0]
    min_common_extras_sum = min_common_extras_tuple[1]
    max_unique_extras = max_unique_extras_tuple[0]
    max_unique_extras_sum = max_unique_extras_tuple[1]
    min_unique_extras = min_unique_extras_tuple[0]
    min_unique_extras_sum = min_unique_extras_tuple[1]
    neuron_unique = neuron_unique_tuple[0]
    neuron_unique_sum = neuron_unique_tuple[1]
    neuron_common = neuron_common_tuple[0]
    neuron_common_sum = neuron_common_tuple[1]
    neuron_coverable = neuron_coverable_tuple[0]
    neuron_coverable_sum = neuron_coverable_tuple[1]
    neuron_sum = neuron_sum_tuple[0]
    neuron_sum_sum = neuron_sum_tuple[1]

    #zero_vector = np.zeros_like(max_common_intersection)
    top_k_common_intersection = improv_common_intersection[k][TOP_INDEX_SAMPLE]
    top_k_common_intersection_sum = improv_common_intersection[k][TOP_INDEX_SUM]
    top_k_unique_intersection = improv_unique_intersection[k][TOP_INDEX_SAMPLE]
    top_k_unique_intersection_sum = improv_unique_intersection[k][TOP_INDEX_SUM]
    top_1_common_extras = improv_common_extras[0][TOP_INDEX_SAMPLE]

    bottom_1_common_extras = improv_common_extras[0][BOTTOM_INDEX_SAMPLE]
    bottom_1_common_intersection = improv_common_intersection[0][BOTTOM_INDEX_SAMPLE]
    bottom_1_common_intersection_sum = improv_common_intersection[0][BOTTOM_INDEX_SUM]

    if max_common_intersection_sum == 0:
        return 0.0, 0.0 
    
    # Max IoU
    if operation_type == 'sum':
        max_intersection = min(max_common_intersection_sum + top_k_common_intersection_sum + top_k_unique_intersection_sum, neuron_coverable_sum - min_unique_intersection_sum)
        min_union = num_hits
    elif operation_type == 'sample':
        max_label_intersection = max_common_intersection + top_k_common_intersection + top_k_unique_intersection
        max_intersection = np.minimum(max_label_intersection, neuron_coverable - min_unique_intersection).sum()
        min_union = np.clip(neuron_sum + np.clip(
                                    min_common_extras + bottom_1_common_extras -
                                        (max_size_mask - neuron_sum - min_unique_extras), a_min =0, a_max=max_size_mask),
                        a_min=0, a_max=max_size_mask).sum()
    elif operation_type == 'hybrid':
        max_intersection = min(max_common_intersection_sum + top_k_common_intersection_sum + top_k_unique_intersection_sum, neuron_coverable_sum - min_unique_intersection_sum)
        min_union = num_hits
        if max_intersection == neuron_coverable_sum - min_unique_intersection_sum:
            # Search for an alternative lower overestimation
            max_label_intersection = max_common_intersection + top_k_common_intersection + top_k_unique_intersection
            max_intersection = np.minimum(max_label_intersection, neuron_coverable - min_unique_intersection).sum()
        
    # if max_intersection == neuron_coverable_sum - min_unique_intersection_sum:
    max_iou = max_intersection / min_union

    # Save computation
    if max_iou == 0 or max_iou < minimum_threshold:
        return 0.0, 0.0
    if bottom_1_common_intersection_sum == 0:
        return max_iou, 0.0

    return max_iou, 0.0


def comb_or_andnot_chain_estimation(label, common_intersection, unique_intersection, common_extras, unique_extras, uncovered, neuron_unique_tuple, neuron_common_tuple, neuron_coverable_tuple, neuron_sum_tuple, max_improvement, *, num_hits, max_size_mask, max_length, minimum_threshold=0, debug=False, operation_type='sum'):
    # Aux variables
    label_len = len(label)
    k = max_length - label_len # Missing length   
    # Unpack improvement information
    improv_common_intersection, improv_unique_intersection, improv_common_extras, improv_unique_extras, improv_uncovered = max_improvement

    # Unpack max and min quantities
    max_common_intersection_tuple, min_common_intersection_tuple = get_max_min_quantity(common_intersection)
    max_unique_intersection_tuple, min_unique_intersection_tuple = get_max_min_quantity(unique_intersection)
    max_common_extras_tuple, min_common_extras_tuple = get_max_min_quantity(common_extras)
    max_unique_extras_tuple, min_unique_extras_tuple = get_max_min_quantity(unique_extras)
    max_common_intersection = max_common_intersection_tuple[0]
    max_common_intersection_sum = max_common_intersection_tuple[1]
    # min_common_intersection = min_common_intersection_tuple[0]
    # min_common_intersection_sum = min_common_intersection_tuple[1]
    max_unique_intersection = max_unique_intersection_tuple[0]
    max_unique_intersection_sum = max_unique_intersection_tuple[1]
    #min_unique_intersection = min_unique_intersection_tuple[0]
    min_unique_intersection_sum = min_unique_intersection_tuple[1]
    max_common_extras = max_common_extras_tuple[0]
    max_common_extras_sum = max_common_extras_tuple[1]
    #min_common_extras = min_common_extras_tuple[0]
    #min_common_extras_sum = min_common_extras_tuple[1]
    max_unique_extras = max_unique_extras_tuple[0]
    max_unique_extras_sum = max_unique_extras_tuple[1]
    min_unique_extras = min_unique_extras_tuple[0]
    min_unique_extras_sum = min_unique_extras_tuple[1]
    neuron_unique = neuron_unique_tuple[0]
    #neuron_unique_sum = neuron_unique_tuple[1]
    neuron_common = neuron_common_tuple[0]
    #neuron_common_sum = neuron_common_tuple[1]
    neuron_coverable = neuron_coverable_tuple[0]
    neuron_coverable_sum = neuron_coverable_tuple[1]
    neuron_sum = neuron_sum_tuple[0]
    neuron_sum_sum = neuron_sum_tuple[1]

    # We discard the labels that cannot increase the IoU
    if max_common_intersection_sum + max_unique_intersection_sum == 0:
        return 0.0, 0.0

    top_k_common_intersection = improv_common_intersection[k][TOP_INDEX_SAMPLE]
    top_k_unique_intersection = improv_unique_intersection[k][TOP_INDEX_SAMPLE]
    top_k_common_intersection_sum = improv_common_intersection[k][TOP_INDEX_SUM]
    top_k_unique_intersection_sum = improv_unique_intersection[k][TOP_INDEX_SUM]
    bott_1_unique_intersection = improv_unique_intersection[0][BOTTOM_INDEX_SAMPLE]
    bott_1_unique_extras = improv_unique_extras[0][BOTTOM_INDEX_SAMPLE]
    top_k_minus_1_extras = improv_unique_extras[k-1][TOP_INDEX_SAMPLE] + improv_common_extras[k-1][TOP_INDEX_SAMPLE]
    top_k_minus_1_extras_sum = improv_unique_extras[k-1][TOP_INDEX_SUM] + improv_common_extras[k-1][TOP_INDEX_SUM]
    tot_size = max_size_mask * len(neuron_coverable)
    # Max IoU
    if operation_type == 'sum':
        max_intersection = min(max_common_intersection_sum + max_unique_intersection_sum + top_k_common_intersection_sum + top_k_unique_intersection_sum, neuron_coverable_sum)
    elif operation_type == 'sample':
        max_label_intersection = max_common_intersection + max_unique_intersection
        max_intersection = np.minimum(
            max_label_intersection + top_k_common_intersection + top_k_unique_intersection,
            neuron_coverable).sum()
    elif operation_type == 'hybrid':
        max_intersection = min(max_common_intersection_sum + max_unique_intersection_sum + top_k_common_intersection_sum + top_k_unique_intersection_sum, neuron_coverable_sum)
        if max_intersection == neuron_coverable_sum:
            # Search for an alternative lower overestimation
            max_label_intersection = max_common_intersection + max_unique_intersection
            max_intersection = np.minimum(
                max_label_intersection + top_k_common_intersection + top_k_unique_intersection,
                neuron_coverable).sum()
    min_union = num_hits + min_unique_extras_sum
    max_iou = max_intersection / min_union
    
    # Save computation
    if max_iou == 0 or max_iou < minimum_threshold:
        return 0.0, 0.0
   
    # Min IoU
    min_intersection = min_unique_intersection_sum
    if min_intersection == 0:
        return max_iou, 0.0
    max_union = min(num_hits + max_unique_extras_sum + max_common_extras_sum + top_k_minus_1_extras_sum, tot_size)
    if operation_type == 'sample' or (max_union == tot_size and operation_type == 'hybrid'):
        top_k_minus_1_extras = improv_unique_extras[k-1][TOP_INDEX_SAMPLE] + improv_common_extras[k-1][TOP_INDEX_SAMPLE]
        max_union = np.clip(neuron_sum + max_unique_extras + max_common_extras + top_k_minus_1_extras, a_min=0, a_max=max_size_mask).sum()
    min_iou = min_intersection / max_union    

    if debug:
        print(f"Max IoU: {max_iou}, Min IoU: {min_iou}")
        print(f"Max Intersection: {max_intersection.sum()}, Min Union: {min_union.sum()}")
        print(f"Min Intersection: {min_intersection.sum()}, Max Union: {max_union.sum()}")
    return max_iou, min_iou

def compute_ratio(label_mask, bitmaps, heuristic_info, debug=False):
    seg_quantities, neuron_quantities, concepts_quantities = heuristic_info
    neuron_unique, neuron_common, (neuron_coverable_mask, _), neuron_coverable, neuron_sum, common_space_extras, unique_space_extras = neuron_quantities
    common_elements, unique_elements, uncoverable_elements = seg_quantities
    unique_elements = unique_elements.to(bitmaps.device)
    common_elements = common_elements.to(bitmaps.device)
    uncoverable_elements = uncoverable_elements.to(bitmaps.device)
    concept_quantities = heuristic_utils.compute_quantities_vector(label_mask, bitmaps, common_elements, unique_elements, neuron_coverable_mask)
    c_common_intersection, c_unique_intersection, c_common_extras, c_unique_extras, _, _ = concept_quantities
    label_iou = (c_common_intersection.sum() + c_unique_intersection.sum()) / (bitmaps.sum() + c_common_extras.sum() + c_unique_extras.sum())
    if debug:
        print(f"Label IoU: {label_iou.item()}")
        print(f"Common Intersection: {c_common_intersection.sum().item()}")
        print(f"Unique Intersection: {c_unique_intersection.sum().item()}")
        print(f"Common Extras: {c_common_extras.sum().item()}")
        print(f"Unique Extras: {c_unique_extras.sum().item()}")
        print(f"Total Elements in Bitmaps: {bitmaps.sum().item()}")
    return  (c_unique_intersection.sum() / c_unique_extras.sum()).item()



def check_problematic_nodes(frontier):
    ordered_frontier = sorted(frontier, key=lambda x: x[0])
    limit_number_recents = 100
    recent_iou = 0
    recent_nodes = []
    for node in ordered_frontier:
        label_node = node[2]
        # if len(label_node) > 1 and 'NOT' in label_node.get_ops():
        #     # If the label is a NOT operation, we skip it
        #     continue
        if node[0] == recent_iou:
            # If the IoU is the same, we have a duplicate
            recent_nodes.append(node)
        else:
            recent_iou = node[0]
            recent_nodes = [node]
        if len(recent_nodes) > limit_number_recents:
            return recent_nodes
    return None

def explore_state_space(node, heuristic_name, candidate_concepts, heuristic_info, disjoint_info, minimum_info, label_mapping, max_improvement, masks, bitmaps, num_hits, max_size_mask, length=3, phase='unknown'):
    disjoint_frontier, overlap_frontier = expand_node(node, candidate_labels=candidate_concepts, max_length=length, disjoint_info=disjoint_info)
    if phase == 'disjoint':
        next_frontier = disjoint_frontier
    elif phase == 'overlap':
        next_frontier = overlap_frontier
    else:
        raise ValueError(f"Unknown phase {phase}. Expected 'disjoint' or 'overlap'.")
    # Compute the estimation for the next frontier
    current_frontier, minimum_info, heuristic_info, label_mapping = update_frontier(
        past_frontier=current_frontier, new_nodes=next_frontier, 
        label_mapping=label_mapping, heuristic=heuristic_name, 
        heuristic_info=heuristic_info, max_improvement=max_improvement,
        num_hits=num_hits, max_size_mask=max_size_mask,
        length=length, minimum_info=minimum_info, disjoint_info=disjoint_info
    )
    return current_frontier, minimum_info, heuristic_info, label_mapping

def explore_depth_first_search_by_minimum(node, heuristic_name, candidate_concepts, heuristic_info, disjoint_info, minimum_info, label_mapping, max_improvement, masks, bitmaps, num_hits, max_size_mask, length=3, phase='unknown'):
    minimum_threshold, minimum_node, next_op_minimum = minimum_info
    print(f"Exploring node {node} with minimum threshold {minimum_threshold} and next operation {next_op_minimum}")
    while next_op_minimum is not None and next_op_minimum != 'INDIVIDUAL':
        current_minimum_threshold, _current_minimum_node, next_op_minimum = minimum_info
        print(f"Current minimum threshold: {current_minimum_threshold} with node {_current_minimum_node} and next operation {next_op_minimum}")
        _, minimum_info, heuristic_info, label_mapping = explore_state_space(
            node, heuristic_name, candidate_concepts, heuristic_info, disjoint_info,
            minimum_info, label_mapping, max_improvement, masks, bitmaps,
            num_hits, max_size_mask=max_size_mask, length=length, phase=phase
        )
        new_minimum_threshold, new_minimum_node, next_op_minimum = minimum_info
        print(f"New minimum threshold: {new_minimum_threshold} with node {new_minimum_node} and next operation {next_op_minimum}")
        assert new_minimum_threshold >= current_minimum_threshold, \
            f"New minimum threshold {new_minimum_threshold} is lower than the current minimum threshold {current_minimum_threshold}."
    return minimum_info
