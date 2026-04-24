from collections import Counter
import heapq
import queue as Q

from . import formula as F
from . import optimal_heuristic
from . import utils
from . import heuristic_utils
from compositional import mask_utils, metrics

def analyze_final_node(label_node, masks, bitmaps):
    # This is the case where the formula has no operations to expand
    # Compute Label Mask
    label_mask = utils.sparse_to_torch(mask_utils.get_formula_mask(
            label_node, masks
        )).to(bitmaps.device)
    # Compute the exact IoU for the label node
    iou = metrics.iou(label_mask, bitmaps).item()
    return iou


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
    for candidate_formula in frontier:
        node = (None, 'INDIVIDUAL' , candidate_formula, [])
        max_score, min_score = optimal_heuristic.estimate_label_iou(heuristic,
            node, label_mapping, heuristic_info,  max_improvement=max_improvement, disjoint_info=disjoint_info, num_hits=num_hits, max_size_mask=max_size_mask,
            max_length=length, minimum_threshold=0)
        # Add nodes and their estimation to the current frontier
        label = node[2]
        for node_path in max_score:
            node_path_max_iou, node_path_next_op, _ , node_path_paths_to_expand = node_path
            if node_path_max_iou > 0:
                # Add the path to the frontier estimates if the IoU is greater than 0.
                # Note: filtering by minimum_threshold is done inside the estimate_label_iou function
                frontier_estimates.append((-node_path_max_iou, node_path_next_op, label, node_path_paths_to_expand))
    return frontier_estimates, global_min_threshold

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

def compute_beam_quantities(beam, masks, beam_masks, heuristic_info, bitmaps):
    seg_quantities, neuron_quantities, concepts_quantities = heuristic_info
    (neuron_unique, _), (neuron_common, _),  neuron_coverable, neuron_sum, common_space_extras, _ = neuron_quantities
    common_elements, unique_elements, uncoverable_elements = seg_quantities
    unique_elements = unique_elements.to(bitmaps.device)
    common_elements = common_elements.to(bitmaps.device)
    uncoverable_elements = uncoverable_elements.to(bitmaps.device)
    beam_info = {}
    for node in beam:
        _, _, label, _ = node
        if label in beam_masks or (isinstance(label, F.Leaf)):
            continue
        label_mask = utils.sparse_to_torch(mask_utils.get_formula_mask(
                    label, masks, beam_masks
                )).to(bitmaps.device)
        concept_quantities = heuristic_utils.compute_quantities_vector(label_mask, bitmaps, common_elements, unique_elements, neuron_common, neuron_unique)
        concept_info = heuristic_utils.get_concept_info(concept_quantities)
        beam_info[label] = (concept_info, None)
        label_mask = label_mask.cpu()
        beam_masks[label] = label_mask
    return beam_info, beam_masks

def get_beam_info(beam, masks, beam_masks, heuristic_info, label_mapping, bitmaps, length):
    """Compute the heuristic info for the beam.

    Args:
        heuristic (str): The heuristic to use.
        beam (dict): A dictionary of formulas of the current beam.
        masks (dict): A dictionary of concept masks. Each mask is a tensor of
            shape (N, H, W).
        bitmaps (torch.Tensor): A tensor of shape (N, H, W) where N is the
            number of sample.
        mask_shape (tuple): The shape of the mask.
        device (torch.device): The device to use.

    Returns:
        beam_masks (dict): A dictionary of labal masks of the formulas in the
        current beam. Each mask is a tensor of shape (N, H, W).
        updated_info (dict): A dictionary of heuristic info.
    """

    # update infos

    beam_info, beam_masks = compute_beam_quantities(beam, masks, beam_masks, heuristic_info, bitmaps)
    new_heuristic_info, new_label_mapping = update_heuristic_info(beam_info, heuristic_info, label_mapping, max_length=length)
    return beam_masks, (new_heuristic_info, new_label_mapping)

def merge_frontiers(frontier_1, frontier_2):
    # Merge the new nodes with the past frontier
    if frontier_1 is not None and len(frontier_1) > 0 and frontier_2 is not None and len(frontier_2) > 0:
        # Merge assumes that the input iterables are sorted
        for new_node in frontier_2:
            heapq.heappush(frontier_1, new_node)
        sorted_frontier = frontier_1
        return sorted_frontier
    elif frontier_1 is not None and len(frontier_1) > 0:
        return frontier_1
    else:
        return frontier_2

def update_frontier(past_frontier, new_nodes, label_mapping, heuristic, heuristic_info, max_improvement, disjoint_info, num_hits, max_size_mask, length, global_min_threshold):

    # Estimate the IoU of the new frontier
    # Min Threshold is set to 0 because beam needs more than 1 top node
    new_frontier, _ = estimate_iou_frontier(
        frontier=new_nodes, label_mapping=label_mapping,
        heuristic=heuristic, heuristic_info=heuristic_info,
        max_improvement=max_improvement,
        num_hits=num_hits, max_size_mask=max_size_mask,
        length=length, global_min_threshold=global_min_threshold,
        disjoint_info=disjoint_info
    )
    
    heapq.heapify(new_frontier)
    sorted_frontier = merge_frontiers(past_frontier, new_frontier)
    return sorted_frontier, global_min_threshold