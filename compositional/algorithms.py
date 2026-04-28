"""
This module contains the implementation of the algorithms described in
the paper. It includes NetDissect, the compositional explanations algorithm
and the heuristic search.
"""
from collections import Counter
import torch

from compositional import augmented_formula as AF
from . import heuristic_search
from . import optimal_search
from . import beam_search
from . import utils
from . import metrics
from . import heuristic_utils
from . import mask_utils


def get_netdissect_scores(bitmaps, masks):
    """Compute the NetDissect score for each concept in the candidate_concepts
    list for the given bitmaps.

    Args:
        bitmaps (torch.Tensor): A tensor of shape (N, H, W) where N is the
            number of sample.
        masks (dict): A dictionary of concept masks. Each mask is a tensor of
            shape (H, W).
        candidate_concepts (list): A list of candidate concepts.

    Returns:
        netdissect_rank (dict): A dictionary of concept scores. Each score is
            a float.
    """
    # hits_unit = torch.count_nonzero(bitmaps)
    netdissect_rank = {}
    candidate_concepts = range(len(masks))
    for concept in candidate_concepts:
        concept_mask = mask_utils.parse_mask_by_type(masks[concept])
        concept_mask = concept_mask.to(bitmaps.device)
        concept_iou = metrics.iou(concept_mask, bitmaps)
        netdissect_rank[concept] = concept_iou

    return netdissect_rank

def get_thresholded_netdissect_scores(bitmaps, masks, threshold):
    """Compute the NetDissect score for each concept in the candidate_concepts
    list for the given bitmaps.

    Args:
        bitmaps (torch.Tensor): A tensor of shape (N, H, W) where N is the
            number of sample.
        masks (dict): A dictionary of concept masks. Each mask is a tensor of
            shape (H, W).
        threshold (float): The threshold to use for the diff between concept_iou and adv_iou to consider a concept as valid.

    Returns:
        netdissect_rank (dict): A dictionary of concept scores. Each score is
            a float.
    """
    netdissect_rank = {}
    candidate_concepts = range(len(masks))
    adv_bitmaps = ~bitmaps
    for concept in candidate_concepts:
        concept_mask = mask_utils.parse_mask_by_type(masks[concept])
        concept_mask = concept_mask.to(bitmaps.device)
        concept_iou = metrics.iou(concept_mask, bitmaps)
        adv_iou = metrics.iou(concept_mask, adv_bitmaps)
        diff = metrics.diff_ratio_iou_adv_iou(concept_iou, adv_iou)
        # Zero out the score of concepts that are not significantly better than the complementary
        if diff < threshold:
            concept_iou = 0.0
        
        netdissect_rank[concept] = concept_iou

    return netdissect_rank

def get_counter_type(concept_iou, adv_iou, tau=0.00):
    threshold_interpretable = 0.04
    threshold_diff = 0.33
    diff = max(concept_iou - adv_iou, 0)
    # normalize diff in a way that if vanilla_iou and adv_iou are small and their difference is small is comparable to a bigger difference but with vanilla_iou and adv_iou larger. Basically, we want to express the difference in percentage and independently from the size of iou
    diff = diff / (concept_iou + 1e-5)
    if concept_iou < threshold_interpretable:
        # Non-interpretable concepts
        return torch.tensor(0.0), "non_interpretable"
    elif concept_iou < adv_iou:
        # Non-significant alignment (the complementary is better)
        return torch.tensor(0.0), "counter_more_significant"
    elif concept_iou - adv_iou < tau:
        # The alignment is not better than the complementary by a margin of tau
        # edge cases where the concept is barely above the threshold and the complementary is barely below the threshold, we want to consider them as non-significant
        return torch.tensor(0.0) , "counter_margin"   
    elif diff < threshold_diff:
        # Difference 
         return torch.tensor(0.0), "counter_diff"
    elif adv_iou > 0.04:
        # Complementary alignment is significant 
        return torch.tensor(0.0), "counter_complementary_significant"

    return concept_iou, None


def get_counter_weighted_netdissect_scores(bitmaps, masks, tau=0.01):
    """Compute the NetDissect score for each concept in the candidate_concepts
    list for the given bitmaps.

    Args:
        bitmaps (torch.Tensor): A tensor of shape (N, H, W) where N is the
            number of sample.
        masks (dict): A dictionary of concept masks. Each mask is a tensor of
            shape (H, W).
        candidate_concepts (list): A list of candidate concepts.

    Returns:
        netdissect_rank (dict): A dictionary of concept scores. Each score is
            a float.
    """
    # hits_unit = torch.count_nonzero(bitmaps)
    netdissect_rank = {}
    candidate_concepts = range(len(masks))
    adv_bitmaps = ~bitmaps
    counter_type_counter = {"fine":0}
    for concept in candidate_concepts:
        concept_mask = mask_utils.parse_mask_by_type(masks[concept])
        concept_mask = concept_mask.to(bitmaps.device)
        concept_iou = metrics.iou(concept_mask, bitmaps)
        adv_iou = metrics.iou(concept_mask, adv_bitmaps)

        _, counter_type = get_counter_type(concept_iou, adv_iou, tau=tau)
        if counter_type is not None:
            if counter_type not in counter_type_counter:
                counter_type_counter[counter_type] = 0
            counter_type_counter[counter_type] += 1
        else:
            counter_type_counter["fine"] += 1
        netdissect_rank[concept] = concept_iou
    return netdissect_rank, counter_type_counter

def get_weighted_netdissect_scores(bitmaps, masks, weights):
    """Compute the NetDissect score for each concept in the candidate_concepts
    list for the given bitmaps.

    Args:
        bitmaps (torch.Tensor): A tensor of shape (N, H, W) where N is the
            number of sample.
        masks (dict): A dictionary of concept masks. Each mask is a tensor of
            shape (H, W).
        candidate_concepts (list): A list of candidate concepts.

    Returns:
        netdissect_rank (dict): A dictionary of concept scores. Each score is
            a float.
    """
    # hits_unit = torch.count_nonzero(bitmaps)
    netdissect_rank = {}
    candidate_concepts = range(len(masks))
    for concept in candidate_concepts:
        concept_mask = mask_utils.parse_mask_by_type(masks[concept])
        concept_mask = concept_mask.to(bitmaps.device)
        concept_iou = metrics.weighted_iou(concept_mask, bitmaps, weights)
        netdissect_rank[concept] = concept_iou

    return netdissect_rank


def get_augmented_netdissect_scores(bitmaps, masks):
    """Compute the NetDissect score for each concept in the candidate_concepts
    list for the given bitmaps.

    Args:
        bitmaps (torch.Tensor): A tensor of shape (N, H, W) where N is the
            number of sample.
        masks (dict): A dictionary of concept masks. Each mask is a tensor of
            shape (H, W).
        candidate_concepts (list): A list of candidate concepts.

    Returns:
        netdissect_rank (dict): A dictionary of concept scores. Each score is
            a float.
    """
    netdissect_rank = {}
    areas = []
    candidate_concepts = range(len(masks))
    for concept in candidate_concepts:
        concept_mask = mask_utils.parse_mask_by_type(masks[concept])
        concept_mask = concept_mask.to(bitmaps.device)
        concept_iou = metrics.iou(concept_mask, bitmaps)
        intersection_area = (concept_mask & bitmaps).sum(
            dim=1, dtype=torch.int32
        )
        netdissect_rank[concept] = concept_iou
        areas.append(intersection_area)
    return netdissect_rank, areas

def get_structured_netdissect_scores(bitmaps, masks, masks_quantities, neuron_quantities):
    """Compute the NetDissect score for each concept in the candidate_concepts
    list for the given bitmaps.

    Args:
        bitmaps (torch.Tensor): A tensor of shape (N, H, W) where N is the
            number of sample.
        masks (dict): A dictionary of concept masks. Each mask is a tensor of
            shape (H, W).
        candidate_concepts (list): A list of candidate concepts.

    Returns:
        netdissect_rank (dict): A dictionary of concept scores. Each score is
            a float.
    """
    
    candidate_concepts = range(len(masks))

    common_elements, unique_elements, uncoverable_elements = masks_quantities
    num_concepts = len(masks)

    (bitmaps_unique, _), (bitmaps_common, _),  _, _, _, _ = neuron_quantities
    unique_elements = unique_elements.to(bitmaps.device)
    common_elements = common_elements.to(bitmaps.device)

    concepts_quantities = [0] * num_concepts 

    for concept in candidate_concepts:
        concept_mask = mask_utils.parse_mask_by_type(masks[concept])
        concept_quantities =  heuristic_utils.compute_quantities_vector(concept_mask, bitmaps, common_elements, unique_elements, bitmaps_common, bitmaps_unique)
        concept_info = heuristic_utils.get_concept_info(
            concept_quantities
        )
        concepts_quantities[concept] = concept_info

       
    return concepts_quantities



def get_neuron_quantities(bitmaps, common_elements, unique_elements, uncoverable_elements):
    common_elements = common_elements.to(bitmaps.device)
    unique_elements = unique_elements.to(bitmaps.device)
    uncoverable_elements = uncoverable_elements.to(bitmaps.device)

    if len(bitmaps.shape) == 2:
        # Sample
        bitmaps_unique = (bitmaps & unique_elements).sum(dim=1)
        bitmaps_common = (bitmaps & common_elements).sum(dim=1)
        bitmaps_coverable = bitmaps & (~uncoverable_elements)
        bitmaps_sum = bitmaps.sum(dim=1)
        bitmaps_coverable_sum = bitmaps_coverable.sum(dim=1)
        space_extras = ~bitmaps
        common_space_extras = (space_extras & common_elements).sum(dim=1)
        unique_space_extras = (space_extras & unique_elements).sum(dim=1)
    elif len(bitmaps.shape) == 1:
        bitmaps_unique = (bitmaps & unique_elements)
        bitmaps_common = (bitmaps & common_elements)
        bitmaps_coverable = bitmaps & (~uncoverable_elements)
        bitmaps_sum = bitmaps
        bitmaps_coverable_sum = bitmaps_coverable
        space_extras = ~bitmaps
        common_space_extras = (space_extras & common_elements)
        unique_space_extras = (space_extras & unique_elements)  
    else:
        raise ValueError("Bitmaps must be of shape (N, H, W) or (N,)")

    # Sum
    bitmaps_unique_sum = bitmaps_unique.sum().item()
    bitmaps_common_sum = bitmaps_common.sum().item()
    num_hits = bitmaps_sum.sum().item()
    coverable_hits = bitmaps_coverable_sum.sum().item()
    common_extras_sum = common_space_extras.sum().item()
    unique_extras_sum = unique_space_extras.sum().item()

    # Move to CPU and convert to numpy
    bitmaps_unique = bitmaps_unique.cpu().numpy()
    bitmaps_common = bitmaps_common.cpu().numpy()
    bitmaps_sum = bitmaps_sum.cpu().numpy()
    bitmaps_coverable_sum = bitmaps_coverable_sum.cpu().numpy()
    common_space_extras = common_space_extras.cpu().numpy()
    unique_space_extras = unique_space_extras.cpu().numpy()
    bitmaps_coverable = bitmaps_coverable.cpu()
    return (bitmaps_unique, bitmaps_unique_sum), (bitmaps_common, bitmaps_common_sum), (bitmaps_coverable_sum, coverable_hits), (bitmaps_sum, num_hits), (common_space_extras, common_extras_sum), (unique_space_extras, unique_extras_sum)

def get_optimal_heuristic_info(masks, bitmaps, seg_quantities):
    common_elements, unique_elements, uncoverable_elements = seg_quantities
    neuron_quantities = get_neuron_quantities(
        bitmaps, common_elements, unique_elements, uncoverable_elements
    )
    concepts_quantities = get_structured_netdissect_scores(
        bitmaps, masks, seg_quantities, neuron_quantities
    )    
    return seg_quantities, neuron_quantities, concepts_quantities

def get_heuristic_scores(
    segmentations,
    activation_masks,
    *,
    heuristic="mmesh",
    segmentations_info=None,
    disjoint_info=None,
    max_size_mask,
    beam_size=5,
    length=3,
    mask_shape=None,
    device=torch.device("cpu"),
    concept_quantities=None,
    neuron_quantities=None,
    labels=None,
    beam_variant=None,
    constraints=None,
    counter_variant=False,
    diff_threshold=0.1,
    block_type_3=True,
):
    """Compute the heuristic score for each concept in the candidate_concepts
    list for the given bitmaps.

    Args:
        segmentations (dict): A dictionary of concept masks. Each mask is a
            tensor of shape (N, H, W) where N is
            the number of sample.
        activation_masks (torch.Tensor): A tensor of shape (N, H, W) where N is
            the number of sample.
        heuristic (str): The heuristic to use for the search. Can be one of
            "mmesh", "cfh", "areas", "none".
        segmentations_info (dict): A dictionary of information about the
            segmentations. None can be used only when the heuristic is none.
        max_size_mask (int): The maximum size of the masks.
        beam_size (int): The beam size for the search.
        length (int): The length of the search.
        mask_shape (tuple): The shape of the masks.
        device (torch.device): The device to use for the computation.

    Returns:
        best_label (int): The label of the best concept.
        best_iou (float): The IOU of the best concept.
        visited (int): The number of visited nodes.
    """

    if segmentations_info is None and heuristic != "none":
        raise ValueError(
            "segmentations_info must be provided when heuristic is not none"
        )
    # Compute commong parameters
    num_hits = activation_masks.sum()
    counters = None
    if length == 1:
        # if counter_variant:
            #     rank, counters = get_counter_weighted_netdissect_scores(activation_masks, segmentations, tau=0.01)
        if beam_variant == "new":
            rank = get_thresholded_netdissect_scores(activation_masks, segmentations, threshold=diff_threshold)
        else:
            # vanilla netdissect
            rank = get_netdissect_scores(activation_masks, segmentations)
        best_label = Counter(rank).most_common(1)[0][0]
        best_iou = Counter(rank).most_common(1)[0][1].item()
        return best_label, best_iou, 0, 0, 0
    if  heuristic == "mmesh":
        sample_activation_areas = activation_masks.sum(1)
        netdissect_scores, intersect_areas = get_augmented_netdissect_scores(
            activation_masks, segmentations
        )
        
        heuristic_info = (
            (segmentations_info[0], (segmentations_info[1][0], segmentations_info[1][1])),
            sample_activation_areas,
            intersect_areas,
        )
    elif heuristic == "optimal" or heuristic == "beam_optimal":
        seg_quantities = segmentations_info[2]
        if concept_quantities is None:
            seg_quantities, neuron_quantities, concept_quantities = get_optimal_heuristic_info(segmentations, activation_masks, seg_quantities)
        elif neuron_quantities is None:
            common_elements, unique_elements, uncoverable_elements = seg_quantities
            neuron_quantities = get_neuron_quantities(
                activation_masks, common_elements, unique_elements, uncoverable_elements
            )
                
        heuristic_info = (
            seg_quantities,
            neuron_quantities,
            concept_quantities,
        )
        if heuristic == "optimal":
            best_label, best_iou, visited, expanded, estimated = optimal_search.perform_exhaustive_heuristic_search(
                heuristic,
                heuristic_info,
                disjoint_info,
                segmentations,
                activation_masks,
                num_hits,
                length=length,
                max_size_mask=max_size_mask,
            )
            return best_label, best_iou, visited, expanded, estimated
        elif heuristic == "beam_optimal":
            best_label, best_iou, visited, expanded, estimated = beam_search.perform_exhaustive_heuristic_search(
                heuristic_info,
                disjoint_info,
                segmentations,
                activation_masks,
                num_hits,
                length=length,
                beam_size=beam_size,
                max_size_mask=max_size_mask,
                labels=labels,
                beam_variant=beam_variant,
                constraints=constraints,
                counter_variant=counter_variant,
                diff_threshold=diff_threshold,
                block_type_3=block_type_3,
            )
            return best_label, best_iou, visited, expanded, estimated
    elif heuristic == "none":
        netdissect_scores = get_netdissect_scores(
            activation_masks, segmentations
        )
        heuristic_info = None
    else:
        raise ValueError(
            f"Unknown heuristic {heuristic}. "
            "Available heuristics: mmesh, optimal, none."
        )
    best_label, best_iou, visited, expanded, estimated = heuristic_search.perform_heuristic_search(
        heuristic,
        netdissect_scores,
        segmentations,
        activation_masks,
        heuristic_info,
        num_hits,
        beam_size=beam_size,
        length=length,
        max_size_mask=max_size_mask,
        mask_shape=mask_shape,
        device=device,
    )
    return best_label, best_iou, visited, expanded, estimated
