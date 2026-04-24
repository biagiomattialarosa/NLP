import torch
from tqdm import tqdm

from . import formula as F
import numpy as np

def compute_quantities_vector(concept_mask, bitmaps, common_elements, unique_elements, bitmaps_coverable):
    """
    Computes the quantities of the concept mask with respect to the bitmaps.
    Args:
        concept_mask (torch.Tensor): The mask of the concept.
        bitmaps (torch.Tensor): The bitmaps of the dataset
        common_elements (torch.Tensor): The common elements in the bitmaps.
        unique_elements (torch.Tensor): The unique elements in the bitmaps.
        bitmaps_coverable (torch.Tensor): The coverable bitmaps.
    Returns:
        tuple: A tuple containing the common intersection, unique intersection,
                common extras, unique extras, and uncovered quantities.

    """

    concept_mask = concept_mask.to(bitmaps.device)   
    intersection = (concept_mask & bitmaps)
    unique_intersection = (intersection & unique_elements).sum(
        dim=1, dtype=torch.int32).to('cpu').numpy()
    common_intersection = (intersection & common_elements).sum(
        dim=1, dtype=torch.int32).to('cpu').numpy()
    extras = (concept_mask & (~bitmaps))
    common_extras = (extras & common_elements).sum(
        dim=1, dtype=torch.int32).to('cpu').numpy()
    unique_extras = (extras & unique_elements).sum(
        dim=1, dtype=torch.int32).to('cpu').numpy()
    uncovered = ((~concept_mask) & bitmaps_coverable).sum(
        dim=1, dtype=torch.int32).to('cpu').numpy()
    return (common_intersection, unique_intersection, common_extras, unique_extras, uncovered)

def formula_is_in(f, unary_areas, max_mask_size, enneary_areas=None):
    """
    Function to check where the formula is present in the data.

    Args:
        unary_areas (list): list of areas of the masks.
        f (src.formula.Formula): formula to check.
        max_mask_size (int): maximum size of the mask.
        enneary_areas (dict): dictionary of additional areas.

    Returns:
        Boolean list
    """
    if enneary_areas is not None and f in enneary_areas.keys():
        return enneary_areas[f] > 0
    if isinstance(f, F.And):
        masks_l = formula_is_in(
            f.left, unary_areas, max_mask_size, enneary_areas)
        masks_r = formula_is_in(
            f.right, unary_areas, max_mask_size, enneary_areas)
        return masks_l & masks_r
    elif isinstance(f, F.Or):
        masks_l = formula_is_in(
            f.left, unary_areas, max_mask_size, enneary_areas)
        masks_r = formula_is_in(
            f.right, unary_areas, max_mask_size, enneary_areas)
        return masks_l | masks_r
    elif isinstance(f, F.Not):
        return unary_areas[f.val.val] < max_mask_size
    elif isinstance(f, F.Leaf):
        return unary_areas[f.val] > 0


def get_coordinates(term, leaf_list, enneary_list):
    """Returns the coordinates of  of the formula
    Args:
        term (src.formula.Formula): term to check.
        leaf_list (dict): dictionary of leaf coordinates.
        enneary_list (dict): dictionary of enneary formulas coordinates.

    Returns:
        Coordinates of the formula
    """
    if isinstance(term, F.BinaryNode):
        coordinates = enneary_list[term]
    elif isinstance(term, F.Leaf):
        coordinates = leaf_list[term.val]
    else:
        coordinates = term
    return coordinates


def get_rectangles_overlap(coordinates_a, coordinates_b, return_points=False):
    """Returns the overlap between two rectangles given their
        top left and bottom right coordinates"""
    a_top_left_x = coordinates_a[:, 0, 1]
    a_top_left_y = coordinates_a[:, 0, 0]
    a_bottom_right_x = coordinates_a[:, 1, 1] + 1
    a_bottom_right_y = coordinates_a[:, 1, 0] + 1
    b_top_left_x = coordinates_b[:, 0, 1]
    b_top_left_y = coordinates_b[:, 0, 0]
    b_bottom_right_x = coordinates_b[:, 1, 1] + 1
    b_bottom_right_y = coordinates_b[:, 1, 0] + 1
    x_overlap = torch.maximum(
        torch.zeros_like(a_bottom_right_x),
        torch.minimum(a_bottom_right_x, b_bottom_right_x)
        - torch.maximum(a_top_left_x, b_top_left_x),
    )
    y_overlap = torch.maximum(
        torch.zeros_like(a_bottom_right_y),
        torch.minimum(a_bottom_right_y, b_bottom_right_y)
        - torch.maximum(a_top_left_y, b_top_left_y),
    )
    overlap = x_overlap * y_overlap
    if return_points:
        # compute coordinates of the intersection rectangle
        top_left_x = torch.maximum(a_top_left_x, b_top_left_x)
        top_left_y = torch.maximum(a_top_left_y, b_top_left_y)
        bottom_right_x = torch.minimum(a_bottom_right_x, b_bottom_right_x) - 1
        bottom_right_y = torch.minimum(a_bottom_right_y, b_bottom_right_y) - 1
        return overlap, torch.tensor(
            list(
                zip(
                    torch.tensor(list(zip(top_left_y, top_left_x))),
                    torch.tensor(list(zip(bottom_right_y, bottom_right_x))),
                )
            )
        )
    else:
        return overlap


def get_intersection_info(
    term, unary_intersect, enneary_intersect, neuron_areas
):
    """
    Returns the intersection between the term and the firing areas
    Args:
        term (src.formula.Formula): term to check.
        unary_intersect (dict): dictionary of unary intersections.
        enneary_intersect (dict): dictionary of enneary intersections.
        neuron_areas (torch.tensor): tensor of neuron areas.

    Returns:
        Intersection between the term and the firing areas
    """

    if isinstance(term, F.BinaryNode):
        term_and_fires_areas = enneary_intersect[term]
    elif isinstance(term, F.Not):
        term_and_fires_areas = neuron_areas - unary_intersect[term.val.val]
    else:
        term_and_fires_areas = unary_intersect[term.val]

    return term_and_fires_areas


def get_area_info(term, unary_areas, enneary_areas, max_size_mask):
    """
    Returns the area of the term
    Args:
        term (src.formula.Formula): term to check.
        unary_areas (dict): dictionary of unary areas.
        enneary_areas (dict): dictionary of enneary areas.
        max_size_mask (int): maximum size of the mask.

    Returns:
        Area of the term
    """
    if isinstance(term, F.BinaryNode):
        areas = enneary_areas[term]
    elif isinstance(term, F.Not):
        areas = max_size_mask - get_area_info(
            term.val, unary_areas, enneary_areas, max_size_mask
        )
    else:
        areas = unary_areas[term.val]
    return areas


def is_scene(areas, max_size_mask):
    """Returns True if the mask is a scene, False otherwise"""
    condition = (areas == 0) | (areas == max_size_mask)
    if condition.all():
        flag = True
    else:
        flag = False
    return flag


def compute_scene_iou(
    formula,
    left_areas,
    right_areas,
    left_intersection_area,
    right_intersection_area,
    neuron_areas,
    max_size_mask,
    num_hits
):
    """Computes the IoU of a scene formula
    Args:
        formula (src.formula.Formula): formula to check.
        left_areas (torch.tensor): sample areas of the left term.
        right_areas (torch.tensor): sample areas of the right term.
        left_intersection_area (torch.tensor): intersection areas
            with the neuron of the left term.
        right_intersection_area (torch.tensor): intersection areas
            with the neuron of the right term.
        neuron_areas (torch.tensor): tensor of neuron areas.
        max_size_mask (int): maximum size of the mask.
        num_hits (int): number of hits.
    Returns:
        IoU of the scene formula
    """
    if isinstance(formula, F.Or):
        # exact computation
        formula_mask = torch.where(
            left_areas > right_areas, left_areas, right_areas
        )
        intersection = torch.where(
            formula_mask == max_size_mask,
            neuron_areas,
            left_intersection_area + right_intersection_area,
        )

        intersection = torch.sum(intersection)
    elif isinstance(formula, F.And):
        formula_mask = torch.minimum(left_areas, right_areas)
        intersection = torch.where(
            left_intersection_area < right_intersection_area,
            left_intersection_area,
            right_intersection_area,
        )
        intersection = torch.sum(intersection)
    estimated_iou = intersection / (
        num_hits + torch.sum(formula_mask) - intersection
    )
    return torch.round(estimated_iou, decimals=4)


def mmesh_heuristic(formula, heuristic_info, *, num_hits, max_size_mask):
    """
    Computes the IoU of a formula using the mmesh heuristic.
    Args:
        formula (src.formula.Formula): formula to check.
        heuristic_info (tuple): tuple of unary and enneary heuristic_info collected from
            the dataset and the parsing of the previous beam.
        num_hits (int): number of hits in the neuron's mask.
        max_size_mask (int): maximum size of the mask.
    Returns:
        float: estimated iou
    """
    dissect_info = heuristic_info[0]
    enneary_info = heuristic_info[1]
    unary_info, neuron_areas, unary_intersection = dissect_info
    unary_areas, (unary_inscribed, unary_bounding_box) = unary_info
    enneary_areas, enneary_inscribed, enneary_bounding_box, enneary_intersection = enneary_info

    formula_in = formula_is_in(
         formula, unary_areas, max_size_mask, enneary_areas
    )

    left_and_fires_areas = get_intersection_info(
        formula.left, unary_intersection, enneary_intersection, neuron_areas
    )
    right_and_fires_areas = get_intersection_info(
        formula.right, unary_intersection, enneary_intersection, neuron_areas
    )
    left_areas = get_area_info(
        formula.left, unary_areas, enneary_areas, max_size_mask
    )
    right_areas = get_area_info(
        formula.right, unary_areas, enneary_areas, max_size_mask
    )

    left_intersection_area = left_and_fires_areas * formula_in
    right_intersection_area = right_and_fires_areas * formula_in

    # In case of scene formula, we can compute the exact formula mask
    # and in the OR case, we can compute the exact intersection
    left_is_scene = is_scene(left_areas, max_size_mask)
    right_is_scene = is_scene(right_areas, max_size_mask)
    if left_is_scene or right_is_scene:
        return compute_scene_iou(
            formula,
            left_areas,
            right_areas,
            left_intersection_area,
            right_intersection_area,
            neuron_areas,
            max_size_mask,
            num_hits,
        )

    # Otherswise, we have to approximate both of them
    if isinstance(formula, F.Or):
        max_intersection_neuron = torch.minimum(
            neuron_areas, left_intersection_area + right_intersection_area
        )
        minimum_area_mask = torch.maximum(left_areas, right_areas)
        coordinates_left = get_coordinates(
            formula.left, unary_bounding_box, enneary_bounding_box
        )
        coordinates_right = get_coordinates(
            formula.right, unary_bounding_box, enneary_bounding_box
        )
        maximum_intersection = get_rectangles_overlap(
            coordinates_left, coordinates_right
        )
        minimum_area_mask = torch.maximum(
            minimum_area_mask,
            left_areas + right_areas - maximum_intersection,
        )
        minimum_area_mask = torch.maximum(
            minimum_area_mask, max_intersection_neuron
        )
    elif isinstance(formula, F.And):
        max_intersection_neuron = torch.minimum(
            left_intersection_area, right_intersection_area
        )
        if isinstance(formula.right, F.Not):
            coordinates_left = get_coordinates(
                formula.left, unary_bounding_box, enneary_bounding_box
            )
            coordinates_right = get_coordinates(
                formula.right.val, unary_bounding_box, enneary_bounding_box
            )
            maximum_intersection = get_rectangles_overlap(
                coordinates_left, coordinates_right
            )
            minimum_area_mask = left_areas - maximum_intersection

        else:
            coordinates_left = get_coordinates(
                formula.left, unary_inscribed, enneary_inscribed
            )
            coordinates_right = get_coordinates(
                formula.right, unary_inscribed, enneary_inscribed
            )
            minimum_area_mask = get_rectangles_overlap(
                coordinates_left, coordinates_right
            )
        minimum_area_mask = torch.maximum(
            minimum_area_mask, max_intersection_neuron
        )

    max_intersection_neuron = torch.sum(max_intersection_neuron)
    minimum_area_mask = torch.sum(minimum_area_mask)
    estimated_iou = max_intersection_neuron / (
        num_hits + minimum_area_mask - max_intersection_neuron
    )
    return torch.round(estimated_iou, decimals=4)


def coordinates_free_heuristic(
        formula, heuristic_info, *, num_hits, max_size_mask):
    """ Heuristic that does not use coordinates
    to compute the minimum and maximum possible extension
    of the label mask.
    Args:
        formula (F.Formula): formula to estimate
        heuristic_info (tuple): tuple of information available for the heuristic
        num_hits (int): number of hits in the neuron mask
        max_size_mask (int): maximum size of the mask
    Returns:
        float: estimated iou
    """
    dissect_info = heuristic_info[0]
    enneary_info = heuristic_info[1]
    unary_areas, neuron_areas, unary_intersection = dissect_info
    enneary_areas, enneary_intersection = enneary_info

    formula_in = formula_is_in(
        formula, unary_areas, max_size_mask, enneary_areas
    )

    left_and_fires_areas = get_intersection_info(
        formula.left, unary_intersection, enneary_intersection, neuron_areas
    )
    right_and_fires_areas = get_intersection_info(
        formula.right, unary_intersection, enneary_intersection, neuron_areas
    )
    left_areas = get_area_info(
        formula.left, unary_areas, enneary_areas, max_size_mask
    )
    right_areas = get_area_info(
        formula.right, unary_areas, enneary_areas, max_size_mask
    )

    left_intersection_area = left_and_fires_areas * formula_in
    right_intersection_area = right_and_fires_areas * formula_in

    # In case of scene formula, we can compute the exact formula mask
    # and in the OR case, we can compute the exact intersection
    left_is_scene = is_scene(left_areas, max_size_mask)
    right_is_scene = is_scene(right_areas, max_size_mask)
    if left_is_scene or right_is_scene:
        return compute_scene_iou(
            formula,
            left_areas,
            right_areas,
            left_intersection_area,
            right_intersection_area,
            neuron_areas,
            max_size_mask,
            num_hits,
        )

    # Otherswise, we have to approximate both of them
    if isinstance(formula, F.Or):
        max_intersection_neuron = torch.minimum(
            neuron_areas, left_intersection_area + right_intersection_area
        )
    elif isinstance(formula, F.And):
        max_intersection_neuron = torch.minimum(
            left_intersection_area, right_intersection_area
        )

    max_intersection_neuron = torch.sum(max_intersection_neuron)
    estimated_iou = max_intersection_neuron / (
        num_hits - max_intersection_neuron
    )
    return torch.round(estimated_iou, decimals=4)


def areas_heuristic(formula, heuristic_info, *, num_hits, max_size_mask):
    """ Compute the heuristic based onnly on the areas of the formula
    and the areas of the neuron's mask.

    Args:
        formula (F.Formula): The formula to compute the heuristic for
        heuristic_info (tuple): The information to compute the heuristic
        num_hits (int): The number of hits in the neuron mask
        max_size_mask (int): The maximum size of the mask

    Returns:
        float: estimated iou
    """
    unary_info = heuristic_info[0]
    enneary_info = heuristic_info[1]
    unary_areas, neuron_areas = unary_info
    enneary_areas = enneary_info

    left_areas = get_area_info(
        formula.left, unary_areas, enneary_areas, max_size_mask
    )
    right_areas = get_area_info(
        formula.right, unary_areas, enneary_areas, max_size_mask
    )

    # Otherswise, we have to approximate both of them
    if isinstance(formula, F.Or):
        max_intersection_neuron = torch.minimum(
            neuron_areas, left_areas + right_areas
        )
    elif isinstance(formula, F.And):
        max_intersection_neuron = torch.minimum(left_areas, right_areas)

    max_intersection_neuron = torch.sum(max_intersection_neuron)
    estimated_iou = max_intersection_neuron / (
        num_hits - max_intersection_neuron
    )
    return torch.round(estimated_iou, decimals=4)


def sort_search_space_by(
        search_space, name_heuristic, *,
        heuristic_info, num_hits,  max_size_mask):
    """
    Sort the search space using the heuristic name_heuristic.

    Args:
        search_space (list of Formula): the search space to sort
        name_heuristic (str): the name of the heuristic to use
        heuristic_info (tuple): the information to be used by the heuristic
        num_hits (int): the number of hits of the neuron
        max_size_mask (int): the maximum size of the mask

    Returns:
        list of Formula: the sorted search space
    """

    if name_heuristic == "mmesh":
        heuristic = mmesh_heuristic
    elif name_heuristic == "none":
        for index_formula, candidate_formula in enumerate(search_space):
            search_space[index_formula].iou = 1.0
            search_space[index_formula] = F.OrderedFormula(
                search_space[index_formula]
            )
        return search_space
    else:
        raise ValueError(f"Unknown heuristic: {name_heuristic}")

    for index_formula, candidate_formula in enumerate(search_space):
        esti = heuristic(
            candidate_formula, heuristic_info, num_hits=num_hits,
            max_size_mask=max_size_mask
        )
        search_space[index_formula] = F.OrderedFormula(
            search_space[index_formula]
        )
        search_space[index_formula].iou = esti

    search_space = sorted(search_space, reverse=True)

    return search_space

def get_max_min_quantity(quantity):
    """
    Returns the maximum and minimum values of a quantity.
    
    Args:
        quantity (tuple): A tuple containing the quantity.
        
    Returns:
        tuple: Maximum and minimum values of the quantity.
    """
    tuple_quantity = quantity[0]
    if len(tuple_quantity) == 2:
        # In this case, the structure of quantity is ((max_sample, max_sum), (min_sample, min_sum))
        max_quantity_sample = tuple_quantity[0]
        max_quantity_sum = tuple_quantity[1]
        min_quantity_sample = quantity[1][0]
        min_quantity_sum = quantity[1][1]
        
    else:
        # In this case, the structure of quantity is (quantity_sample, sum)
        max_quantity_sample = quantity[0]
        max_quantity_sum = quantity[1]
        min_quantity_sample = max_quantity_sample
        min_quantity_sum = max_quantity_sum
    return ( 
        max_quantity_sample, max_quantity_sum  
    ), (
        min_quantity_sample, min_quantity_sum
    )


def or_chain_estimation(label, common_intersection, unique_intersection, common_extras, unique_extras, uncovered, neuron_unique, neuron_common, neuron_coverable, neuron_sum, max_improvement, num_hits, max_size_mask, max_length):
     # Aux variables
    label_len = len(label)
    k = max_length - label_len # Missing length   
    TOP_INDEX = 0
    BOTTOM_INDEX = 1

    # Unpack improvement information
    improv_common_intersection, improv_unique_intersection, improv_common_extras, improv_unique_extras, improv_uncovered = max_improvement

    # Unpack max and min quantities
    max_common_intersection, min_common_intersection = get_max_min_quantity(common_intersection)
    max_unique_intersection, min_unique_intersection = get_max_min_quantity(unique_intersection)
    max_common_extras, min_common_extras = get_max_min_quantity(common_extras)
    max_unique_extras, min_unique_extras = get_max_min_quantity(unique_extras)
    
    max_label_intersection = max_common_intersection + max_unique_intersection
    # We discard the labels that cannot increase the IoU
    if max_label_intersection.sum() == 0:
        return 0.0, 0.0
    zero_vector = torch.zeros_like(max_common_intersection)
    max_top_intersection = improv_common_intersection[k][TOP_INDEX] + improv_unique_intersection[k][TOP_INDEX]
    min_bottom_intersection = improv_common_intersection[0][BOTTOM_INDEX] + improv_unique_intersection[0][BOTTOM_INDEX]
    min_bottom_extras = improv_common_extras[0][BOTTOM_INDEX] + improv_unique_extras[0][BOTTOM_INDEX]
    min_label_intersection = min_common_intersection + min_unique_intersection
    max_label_extras = max_common_extras + max_unique_extras
    min_label_extras = min_common_extras + min_unique_extras
    min_added_extras = torch.maximum(zero_vector, min_bottom_extras - max_label_extras)
    max_top_extras = improv_common_extras[k][TOP_INDEX] + improv_unique_extras[k][TOP_INDEX]

    # Max IoU
    max_intersection = torch.minimum(max_label_intersection + max_top_intersection, neuron_coverable)
    min_union = torch.clamp(neuron_sum + min_label_extras + min_added_extras, max=max_size_mask)
    max_iou = max_intersection.sum() / min_union.sum()

    # Min IoU
    min_intersection = torch.maximum(min_label_intersection, min_bottom_intersection)
    max_union = torch.clamp(neuron_sum + max_label_extras + max_top_extras, max=max_size_mask)
    min_iou = min_intersection.sum() / max_union.sum()
    return max_iou, min_iou

def numpy_or_chain_estimation(label, common_intersection, unique_intersection, common_extras, unique_extras, uncovered, neuron_unique_tuple, neuron_common_tuple, neuron_coverable_tuple, neuron_sum_tuple, max_improvement, num_hits, max_size_mask, max_length, debug=False):
    # Aux variables
    label_len = len(label)
    k = max_length - label_len # Missing length   
    TOP_INDEX = 0
    BOTTOM_INDEX = 1

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
    
    max_label_intersection = max_common_intersection + max_unique_intersection
    
    #zero_vector = np.zeros_like(max_common_intersection)
    max_top_intersection = improv_common_intersection[k][TOP_INDEX] + improv_unique_intersection[k][TOP_INDEX]
    min_bottom_intersection = improv_common_intersection[0][BOTTOM_INDEX] + improv_unique_intersection[0][BOTTOM_INDEX]
    min_bottom_extras = improv_common_extras[0][BOTTOM_INDEX] + improv_unique_extras[0][BOTTOM_INDEX]
    min_label_intersection = min_common_intersection + min_unique_intersection
    max_label_extras = max_common_extras + max_unique_extras
    min_label_extras = min_common_extras + min_unique_extras
    min_added_extras = np.clip(min_bottom_extras - max_label_extras, a_min=0, a_max=None)
    max_top_extras = improv_common_extras[k][TOP_INDEX] + improv_unique_extras[k][TOP_INDEX]

    # Max IoU
    max_intersection = np.minimum(max_label_intersection + max_top_intersection, neuron_coverable)
    min_union = np.clip(neuron_sum + min_label_extras + min_added_extras, a_min=0, a_max=max_size_mask)
    max_iou = max_intersection.sum() / min_union.sum()

    # Min IoU
    min_intersection = np.maximum(min_label_intersection, min_bottom_intersection)
    max_union = np.clip(neuron_sum + max_label_extras + max_top_extras, a_min=0, a_max=max_size_mask)
    min_iou = min_intersection.sum() / max_union.sum()
    if debug:
        print(f"Max IoU: {max_iou}, Min IoU: {min_iou}")
        print(f"Top Intersection: {max_top_intersection.sum()}, Bottom Intersection: {min_added_extras.sum()}")
        print(f"Max Intersection: {max_intersection.sum()}, Min Union: {min_union.sum()}")
        print(f"Min Intersection: {min_intersection.sum()}, Max Union: {max_union.sum()}")
    return max_iou, min_iou


def and_chain_estimation(label, common_intersection, unique_intersection, common_extras, unique_extras, uncovered, neuron_unique, neuron_common, neuron_coverable, neuron_sum, max_improvement, num_hits, max_size_mask, max_length, debug=False):
    # Aux variables
    label_len = len(label)
    k = max_length - label_len # Missing length   
    TOP_INDEX = 0
    BOTTOM_INDEX = 1

    # Unpack improvement information
    improv_common_intersection, improv_unique_intersection, improv_common_extras, improv_unique_extras, improv_uncovered = max_improvement

    # Unpack max and min quantities
    max_common_intersection, min_common_intersection = get_max_min_quantity(common_intersection)
    max_unique_intersection, min_unique_intersection = get_max_min_quantity(unique_intersection)
    max_common_extras, min_common_extras = get_max_min_quantity(common_extras)
    max_unique_extras, min_unique_extras = get_max_min_quantity(unique_extras)

    # We discard the labels that cannot increase the IoU
    max_label_intersection = max_common_intersection + max_unique_intersection
    # We discard the labels that cannot increase the IoU
    if max_label_intersection.sum() == 0:
        return 0.0, 0.0
    
    zero_vector = torch.zeros_like(max_common_intersection)
    one_top_common_intersection = improv_common_intersection[0][TOP_INDEX]
    one_bottom_common_intersection = improv_common_intersection[0][BOTTOM_INDEX]
    one_bottom_common_extras = improv_common_extras[0][BOTTOM_INDEX] 
    one_top_common_extras = improv_common_extras[0][TOP_INDEX]

    # MaxIoU
    max_intersection = torch.minimum(max_common_intersection, one_top_common_intersection)
    min_union = torch.clamp(neuron_sum + torch.maximum(zero_vector,
        max_common_intersection +  one_bottom_common_extras - (max_size_mask - neuron_sum - max_unique_extras),
            ), max=max_size_mask)
    max_iou = max_intersection.sum() / min_union.sum()

    # MinIoU
    min_intersection = torch.minimum(zero_vector, min_common_intersection + one_bottom_common_intersection - neuron_common).sum()
    max_union = num_hits + torch.minimum(max_common_extras, one_top_common_extras).sum()
    min_iou = min_intersection / max_union 
    return max_iou, min_iou

def numpy_and_chain_estimation(label, common_intersection, unique_intersection, common_extras, unique_extras, uncovered, neuron_unique_tuple, neuron_common_tuple, neuron_coverable_tuple, neuron_sum_tuple, max_improvement, num_hits, max_size_mask, max_length, debug=False):
    # Aux variables
    label_len = len(label)
    k = max_length - label_len # Missing length   
    TOP_INDEX = 0
    BOTTOM_INDEX = 1

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
    # max_label_intersection = max_common_intersection + max_unique_intersection
    # # We discard the labels that cannot increase the IoU
    # if max_label_intersection.sum() == 0:
    #     return 0.0, 0.0
    
    #zero_vector = np.zeros_like(max_common_intersection)
    one_top_common_intersection = improv_common_intersection[0][TOP_INDEX]
    one_bottom_common_intersection = improv_common_intersection[0][BOTTOM_INDEX]
    one_bottom_common_extras = improv_common_extras[0][BOTTOM_INDEX] 
    one_top_common_extras = improv_common_extras[0][TOP_INDEX]

    # MaxIoU
    max_intersection = np.minimum(max_common_intersection, one_top_common_intersection)
    min_union = np.clip(neuron_sum + np.clip(
        max_common_intersection +  one_bottom_common_extras - (max_size_mask - neuron_sum - max_unique_extras),
            a_min=0, a_max=max_size_mask), a_min=0, a_max=max_size_mask)
    max_iou = max_intersection.sum() / min_union.sum()

    # MinIoU
    min_intersection = np.clip(min_common_intersection + one_bottom_common_intersection - neuron_common, a_min=0, a_max=None).sum()
    max_union = num_hits + np.minimum(max_common_extras, one_top_common_extras).sum()
    min_iou = min_intersection / max_union 

    if debug:
        print(f"Max IoU: {max_iou}, Min IoU: {min_iou}")
        print(f"Max Intersection: {max_intersection.sum()}, Min Union: {min_union.sum()}")
        print(f"Min Intersection: {min_intersection.sum()}, Max Union: {max_union.sum()}")
    return max_iou, min_iou

def and_not_chain_estimation(label, common_intersection, unique_intersection, common_extras, unique_extras, uncovered, neuron_unique, neuron_common, neuron_coverable, neuron_sum, max_improvement, num_hits, max_size_mask, max_length, debug=False):
    # Aux variables  
    TOP_INDEX = 0


    # Unpack improvement information
    improv_common_intersection, improv_unique_intersection, improv_common_extras, improv_unique_extras, improv_uncovered = max_improvement

    # Unpack max and min quantities
    max_common_intersection, min_common_intersection = get_max_min_quantity(common_intersection)
    max_unique_intersection, min_unique_intersection = get_max_min_quantity(unique_intersection)
    max_common_extras, min_common_extras = get_max_min_quantity(common_extras)
    max_unique_extras, min_unique_extras = get_max_min_quantity(unique_extras)

    zero_vector = torch.zeros_like(max_common_intersection)
    max_label_intersection = max_common_intersection + max_unique_intersection
    # We discard the labels that cannot increase the IoU
    if max_label_intersection.sum() == 0:
        return 0.0, 0.0
    
    # Max IoU
    max_intersection = torch.minimum(max_unique_intersection + torch.minimum(
        max_common_intersection, improv_uncovered[0][TOP_INDEX]
    ), neuron_coverable)
    min_union = num_hits + min_unique_extras.sum()
    max_iou = max_intersection.sum() / min_union

    # Min IoU
    min_intersection = min_unique_intersection
    max_union = num_hits + max_unique_extras.sum() + max_common_extras.sum()
    min_iou = min_intersection.sum() / max_union
    return max_iou, min_iou

def numpy_and_not_chain_estimation(label, common_intersection, unique_intersection, common_extras, unique_extras, uncovered, neuron_unique_tuple, neuron_common_tuple, neuron_coverable_tuple, neuron_sum_tuple, max_improvement, num_hits, max_size_mask, max_length, debug=False):
    # Aux variables
    TOP_INDEX = 0

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
    #max_label_intersection = max_common_intersection + max_unique_intersection
    # We discard the labels that cannot increase the IoU
    if max_common_intersection_sum + max_unique_intersection_sum == 0:
        return 0.0, 0.0
    # if max_label_intersection.sum() == 0:
    #     return 0.0, 0.0
    
    # Max IoU
    max_intersection = np.minimum(max_unique_intersection + np.minimum(
        max_common_intersection, improv_uncovered[0][TOP_INDEX]
    ), neuron_coverable)
    min_union = num_hits + min_unique_extras_sum
    max_iou = max_intersection.sum() / min_union

    # Min IoU
    min_intersection = min_unique_intersection_sum
    max_union = num_hits + max_unique_extras_sum + max_common_extras_sum
    min_iou = min_intersection / max_union

    if debug:
        print(f"Max IoU: {max_iou}, Min IoU: {min_iou}")
        print(f"Max Intersection: {max_intersection.sum()}, Min Union: {min_union.sum()}")
        print(f"Min Intersection: {min_intersection.sum()}, Max Union: {max_union.sum()}")
    
    return max_iou, min_iou

def comb_and_or_chain_estimation(label, common_intersection, unique_intersection, common_extras, unique_extras, uncovered, neuron_unique, neuron_common, neuron_coverable, neuron_sum, max_improvement, num_hits, max_size_mask, max_length, debug=False):
    # Aux variables
    label_len = len(label)
    k = max_length - label_len # Missing length   
    TOP_INDEX = 0
    BOTTOM_INDEX = 1

    # Unpack improvement information
    improv_common_intersection, improv_unique_intersection, improv_common_extras, improv_unique_extras, improv_uncovered = max_improvement

    # Unpack max and min quantities
    max_common_intersection, min_common_intersection = get_max_min_quantity(common_intersection)
    max_common_extras, min_common_extras = get_max_min_quantity(common_extras)
    max_unique_extras, min_unique_extras = get_max_min_quantity(unique_extras)

    zero_vector = torch.zeros_like(max_common_intersection)
    topk_common_intersection = improv_common_intersection[k][TOP_INDEX]
    bottom1_common_extras = improv_common_extras[0][BOTTOM_INDEX]
    bottom1_common_intersection = improv_common_intersection[0][BOTTOM_INDEX]
    top1_common_extras = improv_common_extras[0][TOP_INDEX]
    one_bottom_common_intersection = improv_common_intersection[0][BOTTOM_INDEX]
    one_top_common_extras = improv_common_extras[0][TOP_INDEX]

    if max_common_intersection.sum() == 0:
        return 0.0, 0.0
    
    # Max IoU
    max_intersection = torch.minimum(max_common_intersection + topk_common_intersection, neuron_common)
    min_union = torch.clamp(neuron_sum + torch.maximum(zero_vector,
                                         min_common_extras + bottom1_common_extras -
                                            (max_size_mask - neuron_sum - min_unique_extras)),
                                            max=max_size_mask)
    max_iou = max_intersection.sum() / min_union.sum()

    # Min IoU
    min_intersection = np.clip(min_common_intersection + one_bottom_common_intersection - neuron_common, a_min=0, a_max=None).sum()
    max_union = num_hits + np.minimum(max_common_extras, one_top_common_extras).sum()
    min_iou = min_intersection / max_union 
    
    if debug:
        print(f"Max IoU: {max_iou}, Min IoU: {min_iou}")
        print(f"Max Intersection: {max_intersection.sum()}, Min Union: {min_union.sum()}")
        print(f"Min Intersection: {min_intersection.sum()}, Max Union: {max_union.sum()}")
    return max_iou, min_iou

def numpy_comb_and_or_chain_estimation(label, common_intersection, unique_intersection, common_extras, unique_extras, uncovered, neuron_unique_tuple, neuron_common_tuple, neuron_coverable_tuple, neuron_sum_tuple, max_improvement, num_hits, max_size_mask, max_length, debug=False):
    # Aux variables
    label_len = len(label)
    k = max_length - label_len # Missing length   
    TOP_INDEX = 0
    BOTTOM_INDEX = 1

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
    topk_common_intersection = improv_common_intersection[k][TOP_INDEX]
    topk_unique_intersection = improv_unique_intersection[k][TOP_INDEX]
    bottom1_common_extras = improv_common_extras[0][BOTTOM_INDEX]
    bottom1_common_intersection = improv_common_intersection[0][BOTTOM_INDEX]
    top1_common_extras = improv_common_extras[0][TOP_INDEX]

    if max_common_intersection_sum == 0:
        return 0.0, 0.0 
    # if max_common_intersection.sum() == 0:
    #     return 0.0, 0.0
    
    # Max IoU
    max_intersection = np.minimum(max_common_intersection + topk_common_intersection + topk_unique_intersection, neuron_coverable - min_unique_intersection)
    min_union = np.clip(neuron_sum + np.maximum(0,
                                         min_common_extras + bottom1_common_extras -
                                            (max_size_mask - neuron_sum - min_unique_extras)),
                                            a_min=0, a_max=max_size_mask)
    max_iou = max_intersection.sum() / min_union.sum()

    # Min IoU
    min_intersection = np.minimum(min_common_intersection, bottom1_common_intersection)
    max_union = num_hits + np.minimum(max_common_extras, top1_common_extras).sum()
    min_iou = min_intersection.sum() / max_union
    
    if debug:
        print(f"Max IoU: {max_iou}, Min IoU: {min_iou}")
        print(f"Max Intersection: {max_intersection.sum()}, Min Union: {min_union.sum()}")
        print(f"Min Intersection: {min_intersection.sum()}, Max Union: {max_union.sum()}")
    
    return max_iou, min_iou


def comb_or_andnot_chain_estimation(label, common_intersection, unique_intersection, common_extras, unique_extras, uncovered, neuron_unique, neuron_common, neuron_coverable, neuron_sum, max_improvement, num_hits, max_size_mask, max_length, debug=False):
    # Aux variables
    label_len = len(label)
    k = max_length - label_len # Missing length   
    TOP_INDEX = 0
    BOTTOM_INDEX = 1

    
    # Unpack improvement information
    improv_common_intersection, improv_unique_intersection, improv_common_extras, improv_unique_extras, improv_uncovered = max_improvement

    # Unpack max and min quantities
    max_common_intersection, min_common_intersection = get_max_min_quantity(common_intersection)
    max_unique_intersection, min_unique_intersection = get_max_min_quantity(unique_intersection)
    max_common_extras, min_common_extras = get_max_min_quantity(common_extras)
    max_unique_extras, min_unique_extras = get_max_min_quantity(unique_extras)

    max_label_intersection = max_common_intersection + max_unique_intersection
    # We discard the labels that cannot increase the IoU
    if max_label_intersection.sum() == 0:
        return 0.0, 0.0
    topk_common_intersection = improv_common_intersection[k][TOP_INDEX]
    bott1_unique_intersection = improv_unique_intersection[0][BOTTOM_INDEX]

    # Max IoU
    max_intersection = torch.minimum(
        max_label_intersection + topk_common_intersection - bott1_unique_intersection,
        neuron_coverable
    )
    min_union = num_hits + min_unique_extras.sum()
    max_iou = max_intersection.sum() / min_union
    # Min IoU
    min_intersection = min_unique_intersection
    max_union = num_hits + max_unique_extras.sum() + max_common_extras.sum()
    min_iou = min_intersection.sum() / max_union
    return max_iou, min_iou

def numpy_comb_or_andnot_chain_estimation(label, common_intersection, unique_intersection, common_extras, unique_extras, uncovered, neuron_unique_tuple, neuron_common_tuple, neuron_coverable_tuple, neuron_sum_tuple, max_improvement, num_hits, max_size_mask, max_length, debug=False):
    # Aux variables
    label_len = len(label)
    k = max_length - label_len # Missing length   
    TOP_INDEX = 0
    BOTTOM_INDEX = 1

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
    #max_common_extras = max_common_extras_tuple[0]
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
    #neuron_coverable_sum = neuron_coverable_tuple[1]
    neuron_sum = neuron_sum_tuple[0]
    #neuron_sum_sum = neuron_sum_tuple[1]

    max_label_intersection = max_common_intersection + max_unique_intersection
    # We discard the labels that cannot increase the IoU
    if max_common_intersection_sum + max_unique_intersection_sum == 0:
        return 0.0, 0.0
    # if max_label_intersection.sum() == 0:
    #     return 0.0, 0.0
    topk_common_intersection = improv_common_intersection[k][TOP_INDEX]
    topk_unique_intersection = improv_unique_intersection[k][TOP_INDEX]
    bott1_unique_intersection = improv_unique_intersection[0][BOTTOM_INDEX]
    bott1_unique_extras = improv_unique_extras[0][BOTTOM_INDEX]
    top_k_minus_1_extras = improv_unique_extras[k-1][TOP_INDEX] + improv_common_extras[k-1][TOP_INDEX]
    
    # Max IoU
    max_intersection = np.minimum(
        max_label_intersection + topk_common_intersection + topk_unique_intersection,
        neuron_coverable
    )
    min_union = num_hits + np.maximum(min_unique_extras, bott1_unique_extras).sum()
    max_iou = max_intersection.sum() / min_union
    # Min IoU
    min_intersection = min_unique_intersection_sum
    max_union = np.clip(neuron_sum + max_unique_extras + top_k_minus_1_extras, a_min=0, a_max=max_size_mask).sum()
    min_iou = min_intersection / max_union    

    if debug:
        print(f"Max IoU: {max_iou}, Min IoU: {min_iou}")
        print(f"Max Intersection: {max_intersection.sum()}, Min Union: {min_union.sum()}")
        print(f"Min Intersection: {min_intersection.sum()}, Max Union: {max_union.sum()}")
    return max_iou, min_iou

def individual_estimation(label, common_intersection, unique_intersection, common_extras, unique_extras, uncovered, neuron_unique, neuron_common, neuron_coverable, neuron_sum, max_improvement, num_hits, max_size_mask, max_length, debug=False):

    # Unpack max and min quantities
    max_common_intersection, min_common_intersection = get_max_min_quantity(common_intersection)
    max_unique_intersection, min_unique_intersection = get_max_min_quantity(unique_intersection)
    max_common_extras, min_common_extras = get_max_min_quantity(common_extras)
    max_unique_extras, min_unique_extras = get_max_min_quantity(unique_extras)

    max_label_intersection = max_common_intersection + max_unique_intersection
    # We discard the labels that cannot increase the IoU
    if max_label_intersection.sum() == 0:
        return 0.0, 0.0
    neuron_activation = neuron_common + neuron_unique
    # Max IoU
    max_intersection = torch.minimum(
        max_label_intersection,
        neuron_activation
    )
    min_union = num_hits + min_unique_extras.sum() + min_common_extras.sum()
    max_iou = max_intersection.sum() / min_union

    # Min IoU
    min_intersection = torch.minimum(min_common_intersection+ min_unique_intersection, neuron_activation)
    max_union = num_hits + max_unique_extras.sum() + max_common_extras.sum()
    min_iou = min_intersection.sum() / max_union
   
    if debug:
        print(f"Max IoU: {max_iou}, Min IoU: {min_iou}")
        print(f"Max Intersection: {max_intersection.sum()}, Min Union: {min_union.sum()}")
        print(f"Min Intersection: {min_intersection.sum()}, Max Union: {max_union.sum()}")
    return max_iou, min_iou

def numpy_individual_estimation(label, common_intersection, unique_intersection, common_extras, unique_extras, uncovered, neuron_unique_tuple, neuron_common_tuple, neuron_coverable_tuple, neuron_sum_tuple, max_improvement, num_hits, max_size_mask, max_length, debug=False):
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
    if max_common_intersection_sum + max_unique_intersection_sum == 0:
        return 0.0, 0.0
    # We discard the labels that cannot increase the IoU
    # if max_label_intersection.sum() == 0:
    #     # print("si")
    #     # print(max_common_intersection_sum, max_unique_intersection_sum)
    #     # print(max_common_intersection.sum(), max_unique_intersection.sum())
        return 0.0, 0.0
    #neuron_activation = neuron_common + neuron_unique
    # Max IoU
    # max_intersection = np.minimum(
    #     max_label_intersection,
    #     neuron_coverable
    # )
    min_union = num_hits + min_unique_extras_sum + min_common_extras_sum
    max_iou = (max_common_intersection_sum + max_unique_intersection_sum) / min_union

    # Min IoU
    #min_intersection = np.minimum(min_common_intersection + min_unique_intersection, neuron_coverable)
    min_intersection = min_common_intersection_sum + min_unique_intersection_sum
    max_union = num_hits + max_unique_extras_sum + max_common_extras_sum
    min_iou = min_intersection / max_union

    if debug:
        print(f"Max IoU: {max_iou}, Min IoU: {min_iou}")
        #print(f"Max Intersection: {max_intersection.sum()}, Min Union: {min_union.sum()}")
        print(f"Min Intersection: {min_intersection.sum()}, Max Union: {max_union.sum()}")
    return max_iou, min_iou

def estimate_optimal_label_iou(node, label_mapping, heuristic_info, max_improvement, num_hits, max_size_mask, max_length, fpd_matrix):
    """
    Estimate the IoU of a label using the optimal heuristic.
    
    Args:
        label (F.Formula): The label to estimate the IoU for.
        heuristic_info (tuple): The information to compute the heuristic.
        
    Returns:
        float: Estimated IoU of the label.
    """
    seg_quantities, neuron_quantities, concepts_quantities = heuristic_info
    neuron_unique, neuron_common, neuron_coverable_mask, neuron_coverable, neuron_sum  = neuron_quantities
    #common_intersection, unique_intersection, common_extras, unique_extras, uncovered = concepts_quantities
    
    label = node[2]
    ops_to_expand = node[1]
    label_node = (None, ops_to_expand, label)
    label_common_intersection, label_unique_intersection, label_common_extras, label_unique_extras, label_uncovered = get_esti_quantities(
        label_node, label_mapping, heuristic_info, max_size_mask, fpd_matrix=fpd_matrix)
    if label_common_intersection is None:
        # Label discarded at the previous step
        return [([], 0.0)], 0.0
    start_time = time.time()
    # Individual estimation
    max_individual_iou, min_individual_iou = individual_estimation(label,
        label_common_intersection, label_unique_intersection,
        label_common_extras, label_unique_extras, label_uncovered,
        neuron_unique, neuron_common, neuron_coverable, neuron_sum,
        max_improvement, num_hits, max_size_mask, max_length
    )
    individual_time = time.time()
    print(f"Individual estimation time: {individual_time - start_time:.4f} seconds")

    if len(label) == max_length:
        # If the label is already at maximum length, we cannot estimate any chain
        return [([], max_individual_iou)], min_individual_iou

    if ops_to_expand is None or ops_to_expand[0] == 'OR':
        max_or_chain_iou, min_or_chain_iou = or_chain_estimation(label,
            label_common_intersection, label_unique_intersection,
            label_common_extras, label_unique_extras, label_uncovered,
            neuron_unique, neuron_common, neuron_coverable, neuron_sum,
            max_improvement, num_hits, max_size_mask, max_length
        )
        or_chain_time = time.time()
        print(f"OR chain estimation time: {or_chain_time - individual_time:.4f} seconds")
    else:
        max_or_chain_iou, min_or_chain_iou = 0.0, 0.0

    # This covers AND chains
    if ops_to_expand is None or (len(ops_to_expand) == 1 and (ops_to_expand[0] == 'AND' or ops_to_expand[0] == 'NOT')) :
        max_and_chain_iou, min_and_chain_iou = and_chain_estimation(label,
            label_common_intersection, label_unique_intersection,
            label_common_extras, label_unique_extras, label_uncovered,
            neuron_unique, neuron_common, neuron_coverable, neuron_sum,
            max_improvement, num_hits, max_size_mask, max_length
        )
        and_chain_time = time.time()
        print(f"AND chain estimation time: {and_chain_time - individual_time:.4f} seconds")
    else:
        max_and_chain_iou, min_and_chain_iou = 0.0, 0.0


    if ops_to_expand is None or (len(ops_to_expand) == 2 and (ops_to_expand[0] == 'AND' and ops_to_expand[1] == 'NOT')):
        max_and_not_chain_iou, min_and_not_chain_iou = and_not_chain_estimation(label,
            label_common_intersection, label_unique_intersection,
            label_common_extras, label_unique_extras, label_uncovered,
            neuron_unique, neuron_common, neuron_coverable, neuron_sum,
            max_improvement, num_hits, max_size_mask, max_length
        )
        and_not_chain_time = time.time()
        print(f"AND-NOT chain estimation time: {and_not_chain_time - individual_time:.4f} seconds")
    else:
        max_and_not_chain_iou, min_and_not_chain_iou = 0.0, 0.0


    # Compute them only if there is space for 2-chains
    if max_length - len(label) > 1:
        # This covers AND-NOT chains
        max_and_and_not_chain_iou = max_and_chain_iou
        min_and_and_not_chain_iou = min_and_chain_iou

        if ops_to_expand is None or (len(ops_to_expand) == 3 and (ops_to_expand[0] == 'AND' and ops_to_expand[1] == 'OR' and ops_to_expand[2] == 'NOT')):
        # This covers both AND-OR chaing and everything chain
            max_comb_and_or_chain_iou, min_comb_and_or_chain_iou = comb_and_or_chain_estimation(label,
                label_common_intersection, label_unique_intersection,
                label_common_extras, label_unique_extras, label_uncovered,
                neuron_unique, neuron_common, neuron_coverable, neuron_sum,
                max_improvement, num_hits, max_size_mask, max_length
            )
            print(f"Comb-AND-OR chain estimation time: {time.time() - individual_time:.4f} seconds")
        else:
            max_comb_and_or_chain_iou, min_comb_and_or_chain_iou = 0.0, 0.0

        if ops_to_expand is None or (len(ops_to_expand) == 2 and (ops_to_expand[0] == 'OR' and ops_to_expand[1] == 'NOT')):
            max_comb_or_andnot_chain_iou, min_comb_or_andnot_chain_iou = comb_or_andnot_chain_estimation(label,
                label_common_intersection, label_unique_intersection,
                label_common_extras, label_unique_extras, label_uncovered,
                neuron_unique, neuron_common, neuron_coverable, neuron_sum,
                max_improvement, num_hits, max_size_mask, max_length
            )
            print(f"Comb-OR-ANDNOT chain estimation time: {time.time() - individual_time:.4f} seconds")
        else:
            max_comb_or_andnot_chain_iou, min_comb_or_andnot_chain_iou = 0.0, 0.0

    else:
        max_and_and_not_chain_iou, min_and_and_not_chain_iou = 0.0, 0.0
        max_comb_and_or_chain_iou, min_comb_and_or_chain_iou = 0.0, 0.0
        max_comb_or_andnot_chain_iou, min_comb_or_andnot_chain_iou = 0.0, 0.0


    
    # Greater chains
    if max_length - len(label) > 2:
        # This covers both everything chain
        max_every_chain_iou = max_comb_and_or_chain_iou
        min_every_chain_iou = min_comb_and_or_chain_iou
    else:
        max_every_chain_iou, min_every_chain_iou = 0.0, 0.0
    # Max internal min
    max_minimum = max(
        min_individual_iou, min_or_chain_iou, min_and_chain_iou, min_and_not_chain_iou,
        min_comb_and_or_chain_iou, min_comb_or_andnot_chain_iou
    )


    max_results = []
    for ops, max_iou, min_iou in zip([[], ['OR'], ['AND'], ['AND', 'NOT'], ['NOT'], ['AND','OR'],['AND', 'OR', 'NOT'], ['OR', 'NOT']],
                    [max_individual_iou, max_or_chain_iou, max_and_chain_iou, max_and_and_not_chain_iou, max_and_not_chain_iou, max_comb_and_or_chain_iou, max_every_chain_iou, max_comb_or_andnot_chain_iou],
                    [min_individual_iou, min_or_chain_iou, min_and_chain_iou, min_and_and_not_chain_iou, min_and_not_chain_iou, min_comb_and_or_chain_iou, min_every_chain_iou, min_comb_or_andnot_chain_iou]):
        if max_iou > 0 and max_iou >= max_minimum:
            max_results.append((ops, max_iou))
    # print()
    # print(f"Quantities time : {quantities_esti_time - start_time:.4f}s")
    # print(f"Individual estimation time: {individual_time - start_time:.4f}s")
    # print(f"Or chain estimation time: {or_chain_time - individual_time:.4f}s")
    # print(f"And chain estimation time: {and_chain_time - or_chain_time:.4f}s")
    # print(f"And-Not chain estimation time: {and_not_chain_time - and_chain_time:.4f}s")
    # print(f"Comb-And-OR chain estimation time: {comb_and_or_chain_time - and_not_chain_time:.4f}s")
    # print(f"Comb-OR-AndNot chain estimation time: {comb_or_andnot_chain_time - comb_and_or_chain_time:.4f}s")
    # print(f"Max min estimation time: {max_minimum_time - comb_or_andnot_chain_time:.4f}s")
    # print(f"Results time: {results_time - max_minimum_time:.4f}s")
    # print(f"Total time: {results_time - start_time:.4f}s")
    # print()
    # if isinstance(label, F.Leaf) and label.val == 0:
    #     print()
    #     print("Label 0 results:")
    #     print("Ops", ops_to_expand)
    #     print(min_individual_iou, min_or_chain_iou, min_and_chain_iou, min_and_not_chain_iou,
    #     min_comb_and_or_chain_iou, min_comb_or_andnot_chain_iou)
    #     print(max_results)
    # elif 0 in label.get_vals():
    #     print()
    #     print(f"Label {label} results:")
    #     print("Ops", ops_to_expand)
    #     print(min_individual_iou, min_or_chain_iou, min_and_chain_iou, min_and_not_chain_iou,
    #     min_comb_and_or_chain_iou, min_comb_or_andnot_chain_iou)
    #     print(max_results)
    #     if len(label) == 3:
    #         exit()
    return max_results, max_minimum

def numpy_estimate_optimal_label_iou(node, label_mapping, heuristic_info, max_improvement, num_hits, max_size_mask, max_length, fpd_matrix):
    """
    Estimate the IoU of a label using the optimal heuristic.
    
    Args:
        label (F.Formula): The label to estimate the IoU for.
        heuristic_info (tuple): The information to compute the heuristic.
        
    Returns:
        float: Estimated IoU of the label.
    """
    seg_quantities, neuron_quantities, concepts_quantities = heuristic_info
    neuron_unique, neuron_common, neuron_coverable_mask, neuron_coverable, neuron_sum  = neuron_quantities
    #common_intersection, unique_intersection, common_extras, unique_extras, uncovered = concepts_quantities
    
    label = node[2]
    ops_to_expand = node[1]
    label_node = (None, ops_to_expand, label)
    label_common_intersection, label_unique_intersection, label_common_extras, label_unique_extras, label_uncovered = get_esti_quantities(
        label_node, label_mapping, heuristic_info, max_size_mask, fpd_matrix=fpd_matrix)
    if label_common_intersection is None:
        # Label discarded at the previous step
        return [([], 0.0)], 0.0
    start_time = time.time()
    # Individual estimation
    max_individual_iou, min_individual_iou = numpy_individual_estimation(label,
        label_common_intersection, label_unique_intersection,
        label_common_extras, label_unique_extras, label_uncovered,
        neuron_unique, neuron_common, neuron_coverable, neuron_sum,
        max_improvement, num_hits, max_size_mask, max_length,
    )
    individual_time = time.time()
    if 1 in label.get_vals() and len(label) == 1:
        print(f"Individual estimation time: {individual_time - start_time:.4f} seconds")
    if len(label) == max_length:
        # If the label is already at maximum length, we cannot estimate any chain
        return [([], max_individual_iou)], min_individual_iou

    if ops_to_expand is None or ops_to_expand[0] == 'OR':
        max_or_chain_iou, min_or_chain_iou = numpy_or_chain_estimation(label,
            label_common_intersection, label_unique_intersection,
            label_common_extras, label_unique_extras, label_uncovered,
            neuron_unique, neuron_common, neuron_coverable, neuron_sum,
            max_improvement, num_hits, max_size_mask, max_length
        )
        or_chain_time = time.time()
        if 1 in label.get_vals() and len(label) == 1:
            print(f"OR chain estimation time: {or_chain_time - individual_time:.4f} seconds")
    else:
        max_or_chain_iou, min_or_chain_iou = 0.0, 0.0

    # This covers AND chains
    if ops_to_expand is None or (len(ops_to_expand) == 1 and (ops_to_expand[0] == 'AND' or ops_to_expand[0] == 'NOT')) :
        max_and_chain_iou, min_and_chain_iou = numpy_and_chain_estimation(label,
            label_common_intersection, label_unique_intersection,
            label_common_extras, label_unique_extras, label_uncovered,
            neuron_unique, neuron_common, neuron_coverable, neuron_sum,
            max_improvement, num_hits, max_size_mask, max_length
        )
        and_chain_time = time.time()
        if 1 in label.get_vals() and len(label) == 1:
            print(f"And chain estimation time: {and_chain_time - individual_time:.4f} seconds")
    else:
        max_and_chain_iou, min_and_chain_iou = 0.0, 0.0


    if ops_to_expand is None or (len(ops_to_expand) == 2 and (ops_to_expand[0] == 'AND' and ops_to_expand[1] == 'NOT')):
        max_and_not_chain_iou, min_and_not_chain_iou = numpy_and_not_chain_estimation(label,
            label_common_intersection, label_unique_intersection,
            label_common_extras, label_unique_extras, label_uncovered,
            neuron_unique, neuron_common, neuron_coverable, neuron_sum,
            max_improvement, num_hits, max_size_mask, max_length
        )
        and_not_chain_time = time.time()
        if 1 in label.get_vals() and len(label) == 1:
            print(f"And-Not chain estimation time: {and_not_chain_time - individual_time:.4f} seconds")
    else:
        max_and_not_chain_iou, min_and_not_chain_iou = 0.0, 0.0


    # Compute them only if there is space for 2-chains
    if max_length - len(label) > 1 and (ops_to_expand is None or len(ops_to_expand) == 2):
        # This covers AND-NOT chains
        max_and_and_not_chain_iou = max_and_chain_iou
        min_and_and_not_chain_iou = min_and_chain_iou
        start_time = time.time()

        if ops_to_expand is None or (ops_to_expand[0] == 'AND' and ops_to_expand[1] == 'OR'):
        # This covers both AND-OR chaing and everything chain
            max_comb_and_or_chain_iou, min_comb_and_or_chain_iou = numpy_comb_and_or_chain_estimation(label,
                label_common_intersection, label_unique_intersection,
                label_common_extras, label_unique_extras, label_uncovered,
                neuron_unique, neuron_common, neuron_coverable, neuron_sum,
                max_improvement, num_hits, max_size_mask, max_length
            )
            and_or_chain_time = time.time()
            if 1 in label.get_vals() and len(label) == 1:
                print(f"Comb-AND-OR chain estimation time: {and_or_chain_time - start_time:.4f} seconds")
        else:
            max_comb_and_or_chain_iou, min_comb_and_or_chain_iou = 0.0, 0.0

        if ops_to_expand is None or ('OR' in ops_to_expand and 'NOT' in ops_to_expand):
            max_comb_or_andnot_chain_iou, min_comb_or_andnot_chain_iou = numpy_comb_or_andnot_chain_estimation(label,
                label_common_intersection, label_unique_intersection,
                label_common_extras, label_unique_extras, label_uncovered,
                neuron_unique, neuron_common, neuron_coverable, neuron_sum,
                max_improvement, num_hits, max_size_mask, max_length
            )
            if 1 in label.get_vals() and len(label) == 1:
                print(f"Comb-OR-ANDNOT chain estimation time: {time.time() - start_time:.4f} seconds")
        else:
            max_comb_or_andnot_chain_iou, min_comb_or_andnot_chain_iou = 0.0, 0.0

    else:
        max_and_and_not_chain_iou, min_and_and_not_chain_iou = 0.0, 0.0
        max_comb_and_or_chain_iou, min_comb_and_or_chain_iou = 0.0, 0.0
        max_comb_or_andnot_chain_iou, min_comb_or_andnot_chain_iou = 0.0, 0.0


    
    # Greater chains
    if max_length - len(label) > 2 and (ops_to_expand is None or len(ops_to_expand) > 2):
        # This covers both everything chain
        max_every_chain_iou = max_comb_and_or_chain_iou
        min_every_chain_iou = min_comb_and_or_chain_iou
    else:
        max_every_chain_iou, min_every_chain_iou = 0.0, 0.0
    # Max internal min
    max_minimum = max(
        min_individual_iou, min_or_chain_iou, min_and_chain_iou, min_and_not_chain_iou,
        min_comb_and_or_chain_iou, min_comb_or_andnot_chain_iou
    )


    max_results = []
    for ops, max_iou, min_iou in zip([[], ['OR'], ['AND'], ['AND', 'NOT'], ['NOT'], ['AND','OR'],['AND', 'OR', 'NOT'], ['OR', 'NOT']],
                    [max_individual_iou, max_or_chain_iou, max_and_chain_iou, max_and_and_not_chain_iou, max_and_not_chain_iou, max_comb_and_or_chain_iou, max_every_chain_iou, max_comb_or_andnot_chain_iou],
                    [min_individual_iou, min_or_chain_iou, min_and_chain_iou, min_and_and_not_chain_iou, min_and_not_chain_iou, min_comb_and_or_chain_iou, min_every_chain_iou, min_comb_or_andnot_chain_iou]):
        if max_iou > 0 and max_iou >= max_minimum:
            max_results.append((ops, max_iou))
    return max_results, max_minimum


def optim_estimate_optimal_label_iou(node, label_mapping, heuristic_info, max_improvement, num_hits, max_size_mask, max_length, fpd_matrix):
    """
    Estimate the IoU of a label using the optimal heuristic.
    
    Args:
        label (F.Formula): The label to estimate the IoU for.
        heuristic_info (tuple): The information to compute the heuristic.
        
    Returns:
        float: Estimated IoU of the label.
    """
    seg_quantities, neuron_quantities, concepts_quantities = heuristic_info
    neuron_unique, neuron_common, neuron_coverable_mask, neuron_coverable, neuron_sum  = neuron_quantities
    #common_intersection, unique_intersection, common_extras, unique_extras, uncovered = concepts_quantities
    
    
    label = node[2]
    ops_to_expand = node[1]
    label_node = (None, ops_to_expand, label)
    label_common_intersection, label_unique_intersection, label_common_extras, label_unique_extras, label_uncovered = get_esti_quantities(
        label_node, label_mapping, heuristic_info, max_size_mask, fpd_matrix=fpd_matrix)
    if label_common_intersection is None:
        # Label discarded at the previous step
        return [([], 0.0)], 0.0
    
    # NEWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
    # Aux variables
    label_len = len(label)
    k = max_length - label_len # Missing length   
    TOP_INDEX = 0
    BOTTOM_INDEX = 1

    # Unpack improvement information
    max_common_intersection, min_common_intersection = get_max_min_quantity(label_common_intersection)
    max_unique_intersection, min_unique_intersection = get_max_min_quantity(label_unique_intersection)
    max_common_extras, min_common_extras = get_max_min_quantity(label_common_extras)
    max_unique_extras, min_unique_extras = get_max_min_quantity(label_unique_extras)
    improv_common_intersection, improv_unique_intersection, improv_common_extras, improv_unique_extras, improv_uncovered = max_improvement
    zero_vector = torch.zeros_like(max_common_intersection)

    neuron_activation = neuron_common + neuron_unique
    max_label_intersection = max_common_intersection + max_unique_intersection
    min_label_intersection = min_common_intersection + min_unique_intersection
    max_label_extras = max_common_extras + max_unique_extras
    min_label_extras = min_common_extras + min_unique_extras
    
    topk_common_intersection = improv_common_intersection[k][TOP_INDEX]
    one_top_common_extras = improv_common_extras[k][TOP_INDEX]
    max_top_intersection = improv_common_intersection[k][TOP_INDEX] + improv_unique_intersection[k][TOP_INDEX]
    max_top_extras = improv_common_extras[k][TOP_INDEX] + improv_unique_extras[k][TOP_INDEX]

    
    one_top_common_intersection = improv_common_intersection[0][TOP_INDEX]
    top1_common_extras = improv_common_extras[0][TOP_INDEX]
    bottom1_common_extras = improv_common_extras[0][BOTTOM_INDEX]
    bottom1_common_intersection = improv_common_intersection[0][BOTTOM_INDEX]
    bott1_unique_intersection = improv_unique_intersection[0][BOTTOM_INDEX]

    min_bottom_intersection = improv_common_intersection[0][BOTTOM_INDEX] + improv_unique_intersection[0][BOTTOM_INDEX]
    min_bottom_extras = improv_common_extras[0][BOTTOM_INDEX] + improv_unique_extras[0][BOTTOM_INDEX]

    min_added_extras = torch.maximum(zero_vector, min_bottom_extras - max_label_extras)
    sum_min_unique_extras = min_unique_extras.sum()
    sum_min_common_extras = min_common_extras.sum()
    sum_max_unique_extras = max_unique_extras.sum()
    sum_max_common_extras = max_common_extras.sum()
    sum_min_unique_intersection = min_unique_intersection.sum()

    space_for_extras = max_size_mask - neuron_sum


    # We discard the labels that cannot increase the IoU
    if max_label_intersection.sum() == 0:
        return [([], 0.0)], 0.0
    
    ########################## Individual  ##########################
    # Max IoU
    max_intersection = torch.minimum(
        max_label_intersection,
        neuron_activation
    )
    individual_min_union = num_hits + sum_min_unique_extras + sum_min_common_extras
    max_individual_iou = max_intersection.sum() / individual_min_union

    # Min IoU
    individual_min_intersection = torch.minimum(min_common_intersection+ min_unique_intersection, neuron_activation)
    individual_max_union = num_hits + sum_max_unique_extras + sum_max_common_extras
    min_individual_iou = individual_min_intersection.sum() / individual_max_union

    if len(label) == max_length:
        # If the label is already at maximum length, we cannot estimate any chain
        return [([], max_individual_iou)], min_individual_iou




    ########################### OR Chain Estimation ##########################
    if ops_to_expand is None or ops_to_expand[0] == 'OR':
    # Max IoU
        or_chain_max_intersection = torch.minimum(max_label_intersection + max_top_intersection, neuron_coverable)
        or_chain_min_union = torch.clamp(neuron_sum + min_label_extras + min_added_extras, max=max_size_mask)
        max_or_chain_iou = or_chain_max_intersection.sum() / or_chain_min_union.sum()

        # Min IoU
        or_chain_min_intersection = torch.maximum(min_label_intersection, min_bottom_intersection)
        or_chain_max_union = torch.clamp(neuron_sum + max_label_extras + max_top_extras, max=max_size_mask)
        min_or_chain_iou = or_chain_min_intersection.sum() / or_chain_max_union.sum()
    else:
        max_or_chain_iou, min_or_chain_iou = 0.0, 0.0

    ########################### AND Chain Estimation ##########################
    
    if ops_to_expand is None or (len(ops_to_expand) == 1 and (ops_to_expand[0] == 'AND' or ops_to_expand[0] == 'NOT')) :
        # MaxIoU
        and_chain_max_intersection = torch.minimum(max_common_intersection, one_top_common_intersection)
        and_chain_min_union = torch.clamp(neuron_sum + torch.maximum(zero_vector,
            max_common_intersection +  bottom1_common_extras - (space_for_extras - max_unique_extras),
                ), max=max_size_mask)
        max_and_chain_iou = and_chain_max_intersection.sum() / and_chain_min_union.sum()

        # MinIoU
        and_chain_min_intersection = torch.minimum(zero_vector, min_common_intersection + bottom1_common_intersection - neuron_common).sum()
        and_chain_max_union = num_hits + torch.minimum(max_common_extras, one_top_common_extras).sum()
        min_and_chain_iou = and_chain_min_intersection / and_chain_max_union
    else:
        max_and_chain_iou, min_and_chain_iou = 0.0, 0.0

    ########################### AND-NOT Chain Estimation ##########################    
    if ops_to_expand is None or (len(ops_to_expand) == 2 and (ops_to_expand[0] == 'AND' and ops_to_expand[1] == 'NOT')):
        # Max IoU
        and_not_max_intersection = torch.minimum(max_unique_intersection + torch.minimum(
            max_common_intersection, improv_uncovered[0][TOP_INDEX]
        ), neuron_coverable)
        and_not_min_union = num_hits + sum_min_unique_extras
        max_and_not_chain_iou = and_not_max_intersection.sum() / and_not_min_union

        # Min IoU
        and_not_max_union = num_hits + sum_max_unique_extras + sum_max_common_extras
        min_and_not_chain_iou = sum_min_unique_intersection / and_not_max_union
    else:
        max_and_not_chain_iou, min_and_not_chain_iou = 0.0, 0.0


    # Compute them only if there is space for 2-chains
    if max_length - len(label) > 1:
        # This covers AND-NOT chains
        max_and_and_not_chain_iou = max_and_chain_iou
        min_and_and_not_chain_iou = min_and_chain_iou

        ############################### Comb-And-OR Chain Estimation ##########################

        if ops_to_expand is None or (len(ops_to_expand) == 3 and (ops_to_expand[0] == 'AND' and ops_to_expand[1] == 'OR' and ops_to_expand[2] == 'NOT')):

            if max_common_intersection.sum() == 0:
                max_comb_and_or_chain_iou, min_comb_and_or_chain_iou = 0.0, 0.0
            else:
                # Max IoU
                comb_and_or_chain_max_intersection = torch.minimum(max_common_intersection + topk_common_intersection, neuron_common)
                comb_and_or_chain_min_union = torch.clamp(neuron_sum + torch.maximum(zero_vector,
                                                    min_common_extras + bottom1_common_extras -
                                                        (space_for_extras - min_unique_extras)),
                                                        max=max_size_mask)
                max_comb_and_or_chain_iou = comb_and_or_chain_max_intersection.sum() / comb_and_or_chain_min_union.sum()

                # Min IoU
                comb_and_or_chain_min_intersection = torch.minimum(min_common_intersection, bottom1_common_intersection)
                comb_and_or_chain_max_union = num_hits + torch.minimum(max_common_extras, top1_common_extras).sum()
                min_comb_and_or_chain_iou = comb_and_or_chain_min_intersection.sum() / comb_and_or_chain_max_union
        
        else:
            max_comb_and_or_chain_iou, min_comb_and_or_chain_iou = 0.0, 0.0

        if ops_to_expand is None or (len(ops_to_expand) == 2 and (ops_to_expand[0] == 'OR' and ops_to_expand[1] == 'NOT')):
            
            # Max IoU
            comb_or_andnot_chain_max_intersection = torch.minimum(
                max_label_intersection + topk_common_intersection - bott1_unique_intersection,
                neuron_coverable
            )
            comb_or_andnot_chain_min_union = num_hits + sum_min_unique_extras
            max_comb_or_andnot_chain_iou = comb_or_andnot_chain_max_intersection.sum() / comb_or_andnot_chain_min_union
            # Min IoU
            comb_or_andnot_chain_min_intersection = min_unique_intersection
            comb_or_andnot_chain_max_union = num_hits + sum_max_unique_extras + sum_max_common_extras
            min_comb_or_andnot_chain_iou = comb_or_andnot_chain_min_intersection.sum() / comb_or_andnot_chain_max_union
  
            max_comb_or_andnot_chain_iou, min_comb_or_andnot_chain_iou = comb_or_andnot_chain_estimation(label,
                label_common_intersection, label_unique_intersection,
                label_common_extras, label_unique_extras, label_uncovered,
                neuron_unique, neuron_common, neuron_coverable, neuron_sum,
                max_improvement, num_hits, max_size_mask, max_length
            )
        else:
            max_comb_or_andnot_chain_iou, min_comb_or_andnot_chain_iou = 0.0, 0.0

    else:
        max_and_and_not_chain_iou, min_and_and_not_chain_iou = 0.0, 0.0
        max_comb_and_or_chain_iou, min_comb_and_or_chain_iou = 0.0, 0.0
        max_comb_or_andnot_chain_iou, min_comb_or_andnot_chain_iou = 0.0, 0.0


    
    # Greater chains
    if max_length - len(label) > 2:
        # This covers both everything chain
        max_every_chain_iou = max_comb_and_or_chain_iou
        min_every_chain_iou = min_comb_and_or_chain_iou
    else:
        max_every_chain_iou, min_every_chain_iou = 0.0, 0.0
    # Max internal min
    max_minimum = max(
        min_individual_iou, min_or_chain_iou, min_and_chain_iou, min_and_not_chain_iou,
        min_comb_and_or_chain_iou, min_comb_or_andnot_chain_iou
    )


    max_results = []
    for ops, max_iou, min_iou in zip([[], ['OR'], ['AND'], ['AND', 'NOT'], ['NOT'], ['AND','OR'],['AND', 'OR', 'NOT'], ['OR', 'NOT']],
                    [max_individual_iou, max_or_chain_iou, max_and_chain_iou, max_and_and_not_chain_iou, max_and_not_chain_iou, max_comb_and_or_chain_iou, max_every_chain_iou, max_comb_or_andnot_chain_iou],
                    [min_individual_iou, min_or_chain_iou, min_and_chain_iou, min_and_and_not_chain_iou, min_and_not_chain_iou, min_comb_and_or_chain_iou, min_every_chain_iou, min_comb_or_andnot_chain_iou]):
        if max_iou > 0 and max_iou >= max_minimum:
            max_results.append((ops, max_iou))

    return max_results, max_minimum


def update_optimal_label_iou(label, operators, label_mapping, heuristic_info, max_improvement, fpd_matrix, num_hits, max_size_mask, max_length, debug=False):
    """
    Estimate the IoU of a label using the optimal heuristic.
    
    Args:
        label (F.Formula): The label to estimate the IoU for.
        heuristic_info (tuple): The information to compute the heuristic.
        
    Returns:
        float: Estimated IoU of the label.
    """
    seg_quantities, neuron_quantities, concepts_quantities = heuristic_info
    neuron_unique, neuron_common, neuron_coverable_mask, neuron_coverable, neuron_sum  = neuron_quantities
    common_intersection, unique_intersection, common_extras, unique_extras, uncovered = concepts_quantities
 
    # index_label = label_mapping[label]
    # label_common_intersection = common_intersection[index_label]
    # label_unique_intersection = unique_intersection[index_label]
    # label_common_extras = common_extras[index_label]
    # label_unique_extras = unique_extras[index_label]
    # label_uncovered = uncovered[index_label]

    label_node = (None, None, label)
    label_common_intersection, label_unique_intersection, label_common_extras, label_unique_extras, label_uncovered = get_esti_quantities(
        label_node, label_mapping, heuristic_info, max_size_mask, fpd_matrix=fpd_matrix)

    if label_common_intersection is None:
        # Label discarded at the previous step
        return [([], 0.0)], 0.0
    if len(operators) == 0:
        # Individual estimation
        new_max, new_min = individual_estimation(label,
            label_common_intersection, label_unique_intersection,
            label_common_extras, label_unique_extras, label_uncovered,
            neuron_unique, neuron_common, neuron_coverable, neuron_sum,
            max_improvement, num_hits, max_size_mask, max_length, debug=debug
        )

    elif len(label) == max_length:
        # If the label is already at maximum length, we cannot estimate any chain
        new_max, new_min = individual_estimation(label,
            label_common_intersection, label_unique_intersection,
            label_common_extras, label_unique_extras, label_uncovered,
            neuron_unique, neuron_common, neuron_coverable, neuron_sum,
            max_improvement, num_hits, max_size_mask, max_length, debug=debug
        )
    elif len(operators) == 1 and operators[0] == 'OR':
        new_max, new_min = or_chain_estimation(label,
            label_common_intersection, label_unique_intersection,
            label_common_extras, label_unique_extras, label_uncovered,
            neuron_unique, neuron_common, neuron_coverable, neuron_sum,
            max_improvement, num_hits, max_size_mask, max_length, debug=debug
        )
    elif len(operators) == 2 and operators[0] == 'AND' and operators[1] == 'NOT':
        # This covers both AND and AND-AND NOT chains
        new_max, new_min = and_chain_estimation(label,
            label_common_intersection, label_unique_intersection,
            label_common_extras, label_unique_extras, label_uncovered,
            neuron_unique, neuron_common, neuron_coverable, neuron_sum,
            max_improvement, num_hits, max_size_mask, max_length, debug=debug
        )
    elif len(operators) == 1 and operators[0] == 'NOT':
        new_max, new_min = and_not_chain_estimation(label,
            label_common_intersection, label_unique_intersection,
            label_common_extras, label_unique_extras, label_uncovered,
            neuron_unique, neuron_common, neuron_coverable, neuron_sum,
            max_improvement, num_hits, max_size_mask, max_length, debug=debug
        )
    elif len(operators) == 3 and operators[0] == 'AND' and operators[1] == 'OR' and operators[2] == 'NOT':
        # This covers both AND-OR chaing and everything chain
        new_max, new_min = comb_and_or_chain_estimation(label,
            label_common_intersection, label_unique_intersection,
            label_common_extras, label_unique_extras, label_uncovered,
            neuron_unique, neuron_common, neuron_coverable, neuron_sum,
            max_improvement, num_hits, max_size_mask, max_length, debug=debug
        )
    elif len(operators) == 2 and operators[0] == 'OR' and operators[1] == 'NOT':
        new_max, new_min = comb_or_andnot_chain_estimation(label,
            label_common_intersection, label_unique_intersection,
            label_common_extras, label_unique_extras, label_uncovered,
            neuron_unique, neuron_common, neuron_coverable, neuron_sum,
            max_improvement, num_hits, max_size_mask, max_length, debug=debug
        )
    else:
        raise ValueError(f"Unknown operators: {operators}")

    
    return new_max, new_min

def numpy_update_optimal_label_iou(label, operators, label_mapping, heuristic_info, max_improvement, fpd_matrix, num_hits, max_size_mask, max_length, debug=False):
    seg_quantities, neuron_quantities, concepts_quantities = heuristic_info
    neuron_unique, neuron_common, neuron_coverable_mask, neuron_coverable, neuron_sum  = neuron_quantities
    common_intersection, unique_intersection, common_extras, unique_extras, uncovered = concepts_quantities

    label_node = (None, None, label)
    label_common_intersection, label_unique_intersection, label_common_extras, label_unique_extras, label_uncovered = get_esti_quantities(
        label_node, label_mapping, heuristic_info, max_size_mask, fpd_matrix=fpd_matrix)

    if label_common_intersection is None:
        # Label discarded at the previous step
        return [([], 0.0)], 0.0
    if len(operators) == 0:
        # Individual estimation
        new_max, new_min = numpy_individual_estimation(label,
            label_common_intersection, label_unique_intersection,
            label_common_extras, label_unique_extras, label_uncovered,
            neuron_unique, neuron_common, neuron_coverable, neuron_sum,
            max_improvement, num_hits, max_size_mask, max_length, debug=debug
        )

    elif len(label) == max_length:
        # If the label is already at maximum length, we cannot estimate any chain
        new_max, new_min = numpy_individual_estimation(label,
            label_common_intersection, label_unique_intersection,
            label_common_extras, label_unique_extras, label_uncovered,
            neuron_unique, neuron_common, neuron_coverable, neuron_sum,
            max_improvement, num_hits, max_size_mask, max_length, debug=debug
        )
    elif len(operators) == 1 and operators[0] == 'OR':
        new_max, new_min = numpy_or_chain_estimation(label,
            label_common_intersection, label_unique_intersection,
            label_common_extras, label_unique_extras, label_uncovered,
            neuron_unique, neuron_common, neuron_coverable, neuron_sum,
            max_improvement, num_hits, max_size_mask, max_length, debug=debug
        )
    elif len(operators) == 1 and operators[0] == 'NOT':
        new_max, new_min = numpy_and_not_chain_estimation(label,
            label_common_intersection, label_unique_intersection,
            label_common_extras, label_unique_extras, label_uncovered,
            neuron_unique, neuron_common, neuron_coverable, neuron_sum,
            max_improvement, num_hits, max_size_mask, max_length, debug=debug
        )
    elif len(operators) == 2 and operators[0] == 'AND' and operators[1] == 'NOT':
        # This covers both AND and AND-AND NOT chains
        new_max, new_min = numpy_and_chain_estimation(label,
            label_common_intersection, label_unique_intersection,
            label_common_extras, label_unique_extras, label_uncovered,
            neuron_unique, neuron_common, neuron_coverable, neuron_sum,
            max_improvement, num_hits, max_size_mask, max_length, debug=debug
        )
    elif len(operators) == 2 and operators[0] == 'OR' and operators[1] == 'NOT':
        new_max, new_min = numpy_comb_or_andnot_chain_estimation(label,
            label_common_intersection, label_unique_intersection,
            label_common_extras, label_unique_extras, label_uncovered,
            neuron_unique, neuron_common, neuron_coverable, neuron_sum,
            max_improvement, num_hits, max_size_mask, max_length, debug=debug
        )
    elif len(operators) == 2 and operators[0] == 'AND' and operators[1] == 'OR':
        # This covers both AND-OR chaing and everything chain
        new_max, new_min = numpy_comb_and_or_chain_estimation(label,
            label_common_intersection, label_unique_intersection,
            label_common_extras, label_unique_extras, label_uncovered,
            neuron_unique, neuron_common, neuron_coverable, neuron_sum,
            max_improvement, num_hits, max_size_mask, max_length, debug=debug
        )
    elif len(operators) == 3 and operators[0] == 'AND' and operators[1] == 'OR' and operators[2] == 'NOT':
        # This covers both AND-OR chaing and everything chain
        new_max, new_min = numpy_comb_and_or_chain_estimation(label,
            label_common_intersection, label_unique_intersection,
            label_common_extras, label_unique_extras, label_uncovered,
            neuron_unique, neuron_common, neuron_coverable, neuron_sum,
            max_improvement, num_hits, max_size_mask, max_length, debug=debug
        )

    else:
        raise ValueError(f"Unknown operators: {operators}")

    return new_max, new_min

def estimate_label_info(label, left_quantities, right_quantities, neuron_unique, neuron_common, neuron_coverable, neuron_sum, max_size_mask, label_relation):
    """
    Estimate the label information for a given label based on its left and right quantities.

    Args:
        label (F.Formula): The label for which to estimate the label information.
        left_quantities (tuple): Quantities from the left child of the label.
        right_quantities (tuple): Quantities from the right child of the label.

    Returns:
        tuple: Estimated quantities for the label.
    """
    left_common_intersection, left_unique_intersection, left_common_extras, left_unique_extras, left_uncovered = left_quantities
    max_left_common_intersection, min_left_common_intersection = get_max_min_quantity(left_common_intersection)
    max_left_unique_intersection, min_left_unique_intersection = get_max_min_quantity(left_unique_intersection)
    max_left_common_extras, min_left_common_extras = get_max_min_quantity(left_common_extras)
    max_left_unique_extras, min_left_unique_extras = get_max_min_quantity(left_unique_extras)
    max_left_uncovered, min_left_uncovered = get_max_min_quantity(left_uncovered)
    right_common_intersection, right_unique_intersection, right_common_extras, right_unique_extras, right_uncovered = right_quantities
    max_right_common_intersection, min_right_common_intersection = get_max_min_quantity(right_common_intersection)
    max_right_unique_intersection, min_right_unique_intersection = get_max_min_quantity(right_unique_intersection)
    max_right_common_extras, min_right_common_extras = get_max_min_quantity(right_common_extras)
    max_right_unique_extras, min_right_unique_extras = get_max_min_quantity(right_unique_extras)
    max_right_uncovered, min_right_uncovered = get_max_min_quantity(right_uncovered)

    if isinstance(label, F.Or):
        max_unique_intersection = max_left_unique_intersection + max_right_unique_intersection # I_max^u(L) + I_max^u(c)
        min_unique_intersection = min_left_unique_intersection + min_right_unique_intersection # I_min^u(L) + I_min^u(c)

        min_common_intersection = torch.maximum(min_left_common_intersection, min_right_common_intersection) # max(I_min^c(L), I_min^c(c))
        max_common_intersection = torch.minimum(
            torch.maximum(
                max_left_common_intersection + torch.minimum(
                    max_left_uncovered - min_right_unique_intersection,
                    max_right_common_intersection),
                max_right_common_intersection + torch.minimum(
                    max_right_uncovered - min_left_unique_intersection,
                    max_left_common_intersection)),
                neuron_coverable - min_left_unique_intersection - min_right_unique_intersection)

        if ((torch.all(max_unique_intersection == max_left_unique_intersection) and \
              torch.all(max_common_intersection == max_left_common_intersection)) or \
            (torch.all(max_unique_intersection == max_right_unique_intersection) and \
            torch.all(max_common_intersection == max_right_common_intersection))):
            # If one of the two side does not contribute to the intersection, we can discard this formula
            return None, None, None, None, None

        min_uncovered  = torch.clamp(
            min_left_uncovered - max_right_common_intersection - max_right_unique_intersection, min=0) # max(0, U_min^L - I_max^c(c) - I_max^u(c))
        max_uncovered = torch.minimum(
            max_left_uncovered, max_right_uncovered) # min(U_max^L, U_max^c)

        min_unique_extras = min_left_unique_extras + min_right_unique_extras # E_min^u(L) + E_min^u(c)
        max_unique_extras = max_left_unique_extras + max_right_unique_extras # E_max^u(L) + E_max^u(c)

        min_common_extras = torch.maximum(min_left_common_extras, min_right_common_extras) # max(E_min^c(L), E_min^c(c))
        max_common_extras = torch.minimum(
            max_size_mask - neuron_sum - min_left_unique_extras - min_right_unique_extras,
            max_left_common_extras + max_right_common_extras
        ) # min(max_size_mask - N^u - N^c - E_min^u(L) - E_min^u(c), E_max^c(L) + E_max^c(c))
    elif isinstance(label, F.And) and isinstance(label.right, F.Not):
        max_unique_intersection = max_left_unique_intersection
        min_unique_intersection = min_left_unique_intersection

        min_common_intersection = torch.clamp(
           min_left_common_intersection + min_right_uncovered - neuron_coverable, min=0
        ) # max(0, I_min^c(L) + U_min^c - N^u - N^c)
        max_common_intersection = torch.minimum(
            max_right_uncovered, max_left_common_intersection
        ) # min(U_max^c, I_max^c(L))

        min_uncovered = torch.maximum(
            min_left_uncovered, min_right_common_intersection + min_right_unique_intersection
        ) # max(U_min^L, I_min^c(c) + I_min^u(c))
        max_uncovered = neuron_coverable - torch.clamp(
           max_left_common_intersection + max_right_uncovered - neuron_coverable, min=0
        ) # N^u + N^c - max(0, I_max^c(L) + U_max^c - N^u - N^c)

        min_unique_extras = min_left_unique_extras # E_min^u(L)
        max_unique_extras = max_left_unique_extras # E_max^u(L)

        min_common_extras = torch.clamp(
            min_left_common_extras + (max_size_mask - neuron_sum - max_right_unique_extras - max_right_common_extras) - \
            (max_size_mask - neuron_sum),
            min=0
        )
        max_common_extras = max_left_common_extras
    elif isinstance(label, F.And):
        min_unique_extras = torch.zeros_like(min_left_unique_extras)
        max_unique_extras =  torch.zeros_like(min_left_unique_extras)

        min_common_extras = torch.clamp(
            min_left_common_extras + min_right_common_extras - (max_size_mask - neuron_sum - max_left_unique_extras - max_right_unique_extras)
        , min=0
        ) # max(0, E^c(L) + E^c(c) - (max_size_mask - N^u - N^c))
        max_common_extras = torch.minimum(
            max_left_common_extras,
            max_right_common_extras
        ) # min(E^c(L), E^c(c))
        if (torch.all(max_unique_extras == max_left_unique_extras) and \
           torch.all(max_common_extras == max_left_common_extras)) or \
           (torch.all(max_unique_extras == max_right_unique_extras) and \
           torch.all(max_common_extras == max_right_common_extras)):
            # If one of the two side does not contribute to the intersection, we can discard this formula
            return None, None, None, None, None
        
        max_unique_intersection = torch.zeros_like(min_left_unique_intersection)
        min_unique_intersection = torch.zeros_like(min_left_unique_intersection)
        
        min_common_intersection = torch.clamp(
            min_left_common_intersection + min_right_common_intersection - neuron_coverable,
            min=0    ) # max(0, I_min^c(L) + I_min^c(c) - N^u - N^c)
        max_common_intersection = torch.minimum(
            max_left_common_intersection, max_right_common_intersection) # min(I_max^c(L), I_max^c(c))
        
        min_uncovered = torch.maximum(
            min_left_uncovered, min_right_uncovered) # max(U_min^L, U_min^c)
        max_uncovered = neuron_coverable - torch.clamp(
            min_left_common_intersection + min_right_common_intersection - neuron_coverable, min=0
        ) # N^u + N^c - max(0, I_min^c(L) + I_min^c(c) - N^u - N^c)

    else:
        raise ValueError(f"Unknown label type: {type(label)}")


    return (max_common_intersection, min_common_intersection), (max_unique_intersection, min_unique_intersection), \
           (max_common_extras, min_common_extras), \
           (max_unique_extras, min_unique_extras), \
           (max_uncovered, min_uncovered)


def numpy_estimate_label_info(label, left_quantities, right_quantities, neuron_unique, neuron_common, neuron_coverable, neuron_sum, max_size_mask, label_relation):

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
    
    #left_common_intersection, left_unique_intersection, left_common_extras, left_unique_extras, left_uncovered = left_quantities
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
    #right_common_intersection, right_unique_intersection, right_common_extras, right_unique_extras, right_uncovered = right_quantities
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

        if ((np.all(max_unique_intersection == max_left_unique_intersection) and \
            np.all(max_common_intersection == max_left_common_intersection)) or \
            (np.all(max_unique_intersection == max_right_unique_intersection) and \
            np.all(max_common_intersection == max_right_common_intersection))):
            # If one of the two side does not contribute to the intersection, we can discard this formula
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
        if (np.all(max_unique_extras == max_left_unique_extras) and \
        np.all(max_common_extras == max_left_common_extras)) or \
        (np.all(max_unique_extras == max_right_unique_extras) and \
        np.all(max_common_extras == max_right_common_extras)):
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


import time

def probe_disjointment(f, fpd_matrix):
    """
    Function to return a mask for a given formula.
    Args:
        f (src.formula.Formula): formula.
        masks (list): list of masks.
        optional_masks (dict): dictionary of additional masks (beam masks).
    Returns:
        Formula's Mask.
    """
    if isinstance(f, F.Leaf):
        return True
    elif isinstance(f, int):
        return True
    elif isinstance(f, F.And) and isinstance(f.right, F.Not):
        return False
    elif "NOT" in str(f):
        return False
    elif isinstance(f, F.Or) or isinstance(f, F.And):
        components = f.get_vals()
        disjoint = True
        for c1 in components:
            for c2 in components:
                if c1 == c2:
                    continue
                if fpd_matrix[2, c1, c2] == False:
                    disjoint = False
                    break
        return disjoint
    else:
        raise ValueError(f"Unknown formula type {type(f)}")


def update_optimal_quantities(*, next_frontier, label_mapping, heuristic_info, max_size_mask, fpd_matrix):
    seg_quantities, neuron_quantities, concepts_quantities = heuristic_info
    neuron_unique_tuple, neuron_common_tuple, neuron_coverable_mask, neuron_coverable_tuple, neuron_sum_tuple  = neuron_quantities
    neuron_unique = neuron_unique_tuple[0]
    neuron_common = neuron_common_tuple[0]
    neuron_coverable = neuron_coverable_tuple[0]
    neuron_sum = neuron_sum_tuple[0]
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
    
    #start_time = time.time()
    
    for node in next_frontier:
        label = node[2]
        if label in label_mapping.keys():
            # Equivalence. We already have this label in the heuristic info
            continue
        # Assign first free available index to the node
        index_node = len(common_intersection)
        # if index_node in label_mapping.values():
        #     raise ValueError(f"Index {index_node} already exists in label mapping.")
        label_mapping[label] = index_node

        # Extract heuristic information for the labels included in the node
        left_label = label.left
        right_label = label.right
        if isinstance(left_label, F.Not):
            left_label = left_label.val
        if isinstance(right_label, F.Not):
            right_label = right_label.val

        # if left_label not in label_mapping or right_label not in label_mapping:
        #     raise ValueError(f"Labels {left_label} or {right_label} not found in label mapping.")

        index_left = label_mapping[left_label]
        index_right = label_mapping[right_label]
        left_quantities = (common_intersection[index_left], unique_intersection[index_left],
                           common_extras[index_left], unique_extras[index_left], uncovered[index_left])
        right_quantities = (common_intersection[index_right], unique_intersection[index_right],
                            common_extras[index_right], unique_extras[index_right], uncovered[index_right])
        
        #disjoint = probe_disjointment(label, fpd_matrix)
        disjoint = False

        # Update the heuristic information for the node
        node_common_intersection, node_unique_intersection, node_common_extras, node_unique_extras, node_uncovered = numpy_estimate_label_info(
            label, left_quantities, right_quantities, neuron_unique, neuron_common, neuron_coverable, neuron_sum, max_size_mask, disjoint
        )
        # Append the new quantities to the existing ones
        common_intersection.append(node_common_intersection)
        common_intersection_sum.append(node_common_intersection.sum())
        unique_intersection.append(node_unique_intersection)
        unique_intersection_sum.append(node_unique_intersection.sum())
        common_extras.append(node_common_extras)
        common_extras_sum.append(node_common_extras.sum())
        unique_extras.append(node_unique_extras)
        unique_extras_sum.append(node_unique_extras.sum())
        uncovered.append(node_uncovered)
        uncovered_sum.append(node_uncovered.sum())
    #node_time = time.time() - start_time
    #print(f"Time to estimate label {label}: {node_time:.4f} seconds")

    # Update the heuristic information tuple
    concepts_quantities = (
        (common_intersection, common_intersection_sum), (unique_intersection, unique_intersection_sum), (common_extras, common_extras_sum), (unique_extras, unique_extras_sum), (uncovered, uncovered_sum))
    heuristic_info = (seg_quantities, neuron_quantities, concepts_quantities)
    return heuristic_info


def get_esti_quantities(node, label_mapping, heuristic_info, max_size_mask, fpd_matrix):
    seg_quantities, neuron_quantities, concepts_quantities = heuristic_info
    neuron_unique_tuple, neuron_common_tuple, neuron_coverable_mask, neuron_coverable_tuple, neuron_sum_tuple  = neuron_quantities
    neuron_unique = neuron_unique_tuple[0]
    neuron_common = neuron_common_tuple[0]
    neuron_coverable = neuron_coverable_tuple[0]
    neuron_sum = neuron_sum_tuple[0]
    common_intersection_tuple, unique_intersection_tuple, common_extras_tuple, unique_extras_tuple, uncovered_tuple = concepts_quantities
    # neuron_unique = neuron_unique_tuple[0]
    # neuron_unique_sum = neuron_unique_tuple[1]
    # neuron_common = neuron_common_tuple[0]
    # neuron_common_sum = neuron_common_tuple[1]
    # neuron_coverable = neuron_coverable_tuple[0]
    # neuron_coverable_sum = neuron_coverable_tuple[1]
    # neuron_sum = neuron_sum_tuple[0]
    # neuron_sum_sum = neuron_sum_tuple[1]
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
    label = node[2]
    if label in label_mapping.keys():
        # Equivalence. We already have this label in the heuristic info
        index_node = label_mapping[label]
        # print(common_intersection_sum[index_node], unique_intersection_sum[index_node], common_extras_sum[index_node], unique_extras_sum[index_node], uncovered_sum[index_node])
        # print(common_intersection[index_node].sum(), unique_intersection[index_node].sum(), common_extras[index_node].sum(), unique_extras[index_node].sum(), uncovered[index_node].sum())
        #return common_intersection[index_node], unique_intersection[index_node], common_extras[index_node], unique_extras[index_node], uncovered[index_node]
        return (common_intersection[index_node], common_intersection_sum[index_node]), \
               (unique_intersection[index_node], unique_intersection_sum[index_node]), \
               (common_extras[index_node], common_extras_sum[index_node]), \
               (unique_extras[index_node], unique_extras_sum[index_node]), \
               (uncovered[index_node], uncovered_sum[index_node])
    # print("QUI NO")
    # exit()
    # Extract heuristic information for the labels included in the node
    left_label = label.left
    right_label = label.right
    if isinstance(left_label, F.Not):
        left_label = left_label.val
    if isinstance(right_label, F.Not):
        right_label = right_label.val

    # if left_label not in label_mapping or right_label not in label_mapping:
    #     raise ValueError(f"Labels {left_label} or {right_label} not found in label mapping.")
    if left_label in label_mapping:
        index_left = label_mapping[left_label]
        left_quantities = ((common_intersection[index_left], common_intersection_sum[index_left]), (unique_intersection[index_left], unique_intersection_sum[index_left]),
                    (common_extras[index_left], common_extras_sum[index_left]), (unique_extras[index_left], unique_extras_sum[index_left]), (uncovered[index_left], uncovered_sum[index_left]))
    else:
        left_quantities = get_esti_quantities(
            node=(None, None, left_label), label_mapping=label_mapping,
            heuristic_info=heuristic_info, max_size_mask=max_size_mask, fpd_matrix=fpd_matrix)
    if right_label in label_mapping:
        index_right = label_mapping[right_label]
        right_quantities = ((common_intersection[index_right], common_intersection_sum[index_right]), (unique_intersection[index_right], unique_intersection_sum[index_right]),
                            (common_extras[index_right], common_extras_sum[index_right]), (unique_extras[index_right], unique_extras_sum[index_right]), (uncovered[index_right], uncovered_sum[index_right]))
    else:
        right_quantities = get_esti_quantities(
            node=(None, None, right_label), label_mapping=label_mapping,
            heuristic_info=heuristic_info, max_size_mask=max_size_mask, fpd_matrix=fpd_matrix)
    
    #disjoint = probe_disjointment(label, fpd_matrix)
    disjoint = False

    node_common_intersection, node_unique_intersection, node_common_extras, node_unique_extras, node_uncovered = numpy_estimate_label_info(
        label, left_quantities, right_quantities, neuron_unique, neuron_common, neuron_coverable, neuron_sum, max_size_mask, disjoint
    )

    return node_common_intersection, node_unique_intersection, node_common_extras, node_unique_extras, node_uncovered

def update_quantities_by_ancestors(*, frontier, ancestors, label_mapping, heuristic_info, max_size_mask, fpd_matrix):
    #info, quantities = heuristic_info
    #common_intersection, unique_intersection, common_extras, unique_extras, uncovered, (neuron_unique, neuron_common, neuron_coverable), elements = quantities
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
    for node in frontier:
        label = node[2]
        if label in label_mapping:
            index_node = label_mapping[label]
        else:
            # Assign first free available index to the node
            index_node = len(common_intersection)
            if index_node in label_mapping.values():
                raise ValueError(f"Index {index_node} already exists in label mapping.")
            label_mapping[label] = index_node

        # Extract heuristic information for the labels included in the node
        left_label = label.left
        right_label = label.right
        if isinstance(left_label, F.Not):
            left_label = left_label.val
        if isinstance(right_label, F.Not):
            right_label = right_label.val

        if left_label not in label_mapping or right_label not in label_mapping:
            raise ValueError(f"Labels {left_label} or {right_label} not found in label mapping.")

        index_left = label_mapping[left_label]
        index_right = label_mapping[right_label]
        left_quantities = (common_intersection[index_left], unique_intersection[index_left],
                           common_extras[index_left], unique_extras[index_left], uncovered[index_left])
        right_quantities = (common_intersection[index_right], unique_intersection[index_right],
                            common_extras[index_right], unique_extras[index_right], uncovered[index_right])
        relationship = None

        # Update the heuristic information for the node
        node_common_intersection, node_unique_intersection, node_common_extras, node_unique_extras, node_uncovered = numpy_estimate_label_info(
            label, left_quantities, right_quantities, neuron_unique, neuron_common, neuron_coverable, neuron_sum, max_size_mask, relationship
        )
        # Append the new quantities to the existing ones
        common_intersection.append(node_common_intersection)
        common_intersection_sum.append(node_common_intersection.sum())
        unique_intersection.append(node_unique_intersection)
        unique_intersection_sum.append(node_unique_intersection.sum())
        common_extras.append(node_common_extras)
        common_extras_sum.append(node_common_extras.sum())
        unique_extras.append(node_unique_extras)
        unique_extras_sum.append(node_unique_extras.sum())
        uncovered.append(node_uncovered)
        uncovered_sum.append(node_uncovered.sum())

    # Update the heuristic information tuple
    concepts_quantities = (
        (common_intersection, common_intersection_sum), (unique_intersection, unique_intersection_sum), (common_extras, common_extras_sum), (unique_extras, unique_extras_sum), (uncovered, uncovered_sum))
    heuristic_info = (seg_quantities, neuron_quantities, concepts_quantities)
    #heuristic_info = (info, (common_intersection, unique_intersection, common_extras, unique_extras, uncovered, (neuron_unique, neuron_common, neuron_coverable), seg_quantities))
    return heuristic_info

def update_heuristic_info(*, heuristic, heuristic_info, next_frontier, label_mapping, max_size_mask, fpd_matrix):
    """
    Update the heuristic information based on the next frontier.

    Args:
        heuristic (str): The name of the heuristic.
        heuristic_info (tuple): The current heuristic information.
        next_frontier (list): The next frontier of formulas.
        label_mapping (dict): A mapping of labels to their indices.

    Returns:
        tuple: Updated heuristic information.
    """
    if heuristic == "optimal":
        return update_optimal_quantities(next_frontier=next_frontier, label_mapping=label_mapping,
                                         heuristic_info=heuristic_info, fpd_matrix=fpd_matrix, max_size_mask=max_size_mask)
    else:
        raise ValueError(f"Unknown heuristic: {heuristic}")

def get_heuristic_score(node, label_mapping, heuristic, heuristic_info, max_improvement, fpd_matrix, num_hits, max_size_mask, max_length):
    if heuristic == "optimal":
        return numpy_estimate_optimal_label_iou(node,label_mapping, heuristic_info, max_improvement, num_hits, max_size_mask, max_length, fpd_matrix)
    else:
        raise ValueError(f"Unknown heuristic: {heuristic}")