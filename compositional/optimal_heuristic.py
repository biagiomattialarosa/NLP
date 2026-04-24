
import time

from . import formula as F
from . import sum_heuristic, sample_heuristic
from .heuristic_utils import INDEX_OR, INDEX_AND, INDEX_NOT, INDEX_NODE_IOU_ESTI, INDEX_NODE_OPS


def is_in(op, list_of_lists):
    """
    Check if the operation is in any of the lists in list_of_lists.
    
    Args:
        op (str): The operation to check.
        list_of_lists (list of list): The lists to check against.
        
    Returns:
        bool: True if the operation is found in any of the lists, False otherwise.
    """
    return any(op in lst for lst in list_of_lists)


def are_disjoint(formula_a, formula_b, disjoint_info):
    if isinstance(formula_b, F.Not):
        return are_disjoint(formula_a, formula_b.val, disjoint_info)
    # TEMPORARY FOR DISTRIBUTIVE PROPERTY
    if not isinstance(formula_b, F.Leaf):
        return False
    concept_b = formula_b.val
    if isinstance(formula_a, F.Leaf):
        concept_a = formula_a.val
        if disjoint_info[concept_a, concept_b]:
            return True
        else:
            return False
    elif isinstance(formula_a, F.Or):
        left_disjoint = are_disjoint(formula_a.left, formula_b, disjoint_info)
        right_disjoint = are_disjoint(formula_a.right, formula_b, disjoint_info)
        return left_disjoint and right_disjoint
    elif isinstance(formula_a, F.And):
        if isinstance(formula_a.right, F.Not):
            # In this case, we check if the left part is disjoint with the right part
            left_disjoint = are_disjoint(formula_a.left, formula_b, disjoint_info)
            return left_disjoint
        else:
            # In this case, we check if the left or right part is disjoint. One is enough becase they limit each other
            left_disjoint = are_disjoint(formula_a.left, formula_b, disjoint_info)
            right_disjoint = are_disjoint(formula_a.right, formula_b, disjoint_info)
        return left_disjoint or right_disjoint
    else:
        raise ValueError(f"Unknown formula type: {type(formula_a)}")

def get_esti_quantities(heuristic, node, label_mapping, heuristic_info, max_size_mask, disjoint_info):
    seg_quantities, neuron_quantities, concepts_quantities = heuristic_info
    label = node[2]
    if len(label) == 1 or label in label_mapping:
        # Equivalence. We already have this label in the heuristic info
        index_node = label_mapping[label]
        return concepts_quantities[index_node]
    # Extract heuristic information for the labels included in the node
    if isinstance(label.left, F.Not):
        left_label_val = label.left.val
    else:
        left_label_val = label.left
    if isinstance(label.right, F.Not):
        right_label_val = label.right.val
    else:
        right_label_val = label.right
    left_quantities = get_esti_quantities(heuristic,
        node=(None, None, left_label_val, None), label_mapping=label_mapping,
        heuristic_info=heuristic_info, max_size_mask=max_size_mask, disjoint_info=disjoint_info)
    if left_quantities is None:
        return None
    right_quantities = get_esti_quantities(heuristic,
        node=(None, None, right_label_val, None), label_mapping=label_mapping,
        heuristic_info=heuristic_info, max_size_mask=max_size_mask, disjoint_info=disjoint_info)
    if right_quantities is None:
        return None
    disjoint = are_disjoint(formula_a=label.left, formula_b=label.right, disjoint_info=disjoint_info)
    if disjoint:
        node_quantities = heuristic.estimate_disjoint_label_info(
            label, left_quantities, right_quantities, neuron_quantities, max_size_mask, disjoint_info
        )
    else:
        node_quantities = heuristic.estimate_label_info(
            label, left_quantities, right_quantities, neuron_quantities, max_size_mask,
        )

    return node_quantities


# def update_optimal_label_iou(*, heuristic_name, node, label_mapping, heuristic_info, max_improvement, disjoint_info, num_hits, max_size_mask, max_length):
#     if heuristic_name == 'sum':
#         heuristic = sum_heuristic
#     elif heuristic_name == 'sample':
#         heuristic = sample_heuristic
#     elif heuristic_name == 'hybrid':
#         heuristic = hybrid_heuristic
#     else:
#         raise ValueError(f"Unknown heuristic name: {heuristic_name}")
#     seg_quantities, neuron_quantities, concepts_quantities = heuristic_info
#     label = node[2]
#     previous_paths_to_expand = node[3]
#     if previous_paths_to_expand is not None:
#         _, previous_or_paths, previous_and_paths, previous_and_not_paths = previous_paths_to_expand
#     else:
#         _, previous_or_paths, previous_and_paths, previous_and_not_paths = [], [], [], []
#     label_node = (None, None, label, None)
#     label_quantities = get_esti_quantities(heuristic,
#         label_node, label_mapping, heuristic_info, max_size_mask, disjoint_info=disjoint_info)

#     if label_quantities is None:
#         # Label discarded at the previous step
#         return  0.0, 0.0

#     or_condition = is_in('OR', previous_or_paths)
#     and_condition = is_in('AND', previous_and_paths)
#     and_not_condition = is_in('NOT', previous_and_not_paths)
#     and_and_not_condition =(is_in(['AND', 'NOT'], previous_and_paths) or is_in(['AND', 'NOT'], previous_and_not_paths))
#     and_or_condition = (is_in(['OR', 'AND'], previous_or_paths) or \
#         is_in(['OR', 'AND'], previous_and_paths))
#     or_not_condition =  (is_in(['OR', 'NOT'], previous_or_paths) or \
#         is_in(['OR', 'NOT'], previous_and_not_paths))
#     every_condition = (is_in(['OR', 'AND', 'NOT'], previous_or_paths) or \
#         is_in(['OR', 'AND', 'NOT'], previous_and_paths) or \
#         is_in(['OR', 'AND', 'NOT'], previous_and_not_paths))

#     # Individual estimation
#     ind_new_max, ind_new_min = heuristic.individual_estimation(label,
#         label_quantities,
#         neuron_quantities,
#         max_improvement=max_improvement, num_hits=num_hits, max_size_mask=max_size_mask, max_length=max_length
#     )

#     if or_condition:
#         or_new_max, or_new_min = heuristic.or_chain_estimation(label,
#             label_quantities,
#             neuron_quantities,
#             max_improvement=max_improvement, num_hits=num_hits, max_size_mask=max_size_mask, max_length=max_length
#         )
#     else:
#         or_new_max, or_new_min = 0, 0
#     if and_not_condition:
#         and_not_new_max, and_not_new_min = heuristic.and_not_chain_estimation(label,
#             label_quantities,
#             neuron_quantities,
#             max_improvement=max_improvement, num_hits=num_hits, max_size_mask=max_size_mask, max_length=max_length
#         )
#     else:
#         and_not_new_max, and_not_new_min = 0, 0
#     if and_condition:
#         # This covers both AND and AND-AND NOT chains
#         and_new_max, and_new_min = heuristic.and_chain_estimation(label,
#             label_quantities,
#             neuron_quantities,
#             max_improvement=max_improvement, num_hits=num_hits, max_size_mask=max_size_mask, max_length=max_length
#         )
#     else:
#         and_new_max, and_new_min = 0, 0
#     if and_and_not_condition:
#         and_andnot_new_max, and_andnot_new_min = heuristic.and_chain_estimation(label,
#             label_quantities,
#             neuron_quantities,
#             max_improvement=max_improvement, num_hits=num_hits, max_size_mask=max_size_mask, max_length=max_length
#         )
#     else:
#         and_andnot_new_max, and_andnot_new_min = 0, 0
#     if and_or_condition:
#         # This covers both AND-OR chaing and everything chain
#         comb_and_or_new_max, comb_and_or_new_min = heuristic.comb_and_or_chain_estimation(label,
#             label_quantities,
#             neuron_quantities,
#             max_improvement=max_improvement, num_hits=num_hits, max_size_mask=max_size_mask, max_length=max_length
#         )
#     else:
#         comb_and_or_new_max, comb_and_or_new_min = 0, 0
#     if or_not_condition:
#         or_not_new_max, or_not_new_min = heuristic.comb_or_andnot_chain_estimation(label,
#             label_quantities,
#             neuron_quantities,
#             max_improvement=max_improvement, num_hits=num_hits, max_size_mask=max_size_mask, max_length=max_length
#         )
#     else:
#         or_not_new_max, or_not_new_min = 0, 0
    
#     if every_condition:
#         # This covers both AND-OR chaing and everything chain
#         every_new_max, every_new_min = heuristic.comb_and_or_chain_estimation(label,
#             label_quantities,
#             neuron_quantities,
#             max_improvement=max_improvement, num_hits=num_hits, max_size_mask=max_size_mask, max_length=max_length
#         )
#     else:
#         every_new_max, every_new_min = 0, 0

#     new_max = max(ind_new_max, or_new_max, and_not_new_max, and_new_max, and_andnot_new_max, comb_and_or_new_max, or_not_new_max, every_new_max)
#     new_min = min(ind_new_min, or_new_min, and_not_new_min, and_new_min, and_andnot_new_min, comb_and_or_new_min, or_not_new_min, every_new_min)

#     return new_max, new_min



# def estimate_label_iou(heuristic_name, node, label_mapping, heuristic_info, max_improvement, *, num_hits, max_size_mask, max_length, disjoint_info, minimum_threshold=0):
#     """
#     Estimate the IoU of a label using the optimal heuristic.
    
#     Args:
#         label (F.Formula): The label to estimate the IoU for.
#         heuristic_info (tuple): The information to compute the heuristic.
        
#     Returns:
#         float: Estimated IoU of the label.
#     """
#     if heuristic_name == 'sum':
#         heuristic = sum_heuristic
#     elif heuristic_name == 'sample':
#         heuristic = sample_heuristic
#     elif heuristic_name == 'hybrid':
#         heuristic = hybrid_heuristic
#     else:
#         raise ValueError(f"Unknown heuristic name: {heuristic_name}")
#     seg_quantities, neuron_quantities, concepts_quantities = heuristic_info

#     label = node[2]
#     previous_paths_to_expand = node[3]
#     next_path = node[1]

#     label_node = (None, None, label, previous_paths_to_expand)
#     node_or_paths = [0, 'OR', label, [[], [], [], []]]
#     node_and_paths = [0, 'AND', label, [[], [], [], []]]
#     node_and_not_paths = [0, 'NOT', label, [[], [], [], []]]
#     node_individual_path = [0, 'INDIVIDUAL', label, [[], [], [], []]]
#     label_quantities = get_esti_quantities(heuristic,
#         label_node, label_mapping, heuristic_info, max_size_mask, disjoint_info=disjoint_info, 
#         )
#     if label_quantities is None:
#         # Label discarded at the previous step
#         return [node_individual_path, node_or_paths, node_and_paths, node_and_not_paths], 0.0

#     # Individual estimation
#     init_time = time.time()
#     max_individual_iou, min_individual_iou = heuristic.individual_estimation(label,
#         label_quantities,
#         neuron_quantities,
#         max_improvement, num_hits=num_hits, max_size_mask=max_size_mask, max_length=max_length, minimum_threshold=minimum_threshold
#     )


#     if len(label) == max_length or next_path == 'INDIVIDUAL':
#         # If the label is already at maximum length, we cannot estimate any chain
#         if max_individual_iou > 0.0:
#             # Update IoU of the individual path
#             node_individual_path[0] = max_individual_iou
#         return [node_individual_path, node_or_paths, node_and_paths, node_and_not_paths], min_individual_iou, 
#     if previous_paths_to_expand is not None:
#         _, previous_or_paths, previous_and_paths, previous_and_not_paths = previous_paths_to_expand
#     else:
#         _, previous_or_paths, previous_and_paths, previous_and_not_paths = [], [], [], []
#     # Conditions to steer the estimation of the chains
#     available_spots = max_length - len(label)
#     or_condition = available_spots > 0 and (previous_paths_to_expand is None or is_in('OR', previous_or_paths))
#     and_condition = available_spots > 0 and (previous_paths_to_expand is None or is_in('AND', previous_and_paths))
#     and_not_condition = available_spots > 0 and (previous_paths_to_expand is None or is_in('NOT', previous_and_not_paths))
#     and_and_not_condition = available_spots > 1 and (previous_paths_to_expand is None or (is_in(['AND', 'NOT'], previous_and_paths) or is_in(['AND', 'NOT'], previous_and_not_paths)))
#     and_or_condition = available_spots > 1 and (previous_paths_to_expand is None or (is_in(['OR', 'AND'], previous_or_paths) or \
#         is_in(['OR', 'AND'], previous_and_paths)))
#     or_not_condition = available_spots > 1 and (previous_paths_to_expand is None or (is_in(['OR', 'NOT'], previous_or_paths) or \
#         is_in(['OR', 'NOT'], previous_and_not_paths)))
#     every_condition = available_spots > 2 and (previous_paths_to_expand is None or (is_in(['OR', 'AND', 'NOT'], previous_or_paths) or \
#         is_in(['OR', 'AND', 'NOT'], previous_and_paths) or \
#         is_in(['OR', 'AND', 'NOT'], previous_and_not_paths)))
#     if or_condition:
#         init_time = time.time()
#         max_or_chain_iou, min_or_chain_iou = heuristic.or_chain_estimation(label,
#             label_quantities,
#             neuron_quantities,
#             max_improvement, num_hits=num_hits, max_size_mask=max_size_mask, max_length=max_length,
#             minimum_threshold=minimum_threshold
#         )
#     else:
#         max_or_chain_iou, min_or_chain_iou = 0.0, 0.0

#     # This covers AND chains
#     if and_condition:
#         init_time = time.time()
#         max_and_chain_iou, min_and_chain_iou = heuristic.and_chain_estimation(label,
#             label_quantities,
#             neuron_quantities,
#             max_improvement, num_hits=num_hits, max_size_mask=max_size_mask, max_length=max_length,
#             minimum_threshold=minimum_threshold
#         )
#     else:
#         max_and_chain_iou, min_and_chain_iou = 0.0, 0.0
#     if and_not_condition:
#         init_time = time.time()
#         max_and_not_chain_iou, min_and_not_chain_iou = heuristic.and_not_chain_estimation(label,
#             label_quantities,
#             neuron_quantities,
#             max_improvement, num_hits=num_hits, max_size_mask=max_size_mask, max_length=max_length,
#             minimum_threshold=minimum_threshold
#         )
#     else:
#         max_and_not_chain_iou, min_and_not_chain_iou = 0.0, 0.0


#     if and_and_not_condition:
#         max_and_and_not_chain_iou = max_and_chain_iou
#         min_and_and_not_chain_iou = min_and_chain_iou
#     else:
#         max_and_and_not_chain_iou, min_and_and_not_chain_iou = 0.0, 0.0

#     if and_or_condition:
#         # This covers both AND-OR chaing and everything chain
#         init_time = time.time()
#         max_comb_and_or_chain_iou, min_comb_and_or_chain_iou = heuristic.comb_and_or_chain_estimation(label,
#             label_quantities,
#             neuron_quantities,
#             max_improvement, num_hits=num_hits, max_size_mask=max_size_mask, max_length=max_length,
#             minimum_threshold=minimum_threshold
#         )
#     else:
#         max_comb_and_or_chain_iou, min_comb_and_or_chain_iou = 0.0, 0.0

#     if or_not_condition:
#         init_time = time.time()
#         max_comb_or_andnot_chain_iou, min_comb_or_andnot_chain_iou = heuristic.comb_or_andnot_chain_estimation(label,
#             label_quantities,
#             neuron_quantities,
#             max_improvement, num_hits=num_hits, max_size_mask=max_size_mask, max_length=max_length,
#             minimum_threshold=minimum_threshold
#         )
#     else:
#         max_comb_or_andnot_chain_iou, min_comb_or_andnot_chain_iou = 0.0, 0.0


    
#     # Greater chains
#     if every_condition:
#         max_every_chain_iou = max_comb_and_or_chain_iou
#         min_every_chain_iou = min_comb_and_or_chain_iou
#     else:
#         max_every_chain_iou, min_every_chain_iou = 0.0, 0.0
#     # Max internal min
#     max_minimum = max(
#         min_individual_iou, min_or_chain_iou, min_and_chain_iou, min_and_not_chain_iou,
#         min_comb_and_or_chain_iou, min_comb_or_andnot_chain_iou
#     )

#     minimum_threshold = max(minimum_threshold, max_minimum)


#     max_results = []
#     for ops, max_iou, min_iou in zip([[], ['OR'], ['AND'], ['AND', 'NOT'], ['NOT'], ['AND','OR'],['AND', 'OR', 'NOT'], ['OR', 'NOT']],
#                     [max_individual_iou, max_or_chain_iou, max_and_chain_iou, max_and_and_not_chain_iou, max_and_not_chain_iou, max_comb_and_or_chain_iou, max_every_chain_iou, max_comb_or_andnot_chain_iou],
#                     [min_individual_iou, min_or_chain_iou, min_and_chain_iou, min_and_and_not_chain_iou, min_and_not_chain_iou, min_comb_and_or_chain_iou, min_every_chain_iou, min_comb_or_andnot_chain_iou]):
#         if max_iou > 0 and max_iou >= minimum_threshold:
#             if len(ops) == 0:
#                 node_individual_path[INDEX_NODE_IOU_ESTI] = max_iou
#             elif len(ops) == 1:
#                 op = ops[0]
#                 if op == 'OR':
#                     if node_or_paths[INDEX_NODE_IOU_ESTI] < max_iou:
#                         node_or_paths[INDEX_NODE_IOU_ESTI] = max_iou
#                     node_or_paths[INDEX_NODE_OPS][INDEX_OR].append(ops)
#                 elif op == 'AND':
#                     if node_and_paths[INDEX_NODE_IOU_ESTI] < max_iou:
#                         node_and_paths[INDEX_NODE_IOU_ESTI] = max_iou
#                     node_and_paths[INDEX_NODE_OPS][INDEX_AND].append(ops)
#                 elif op == 'NOT':
#                     if node_and_not_paths[INDEX_NODE_IOU_ESTI] < max_iou:
#                         node_and_not_paths[INDEX_NODE_IOU_ESTI] = max_iou
#                     node_and_not_paths[INDEX_NODE_OPS][INDEX_NOT].append(ops)
#                 elif op == 'INDIVIDUAL':
#                     node_individual_path[INDEX_NODE_IOU_ESTI] = max_iou
#             elif len(ops) == 2:
#                 if 'OR' in ops and 'AND' in ops:
#                     if node_or_paths[INDEX_NODE_IOU_ESTI] < max_iou:
#                         node_or_paths[INDEX_NODE_IOU_ESTI] = max_iou
#                     if node_and_paths[INDEX_NODE_IOU_ESTI] < max_iou:
#                         node_and_paths[INDEX_NODE_IOU_ESTI] = max_iou
#                     node_or_paths[INDEX_NODE_OPS][INDEX_OR].append(ops)
#                     node_or_paths[INDEX_NODE_OPS][INDEX_AND].append(ops)
#                     node_and_paths[INDEX_NODE_OPS][INDEX_AND].append(ops)
#                     node_and_paths[INDEX_NODE_OPS][INDEX_OR].append(ops)
#                 elif 'OR' in ops and 'NOT' in ops:
#                     if node_or_paths[INDEX_NODE_IOU_ESTI] < max_iou:
#                         node_or_paths[INDEX_NODE_IOU_ESTI] = max_iou
#                     if node_and_not_paths[INDEX_NODE_IOU_ESTI] < max_iou:
#                         node_and_not_paths[INDEX_NODE_IOU_ESTI] = max_iou
#                     node_or_paths[INDEX_NODE_OPS][INDEX_OR].append(ops)
#                     node_or_paths[INDEX_NODE_OPS][INDEX_NOT].append(ops)
#                     node_and_not_paths[INDEX_NODE_OPS][INDEX_NOT].append(ops)
#                     node_and_not_paths[INDEX_NODE_OPS][INDEX_OR].append(ops)
#                 elif 'AND' in ops and 'NOT' in ops:
#                     if node_and_paths[INDEX_NODE_IOU_ESTI] < max_iou:
#                         node_and_paths[INDEX_NODE_IOU_ESTI] = max_iou
#                     if node_and_not_paths[INDEX_NODE_IOU_ESTI] < max_iou:
#                         node_and_not_paths[INDEX_NODE_IOU_ESTI] = max_iou
#                     node_and_paths[INDEX_NODE_OPS][INDEX_AND].append(ops)
#                     node_and_paths[INDEX_NODE_OPS][INDEX_NOT].append(ops)
#                     node_and_not_paths[INDEX_NODE_OPS][INDEX_NOT].append(ops)
#                     node_and_not_paths[INDEX_NODE_OPS][INDEX_AND].append(ops)
#             elif len(ops) == 3:
#                 if 'OR' in ops and 'AND' in ops and 'NOT' in ops:
#                     if node_or_paths[INDEX_NODE_IOU_ESTI] < max_iou:
#                         node_or_paths[INDEX_NODE_IOU_ESTI] = max_iou
#                     if node_and_paths[INDEX_NODE_IOU_ESTI] < max_iou:
#                         node_and_paths[INDEX_NODE_IOU_ESTI] = max_iou
#                     if node_and_not_paths[INDEX_NODE_IOU_ESTI] < max_iou:
#                         node_and_not_paths[INDEX_NODE_IOU_ESTI] = max_iou
#                     node_or_paths[INDEX_NODE_OPS][INDEX_OR].append(ops)
#                     node_or_paths[INDEX_NODE_OPS][INDEX_AND].append(ops)
#                     node_or_paths[INDEX_NODE_OPS][INDEX_NOT].append(ops)
#                     node_and_paths[INDEX_NODE_OPS][INDEX_AND].append(ops)
#                     node_and_paths[INDEX_NODE_OPS][INDEX_OR].append(ops)
#                     node_and_paths[INDEX_NODE_OPS][INDEX_NOT].append(ops)
#                     node_and_not_paths[INDEX_NODE_OPS][INDEX_NOT].append(ops)
#                     node_and_not_paths[INDEX_NODE_OPS][INDEX_AND].append(ops)
#                     node_and_not_paths[INDEX_NODE_OPS][INDEX_OR].append(ops)
#             else:
#                 raise ValueError(f"Unexpected number of operations: {len(ops)}")
#     max_results = [node_individual_path, node_or_paths, node_and_paths, node_and_not_paths]
#     return max_results, max_minimum


def estimate_iou_from_quantities(quantities, minimum_threshold=0.0):
    (max_intersection, min_intersection), (max_union, min_union) = quantities    
    max_iou = max_intersection / min_union if min_union > 0 else 0.0
    min_iou = min_intersection / max_union if max_union > 0 else 0.0
    if max_iou < minimum_threshold:
        return 0.0, 0.0
    return max_iou, min_iou

def get_combo_quantities(quantities_a, quantities_b):
    (max_intersection_a, min_intersection_a), (max_union_a, min_union_a) = quantities_a
    (max_intersection_b, min_intersection_b), (max_union_b, min_union_b) = quantities_b
    max_intersection = max(max_intersection_a, max_intersection_b)
    min_intersection = min(min_intersection_a, min_intersection_b)
    max_union = max(max_union_a, max_union_b)
    min_union = min(min_union_a, min_union_b)
    return (max_intersection, min_intersection), (max_union, min_union)


def update_optimal_label_iou(*, heuristic_name, node, label_mapping, heuristic_info, max_improvement, disjoint_info, num_hits, max_size_mask, max_length):
    if heuristic_name == 'sum':
        heuristic = sum_heuristic
    elif heuristic_name == 'sample':
        heuristic = sample_heuristic
    else:
        raise ValueError(f"Unknown heuristic name: {heuristic_name}")
    seg_quantities, neuron_quantities, concepts_quantities = heuristic_info
    label = node[2]
    previous_paths_to_expand = node[3]
    if previous_paths_to_expand is not None:
        _, previous_or_paths, previous_and_paths, previous_and_not_paths = previous_paths_to_expand
    else:
        _, previous_or_paths, previous_and_paths, previous_and_not_paths = [], [], [], []
    label_node = (None, None, label, None)
    label_quantities = get_esti_quantities(heuristic,
        label_node, label_mapping, heuristic_info, max_size_mask, disjoint_info=disjoint_info)

    if label_quantities is None:
        # Label discarded at the previous step
        return 0.0, 0.0

    or_condition = is_in('OR', previous_or_paths)
    and_condition = is_in('AND', previous_and_paths)
    and_not_condition = is_in('NOT', previous_and_not_paths)
    and_and_not_condition =(is_in(['AND', 'NOT'], previous_and_paths) or is_in(['AND', 'NOT'], previous_and_not_paths))
    and_or_condition = (is_in(['OR', 'AND'], previous_or_paths) or \
        is_in(['OR', 'AND'], previous_and_paths))
    or_not_condition =  (is_in(['OR', 'NOT'], previous_or_paths) or \
        is_in(['OR', 'NOT'], previous_and_not_paths))
    every_condition = (is_in(['OR', 'AND', 'NOT'], previous_or_paths) or \
        is_in(['OR', 'AND', 'NOT'], previous_and_paths) or \
        is_in(['OR', 'AND', 'NOT'], previous_and_not_paths))

    # Individual estimation
    ind_new_max, ind_new_min = heuristic.individual_estimation(label,
        label_quantities,
        neuron_quantities,
        max_improvement=max_improvement, num_hits=num_hits, max_size_mask=max_size_mask, max_length=max_length
    )

    if or_condition:
        or_quantities = heuristic.or_chain_estimation(label,
            label_quantities,
            neuron_quantities,
            max_improvement=max_improvement, num_hits=num_hits, max_size_mask=max_size_mask, max_length=max_length
        )
        or_new_max, or_new_min = estimate_iou_from_quantities(or_quantities)
    else:
        or_new_max, or_new_min = 0, 0
    if and_not_condition:
        and_not_quantities = heuristic.and_not_chain_estimation(label,
            label_quantities,
            neuron_quantities,
            max_improvement=max_improvement, num_hits=num_hits, max_size_mask=max_size_mask, max_length=max_length
        )
        and_not_new_max, and_not_new_min = estimate_iou_from_quantities(and_not_quantities)
    else:
        and_not_new_max, and_not_new_min = 0, 0
    if and_condition:
        # This covers both AND and AND-AND NOT chains
        and_quantities = heuristic.and_chain_estimation(label,
            label_quantities,
            neuron_quantities,
            max_improvement=max_improvement, num_hits=num_hits, max_size_mask=max_size_mask, max_length=max_length
        )
        and_new_max, and_new_min = estimate_iou_from_quantities(and_quantities)
    else:
        and_new_max, and_new_min = 0, 0
    if and_and_not_condition:
        and_andnot_quantities = heuristic.and_chain_estimation(label,
            label_quantities,
            neuron_quantities,
            max_improvement=max_improvement, num_hits=num_hits, max_size_mask=max_size_mask, max_length=max_length
        )
        and_andnot_new_max, and_andnot_new_min = estimate_iou_from_quantities(and_andnot_quantities)
    else:
        and_andnot_new_max, and_andnot_new_min = 0, 0
    if and_or_condition:
        # This covers both AND-OR chaing and everything chain
        and_or_quantities = get_combo_quantities(or_quantities, and_quantities)
        comb_and_or_new_max, comb_and_or_new_min = estimate_iou_from_quantities(and_or_quantities)
    else:
        comb_and_or_new_max, comb_and_or_new_min = 0, 0
    if or_not_condition:
        or_not_quantities = get_combo_quantities(or_quantities, and_not_quantities)
        or_not_new_max, or_not_new_min = estimate_iou_from_quantities(or_not_quantities)
    else:
        or_not_new_max, or_not_new_min = 0, 0
    
    if every_condition:
        # This covers both AND-OR chaing and everything chain
        every_quantities = get_combo_quantities(and_or_quantities, and_not_quantities)
        every_new_max, every_new_min = estimate_iou_from_quantities(every_quantities)
    else:
        every_new_max, every_new_min = 0, 0

    new_max = max(ind_new_max, or_new_max, and_not_new_max, and_new_max, and_andnot_new_max, comb_and_or_new_max, or_not_new_max, every_new_max)
    new_min = min(ind_new_min, or_new_min, and_not_new_min, and_new_min, and_andnot_new_min, comb_and_or_new_min, or_not_new_min, every_new_min)

    return new_max, new_min



def estimate_label_iou(heuristic_name, node, label_mapping, heuristic_info, max_improvement, *, num_hits, max_size_mask, max_length, disjoint_info, minimum_threshold=0):
    """
    Estimate the IoU of a label using the optimal heuristic.
    
    Args:
        label (F.Formula): The label to estimate the IoU for.
        heuristic_info (tuple): The information to compute the heuristic.
        
    Returns:
        float: Estimated IoU of the label.
    """
    if heuristic_name == 'sum':
        heuristic = sum_heuristic
    elif heuristic_name == 'sample':
        heuristic = sample_heuristic
    else:
        raise ValueError(f"Unknown heuristic name: {heuristic_name}")
    seg_quantities, neuron_quantities, concepts_quantities = heuristic_info

    label = node[2]
    previous_paths_to_expand = node[3]
    next_path = node[1]

    label_node = (None, None, label, previous_paths_to_expand)
    node_or_paths = [0, 'OR', label, [[], [], [], []]]
    node_and_paths = [0, 'AND', label, [[], [], [], []]]
    node_and_not_paths = [0, 'NOT', label, [[], [], [], []]]
    node_individual_path = [0, 'INDIVIDUAL', label, [[], [], [], []]]
    label_quantities = get_esti_quantities(heuristic,
        label_node, label_mapping, heuristic_info, max_size_mask, disjoint_info=disjoint_info, 
        )
    if label_quantities is None:
        # Label discarded at the previous step
        return [node_individual_path, node_or_paths, node_and_paths, node_and_not_paths], 0.0

    # Individual estimation
    max_individual_iou, min_individual_iou = heuristic.individual_estimation(label,
        label_quantities,
        neuron_quantities,
        max_improvement, num_hits=num_hits, max_size_mask=max_size_mask, max_length=max_length, minimum_threshold=minimum_threshold
    )


    if len(label) == max_length or next_path == 'INDIVIDUAL':
        # If the label is already at maximum length, we cannot estimate any chain
        if max_individual_iou > 0.0:
            # Update IoU of the individual path
            node_individual_path[0] = max_individual_iou
        return [node_individual_path, node_or_paths, node_and_paths, node_and_not_paths], min_individual_iou, 
    if previous_paths_to_expand is not None:
        _, previous_or_paths, previous_and_paths, previous_and_not_paths = previous_paths_to_expand
    else:
        _, previous_or_paths, previous_and_paths, previous_and_not_paths = [], [], [], []
    # Conditions to steer the estimation of the chains
    available_spots = max_length - len(label)
    or_condition = available_spots > 0 and (previous_paths_to_expand is None or is_in('OR', previous_or_paths))
    and_condition = available_spots > 0 and (previous_paths_to_expand is None or is_in('AND', previous_and_paths))
    and_not_condition = available_spots > 0 and (previous_paths_to_expand is None or is_in('NOT', previous_and_not_paths))
    and_and_not_condition = available_spots > 1 and (previous_paths_to_expand is None or (is_in(['AND', 'NOT'], previous_and_paths) or is_in(['AND', 'NOT'], previous_and_not_paths)))
    and_or_condition = available_spots > 1 and (previous_paths_to_expand is None or (is_in(['OR', 'AND'], previous_or_paths) or \
        is_in(['OR', 'AND'], previous_and_paths)))
    or_not_condition = available_spots > 1 and (previous_paths_to_expand is None or (is_in(['OR', 'NOT'], previous_or_paths) or \
        is_in(['OR', 'NOT'], previous_and_not_paths)))
    every_condition = available_spots > 2 and (previous_paths_to_expand is None or (is_in(['OR', 'AND', 'NOT'], previous_or_paths) or \
        is_in(['OR', 'AND', 'NOT'], previous_and_paths) or \
        is_in(['OR', 'AND', 'NOT'], previous_and_not_paths)))
    if or_condition:
        or_quantities = heuristic.or_chain_estimation(label,
            label_quantities,
            neuron_quantities,
            max_improvement, num_hits=num_hits, max_size_mask=max_size_mask, max_length=max_length,
            minimum_threshold=minimum_threshold
        )
        max_or_chain_iou, min_or_chain_iou = estimate_iou_from_quantities(or_quantities, minimum_threshold)
    else:
        max_or_chain_iou, min_or_chain_iou = 0.0, 0.0

    # This covers AND chains
    if and_condition:
        and_quantities = heuristic.and_chain_estimation(label,
            label_quantities,
            neuron_quantities,
            max_improvement, num_hits=num_hits, max_size_mask=max_size_mask, max_length=max_length,
            minimum_threshold=minimum_threshold
        )
        max_and_chain_iou, min_and_chain_iou = estimate_iou_from_quantities(and_quantities, minimum_threshold)
    else:
        max_and_chain_iou, min_and_chain_iou = 0.0, 0.0
    if and_not_condition:
        and_not_quantities = heuristic.and_not_chain_estimation(label,
            label_quantities,
            neuron_quantities,
            max_improvement, num_hits=num_hits, max_size_mask=max_size_mask, max_length=max_length,
            minimum_threshold=minimum_threshold
        )
        max_and_not_chain_iou, min_and_not_chain_iou = estimate_iou_from_quantities(and_not_quantities, minimum_threshold)
    else:
        max_and_not_chain_iou, min_and_not_chain_iou = 0.0, 0.0


    if and_and_not_condition:
        max_and_and_not_chain_iou = max_and_chain_iou
        min_and_and_not_chain_iou = min_and_chain_iou
    else:
        max_and_and_not_chain_iou, min_and_and_not_chain_iou = 0.0, 0.0

    if and_or_condition:
        # This covers both AND-OR chaing and everything chain
        and_or_quantities = get_combo_quantities(or_quantities, and_quantities)
        max_comb_and_or_chain_iou, min_comb_and_or_chain_iou = estimate_iou_from_quantities(and_or_quantities, minimum_threshold)
    else:
        max_comb_and_or_chain_iou, min_comb_and_or_chain_iou = 0.0, 0.0

    if or_not_condition:
        or_not_quantities = get_combo_quantities(or_quantities, and_not_quantities)
        max_comb_or_andnot_chain_iou, min_comb_or_andnot_chain_iou = estimate_iou_from_quantities(or_not_quantities, minimum_threshold)
    else:
        max_comb_or_andnot_chain_iou, min_comb_or_andnot_chain_iou = 0.0, 0.0


    
    # Greater chains
    if every_condition:
        max_every_chain_iou = max_comb_and_or_chain_iou
        min_every_chain_iou = min_comb_and_or_chain_iou
    else:
        max_every_chain_iou, min_every_chain_iou = 0.0, 0.0
    # Max internal min
    max_minimum = max(
        min_individual_iou, min_or_chain_iou, min_and_chain_iou, min_and_not_chain_iou,
        min_comb_and_or_chain_iou, min_comb_or_andnot_chain_iou
    )

    minimum_threshold = max(minimum_threshold, max_minimum)


    max_results = []
    for ops, max_iou, min_iou in zip([[], ['OR'], ['AND'], ['AND', 'NOT'], ['NOT'], ['AND','OR'],['AND', 'OR', 'NOT'], ['OR', 'NOT']],
                    [max_individual_iou, max_or_chain_iou, max_and_chain_iou, max_and_and_not_chain_iou, max_and_not_chain_iou, max_comb_and_or_chain_iou, max_every_chain_iou, max_comb_or_andnot_chain_iou],
                    [min_individual_iou, min_or_chain_iou, min_and_chain_iou, min_and_and_not_chain_iou, min_and_not_chain_iou, min_comb_and_or_chain_iou, min_every_chain_iou, min_comb_or_andnot_chain_iou]):
        if max_iou > 0 and max_iou >= minimum_threshold:
            if len(ops) == 0:
                node_individual_path[INDEX_NODE_IOU_ESTI] = max_iou
            elif len(ops) == 1:
                op = ops[0]
                if op == 'OR':
                    if node_or_paths[INDEX_NODE_IOU_ESTI] < max_iou:
                        node_or_paths[INDEX_NODE_IOU_ESTI] = max_iou
                    node_or_paths[INDEX_NODE_OPS][INDEX_OR].append(ops)
                elif op == 'AND':
                    if node_and_paths[INDEX_NODE_IOU_ESTI] < max_iou:
                        node_and_paths[INDEX_NODE_IOU_ESTI] = max_iou
                    node_and_paths[INDEX_NODE_OPS][INDEX_AND].append(ops)
                elif op == 'NOT':
                    if node_and_not_paths[INDEX_NODE_IOU_ESTI] < max_iou:
                        node_and_not_paths[INDEX_NODE_IOU_ESTI] = max_iou
                    node_and_not_paths[INDEX_NODE_OPS][INDEX_NOT].append(ops)
                elif op == 'INDIVIDUAL':
                    node_individual_path[INDEX_NODE_IOU_ESTI] = max_iou
            elif len(ops) == 2:
                if 'OR' in ops and 'AND' in ops:
                    if node_or_paths[INDEX_NODE_IOU_ESTI] < max_iou:
                        node_or_paths[INDEX_NODE_IOU_ESTI] = max_iou
                    if node_and_paths[INDEX_NODE_IOU_ESTI] < max_iou:
                        node_and_paths[INDEX_NODE_IOU_ESTI] = max_iou
                    node_or_paths[INDEX_NODE_OPS][INDEX_OR].append(ops)
                    node_or_paths[INDEX_NODE_OPS][INDEX_AND].append(ops)
                    node_and_paths[INDEX_NODE_OPS][INDEX_AND].append(ops)
                    node_and_paths[INDEX_NODE_OPS][INDEX_OR].append(ops)
                elif 'OR' in ops and 'NOT' in ops:
                    if node_or_paths[INDEX_NODE_IOU_ESTI] < max_iou:
                        node_or_paths[INDEX_NODE_IOU_ESTI] = max_iou
                    if node_and_not_paths[INDEX_NODE_IOU_ESTI] < max_iou:
                        node_and_not_paths[INDEX_NODE_IOU_ESTI] = max_iou
                    node_or_paths[INDEX_NODE_OPS][INDEX_OR].append(ops)
                    node_or_paths[INDEX_NODE_OPS][INDEX_NOT].append(ops)
                    node_and_not_paths[INDEX_NODE_OPS][INDEX_NOT].append(ops)
                    node_and_not_paths[INDEX_NODE_OPS][INDEX_OR].append(ops)
                elif 'AND' in ops and 'NOT' in ops:
                    if node_and_paths[INDEX_NODE_IOU_ESTI] < max_iou:
                        node_and_paths[INDEX_NODE_IOU_ESTI] = max_iou
                    if node_and_not_paths[INDEX_NODE_IOU_ESTI] < max_iou:
                        node_and_not_paths[INDEX_NODE_IOU_ESTI] = max_iou
                    node_and_paths[INDEX_NODE_OPS][INDEX_AND].append(ops)
                    node_and_paths[INDEX_NODE_OPS][INDEX_NOT].append(ops)
                    node_and_not_paths[INDEX_NODE_OPS][INDEX_NOT].append(ops)
                    node_and_not_paths[INDEX_NODE_OPS][INDEX_AND].append(ops)
            elif len(ops) == 3:
                if 'OR' in ops and 'AND' in ops and 'NOT' in ops:
                    if node_or_paths[INDEX_NODE_IOU_ESTI] < max_iou:
                        node_or_paths[INDEX_NODE_IOU_ESTI] = max_iou
                    if node_and_paths[INDEX_NODE_IOU_ESTI] < max_iou:
                        node_and_paths[INDEX_NODE_IOU_ESTI] = max_iou
                    if node_and_not_paths[INDEX_NODE_IOU_ESTI] < max_iou:
                        node_and_not_paths[INDEX_NODE_IOU_ESTI] = max_iou
                    node_or_paths[INDEX_NODE_OPS][INDEX_OR].append(ops)
                    node_or_paths[INDEX_NODE_OPS][INDEX_AND].append(ops)
                    node_or_paths[INDEX_NODE_OPS][INDEX_NOT].append(ops)
                    node_and_paths[INDEX_NODE_OPS][INDEX_AND].append(ops)
                    node_and_paths[INDEX_NODE_OPS][INDEX_OR].append(ops)
                    node_and_paths[INDEX_NODE_OPS][INDEX_NOT].append(ops)
                    node_and_not_paths[INDEX_NODE_OPS][INDEX_NOT].append(ops)
                    node_and_not_paths[INDEX_NODE_OPS][INDEX_AND].append(ops)
                    node_and_not_paths[INDEX_NODE_OPS][INDEX_OR].append(ops)
            else:
                raise ValueError(f"Unexpected number of operations: {len(ops)}")
    max_results = [node_individual_path, node_or_paths, node_and_paths, node_and_not_paths]
    return max_results, max_minimum
