from collections import Counter
import heapq
import queue as Q

import torch

from . import formula as F
from compositional import mask_utils, metrics

def get_or_compounds(formula):
    if isinstance(formula, F.Or):
        return get_or_compounds(formula.left) + get_or_compounds(formula.right)
    else:
        return [formula]
    
def get_and_compounds(formula):
    if isinstance(formula, F.And) and not isinstance(formula.right, F.Not):
        return get_and_compounds(formula.left) + get_and_compounds(formula.right)
    else:
        return [formula]
    
def fail_check_functional_equivalence(mask_formula1, mask_formula2):
    if metrics.iou(mask_formula1, mask_formula2) == 1.0:
        return True
    return False

def fail_check_beam_functional_equivalence(beam_masks, mask_formula):
    for label, mask in beam_masks.items():
        if fail_check_functional_equivalence(mask, mask_formula):
            return True, label
    return False, None

def extract_functional_equivalents(mask_formula, iou_formula, info_formulas):
    equivalent = []
    for label, (mask, iou) in info_formulas.items():
        if iou == iou_formula:
            # The only case where there could be logical equivalence
            if fail_check_functional_equivalence(mask, mask_formula):
                equivalent.append(label)
    return equivalent

def get_intersection(left_formula, right_formula):
    left_vals = left_formula.get_leaves()
    right_vals = right_formula.get_leaves()
    intersection = set(left_vals) & set(right_vals)
    return intersection

def share_same_concepts(left_formula, right_formula):
    left_vals = left_formula.get_leaves()
    right_vals = right_formula.get_leaves()
    intersection = set(left_vals) & set(right_vals)
    union = set(left_vals) | set(right_vals)
    return len(intersection) > 0 and len(union) > 0 and len(intersection) == len(union)

def combine_disjoint_formulas(left_formula, right_formula):
    if get_intersection(left_formula, right_formula):
        # case where there is an intersection, we cannot combine the formulas
        return []
    spanned_formulas = [F.And(left_formula, right_formula), F.Or(left_formula, right_formula)]
    return spanned_formulas

def apply_distributivity(left_formula, right_formula, op):
    intersection = get_intersection(left_formula, right_formula)
    if len(intersection) == 0:
        # No intersection. This case is managed by combine disjoint
        return None
    # Chain using the same common operator
    opposite_op = F.Or if op == F.And else F.And
    leaves = left_formula.get_leaves() + right_formula.get_leaves()
    leaves = sorted(set(leaves)) # Sort the different terms to ensure NOT are at the end of the list
    common_terms = [term for term in leaves if term in intersection]
    distributed_formula_left = common_terms[0]
    for term in common_terms[1:]:
        distributed_formula_left = op(distributed_formula_left, term)
    different_terms = [term for term in leaves if term not in intersection]
    different_terms = sorted(set(different_terms)) # Sort the different terms to ensure NOT are at the end of the list
    distributed_formula_right = different_terms[0]
    if isinstance(distributed_formula_right, F.Not):
        return None # We cannot distribute if we have only NOT terms due to the design of the heuristic that assumes each left side is a positive concept
    for term in different_terms[1:]:
        distributed_formula_right = opposite_op(distributed_formula_right, term)
    distributed_formula = op(distributed_formula_left, distributed_formula_right)
    return distributed_formula

def is_full_not(formula):
    formula_ops = formula.get_ops()
    if "OR" in formula_ops:
        # We cannot combine if we have OR, since we cannot distribute NOT over OR
        return False
    counter_and = formula_ops.count('AND')
    counter_not = formula_ops.count('NOT')
    if "AND" in formula_ops and "NOT" in formula_ops:
        # We should check if it is a full not operator
        if counter_and == counter_not:
            # We have a full not operator, we can combine
            return True
    return False

def combine_same_exclusive_op(left_formula, right_formula):
    left_op_list = list(set(left_formula.get_ops()))
    right_op_list = list(set(right_formula.get_ops()))
    if len(left_op_list) != len(right_op_list):
        # Different number of operators or not exactly one operator, cannot combine
        return []
    left_is_full_not = is_full_not(left_formula)
    right_is_full_not = is_full_not(right_formula)
    if not (left_is_full_not and right_is_full_not) and len(left_op_list) != 1:
        # Different number of operators or not exactly one operator, cannot combine
        return []
    left_op = left_op_list[0]
    right_op = right_op_list[0]
    if left_op != right_op:
        # Different operators, cannot combine
        return []
    if left_op == 'NOT':
        # NOT operator, cannot combine
        return []
    
    # We can combine in all the other cases
    if left_op == 'AND':
        op = F.And
    elif left_op == 'OR':
        op = F.Or
    
    left_leaves = left_formula.get_leaves()
    right_leaves = right_formula.get_leaves()
    intersection = set(left_leaves) & set(right_leaves)
    if len(intersection) == 0:
        # No intersection. This case is managed by combine disjoint
        return []
    
    # Chain using the same common operator
    leaves = left_leaves + right_leaves
    vals = sorted(set(leaves)) # Sort the different terms to ensure NOT are at the end of the list
    combined_formula = vals[0] # Note that we have at least one leaf since the leaf nodes would be discarded by the first condition in this function
    for term in vals[1:]:
        combined_formula = op(combined_formula, term)

    # Chain using the opposite operator. Distributivity. Can be applied only if we don't have NOT
    if "NOT" not in left_op_list and "NOT" not in right_op_list:
        distributed_formula = apply_distributivity(left_formula, right_formula, op)
    else:
        distributed_formula = None

    # assert not isinstance(distributed_formula, F.Not) # We should not have NOT at this stage, but we can have it in the different terms that are distributed
    if distributed_formula is None:
        return [combined_formula]
    else:
        return [combined_formula, distributed_formula]

def combine_different_op(left_formula, right_formula):
    left_op_list = list(set(left_formula.get_ops()))
    right_op_list = list(set(right_formula.get_ops()))
    if len(left_op_list) != 1 or len(right_op_list) != 1:
        # Not exactly one operator, cannot combine
        return []
    left_op = left_op_list[0]
    right_op = right_op_list[0]
    if left_op == right_op:
        # Same operators, cannot combine with this function
        return []
    if 'NOT' in (left_op, right_op):
        # NOT operator, cannot combine
        return []
    
    left_leaves = left_formula.get_leaves()
    right_leaves = right_formula.get_leaves()
    common_terms = set(left_leaves) & set(right_leaves)
    if len(common_terms) != 1:
        # No intersection. This case is managed by combine disjoint
        # Multiple intersection: the formulas are in contradiction, we cannot combine
        return []
    
    # Here, given (a OR b) and (a AND c) I would like to obtain ((a AND c) OR b) by replacing a with its specialization in the first formula
    or_leaves = left_leaves if left_op == 'OR' else right_leaves
    combined_formula = left_formula if left_op == 'AND' else right_formula # The end formula will be always full preserved
    for term in or_leaves:
        # We aggregate all the disjoint terms in or
        if term not in common_terms:
            combined_formula = F.Or(combined_formula, term)

    return [combined_formula]

def combine_formulas(formula_1, formula_2):
    new_formulas = []
    new_formulas.extend(combine_same_exclusive_op(formula_1, formula_2))
    new_formulas.extend(combine_disjoint_formulas(formula_1, formula_2))
    new_formulas.extend(combine_different_op(formula_1, formula_2))
    return new_formulas

def combine_beam_candidates(nodes_current_length):
    combinations = []
    for i in range(len(nodes_current_length) - 1):
        for j in range(i + 1, len(nodes_current_length)):
            first_node = nodes_current_length[i]
            second_node = nodes_current_length[j]
            if first_node != second_node:     
                new_nodes = combine_formulas(first_node, second_node)
                if len(new_nodes) > 0:
                    new_nodes = list(set(new_nodes)) # Remove duplicates
                    combinations.extend(new_nodes)
    return combinations

def combine_with_history(nodes_current_length, history):
        combinations = []
        for i in range(len(nodes_current_length)):
            node_current_length = nodes_current_length[i]
            current_length = len(node_current_length)
            for length_history in range(2,current_length):
                history_candidates = history.get(length_history, [])
                for history_candidate in history_candidates:
                     if node_current_length != history_candidate:     
                        new_nodes = combine_formulas(node_current_length, history_candidate)
                        if len(new_nodes) > 0:
                            new_nodes = list(set(new_nodes)) # Remove duplicates
                            combinations.extend(new_nodes)
        return combinations

def update_history(history_beam, new_nodes, current_length, max_length):
    for new_node in new_nodes:
        len_new_node = len(new_node)
        if len_new_node > current_length and len_new_node <= max_length:
            history_beam[len_new_node] = history_beam.get(len_new_node, []) + [new_node]
    return history_beam

def is_verified(formula, masks, device=torch.device("cpu")):
    compounds = get_or_compounds(formula)
    verified = True
    for compound in compounds:
        if isinstance(compound, F.And) and isinstance(compound.right, F.Not):
            left_compounds = get_or_compounds(compound.left)
            right_compounds = get_or_compounds(compound.right)
            verified = True
            for left in left_compounds:
                left_mask = mask_utils.get_formula_mask(left, masks, device=device)
                for right in right_compounds:
                    right_mask = mask_utils.get_formula_mask(right.val, masks, device=device)
                    iou = metrics.iou(left_mask, right_mask)
                    if iou == 0.0:
                        # This means that the AND NOT doesn't modify this component so it cannot be verified by data
                        verified = False
                        return False
        else:
            continue
    return verified
 

def add_from_discarded_nodes(discarded_nodes, masks, beam_info, num_spots, device):
    beam_masks = {label: mask for label, (mask, _) in beam_info.items()}
    top_discarded_node = None  
    to_add = {}
    combinations = []
    heapq.heapify(discarded_nodes)
    while discarded_nodes is not None and len(discarded_nodes) > 0 and len(to_add.keys()) < num_spots:
        node_combinations = []
        # Most promising candidate (higher IoU)
        top_discarded_node = heapq.heappop(discarded_nodes)
        neg_iou = top_discarded_node[0]
        pos_iou = -neg_iou
        top_discarded_label = top_discarded_node[2]

        # We need to recompute the mask, since storing in memory them would be to costly
        mask_top_discarded = mask_utils.get_formula_mask(
            top_discarded_node[2], masks, beam_masks, device
        )
    
        # Extract without removing discarded nodes with the same IoU
        same_iou_nodes = []
        neg_iou_other = neg_iou
        while discarded_nodes is not None and len(discarded_nodes) > 0 and neg_iou_other == neg_iou:
            other_discarded_node = heapq.heappop(discarded_nodes)
            neg_iou_other = other_discarded_node[0]
            if neg_iou_other == neg_iou:
                same_iou_nodes.append(other_discarded_node)
            else:
                # We put back the node with different IoU
                heapq.heappush(discarded_nodes, other_discarded_node)

        # Check if the top discarded node is functionally equivalent to any of the current beam nodes

        # Collect masks for all the same iou nodes
        same_iou_info = {}
        for same_iou_node in same_iou_nodes:
            mask_same_iou_node = mask_utils.get_formula_mask(
                same_iou_node[2], masks, beam_masks, device
            )
            same_iou_info[same_iou_node[2]] = (mask_same_iou_node, -same_iou_node[0])

        # Compute the labels that are equivalent to the top candidate
        equivalent_labels = extract_functional_equivalents(mask_top_discarded, pos_iou, same_iou_info)
        
        # Add back the nodes that are not equivalent labels
        for same_iou_node in same_iou_nodes:
            if same_iou_node[2] not in equivalent_labels:
                heapq.heappush(discarded_nodes, same_iou_node)
        
        # Manage logical equivalence
        if len(equivalent_labels) == 0:
            # This is a good candidate
            to_add[top_discarded_label] = (pos_iou, top_discarded_label, mask_top_discarded)
        else:
            # We have logical equivalent formulas in the set of same iou
            # Let's check the type of equivalence
            safe_to_add = True
            is_top_verified = None # to avoid recomputation
            for equivalent_label in equivalent_labels:
                type_1_or_2 = share_same_concepts(top_discarded_label, equivalent_label)
                if type_1_or_2:

                    if is_top_verified is None:
                        # The first time we compute it
                        is_top_verified = is_verified(top_discarded_label, masks, device=device)
                    
                    if not is_top_verified:
                        # We can check if the equivalent label is verified by data 
                        if is_verified(equivalent_label, masks, device=device):
                            # We discard the current top candidate but we add back to the discarded nodes the equivalent label 
                            heapq.heappush(discarded_nodes, (neg_iou, 'INDIVIDUAL', equivalent_label, None))
                            safe_to_add = False
                            break
                    # In all the other cases, it is safe to add and we can remove the equivalent label from the discarded nodes (i.e., not adding it back)
                else:           
                    if len(equivalent_label) < len(top_discarded_label):
                        # In this case, the equivalent label comes from the previous beam.
                        # If the current formula is verified, we can add it to the beam because it is more specific than the previous one and in general better
                        safe_to_add = True
                    elif len(equivalent_label) > len(top_discarded_label):
                        safe_to_add = False
                        heapq.heappush(discarded_nodes, (neg_iou, 'INDIVIDUAL', equivalent_label, None))
                    else:
                        # In this case, we have different concepts and it means the current data do not support an unambiguous explanation at this length
                        # Therefore, the top discarded node is not safe to add 

                        safe_to_add = False
                        # We try to combine them and to suggest them to the next length, if any
                        node_combinations.extend(combine_formulas(top_discarded_label, equivalent_label))
            
            if safe_to_add:
                # We can add this node to the beam 
                to_add[top_discarded_label] = (pos_iou, top_discarded_label, mask_top_discarded)
                
                # Add combinations of this node
                combinations.extend(node_combinations)

    return to_add, discarded_nodes, combinations


