from collections import Counter
import heapq
import queue as Q

import torch

from . import formula as F
from compositional import mask_utils, metrics, search_utils

#TODO TO REMOVE. DUPLICATE IN BEAM SEARCH
def beam_expand_node(frontier_node, *, candidate_labels, non_zero_labels, max_length, constraints=None):
    _, _, label, _ = frontier_node
    vals_formula = set(label.get_vals())
    next_frontier = []
    constraints_label = []
    if constraints is not None:
        for val in vals_formula:
            constraint_val = constraints[val]
            constraints_label.extend(constraint_val)         
    for candidate_term in candidate_labels:
        # Skip the candidate term if it is already in the label or in the constraints
        if candidate_term.val in vals_formula or candidate_term.val in constraints_label:
            continue
        #allowed_next_op = ["OR", "AND", "NOT"] if candidate_term.val < num_non_neighbors else ["OR", "AND"]
        for next_op in ["OR", "AND", "NOT"]:
            # A zero term cannot improve AND or OR label 
            if next_op != "NOT" and candidate_term.val not in non_zero_labels:
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

            next_frontier.append(candidate_formula)

    return next_frontier

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

def beam_expand_compound(frontier_node, *, candidate_labels, non_zero_labels, max_length, constraints=None):
    _, _, label, _ = frontier_node
    vals_formula = set(label.get_vals())
    next_frontier = []
    constraints_label = []
    if constraints is not None:
        for val in vals_formula:
            constraint_val = constraints[val]
            constraints_label.extend(constraint_val)         
    for candidate_term in candidate_labels:
        # Skip the candidate term if it is already in the label or in the constraints
        if candidate_term.val in vals_formula or candidate_term.val in constraints_label:
            continue
        for next_op in ["OR", "AND", "NOT"]:
            # A zero term cannot improve AND or OR label 
            if next_op != "NOT" and candidate_term.val not in non_zero_labels:
                continue

            # Build the candidate formula based on the next operation

            # Standard Incremental
            if next_op == "OR":
                candidate_formula = F.Or(label, candidate_term)
            elif next_op == "AND":
                candidate_formula = F.And(label, candidate_term)
            elif next_op == "NOT":
                candidate_formula = F.And(label, F.Not(candidate_term))
            else:
                raise ValueError(f"Unknown operation {next_op}")

            next_frontier.append(candidate_formula)
            
            # Compound
            if next_op == "AND" or next_op == "NOT":
                or_compounds = get_or_compounds(label)
                # We can search for a specialization of each compound. The OR case would be covered by beam search and history in this case. So we don't cover it here
                for i, compound in enumerate(or_compounds):
                    other_compounds_before = or_compounds[:i]
                    other_compounds_after = or_compounds[i+1:]
                    # attach the term to this compound and then chain the other compounds to it with the OR
                    if next_op == "AND":
                        candidate_compound = F.And(compound, candidate_term)
                    elif next_op == "NOT":
                        candidate_compound = F.And(compound, F.Not(candidate_term))
                    else:
                        raise ValueError(f"Unknown operation {next_op}")
                    if i == 0:
                        candidate_formula = candidate_compound
                    else:
                        candidate_formula = other_compounds_before[0]
                        for other_compound in other_compounds_before[1:]:
                            candidate_formula = F.Or(candidate_formula, other_compound)
                        # Now we chain the candidate compound with the other compounds after
                        candidate_formula = F.Or(candidate_formula, candidate_compound)
                    # Finally, we chain the candidate formula with the other compounds after
                    for other_compound in other_compounds_after:
                        candidate_formula = F.Or(candidate_formula, other_compound)
                    next_frontier.append(candidate_formula)
            elif next_op == "OR":
                and_compounds = get_and_compounds(label)
                # We can search for a generalization of each compound. The AND case would be covered by beam search and history in this case. So we don't cover it here
                for i, compound in enumerate(and_compounds):
                    other_compounds_before = and_compounds[:i]
                    other_compounds_after = and_compounds[i+1:]
                    # attach the term to this compound and then chain the other compounds to it with the AND
                    candidate_compound = F.Or(compound, candidate_term)
                    if i == 0:
                        candidate_formula = candidate_compound
                    else:
                        candidate_formula = other_compounds_before[0]
                        for other_compound in other_compounds_before[1:]:
                            candidate_formula = F.And(candidate_formula, other_compound)
                        # Now we chain the candidate compound with the other compounds after
                        candidate_formula = F.And(candidate_formula, candidate_compound)
                    # Finally, we chain the candidate formula with the other compounds after
                    for other_compound in other_compounds_after:
                        candidate_formula = F.And(candidate_formula, other_compound)
                    next_frontier.append(candidate_formula)
 
    return next_frontier

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

def combine_same_exclusive_op(left_formula, right_formula):
    left_op_list = list(set(left_formula.get_ops()))
    right_op_list = list(set(right_formula.get_ops()))
    if len(left_op_list) != len(right_op_list) or len(left_op_list) != 1:
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
    #assert isinstance(combined_formula, F.Leaf)
    for term in vals[1:]:
        #assert isinstance(term, F.Leaf) or isinstance(term, F.Not)
        combined_formula = op(combined_formula, term)

    # Chain using the opposite operator. Distributivity
    #assert not isinstance(combined_formula, F.Not) # We should not have NOT at this stage, but we can have it in the different terms that are distributed
    distributed_formula = apply_distributivity(left_formula, right_formula, op)

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
    if len(common_terms) == 0:
        # No intersection. This case is managed by combine disjoint
        return []
    
    # Here, given (a OR b) and (a AND c) I would like to obtain ((a AND c) OR b) by replacing a with its specialization in the first formula
    or_leaves = left_leaves if left_op == 'OR' else right_leaves
    combined_formula = left_formula if left_op == 'AND' else right_formula # The end formula will be always full preserved
    for term in or_leaves:
        # We aggregate all the disjoint terms in or
        if term not in common_terms:
            combined_formula = F.Or(combined_formula, term)

    #assert not isinstance(combined_formula, F.Not) # We should not have NOT at this stage, but we can have it in the different terms that are distributed
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
        #assert neg_iou <= 0, f"The IoU should be negative in the discarded nodes, but we have a node with positive IoU: {top_discarded_node}"
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
                    #assert set(top_discarded_label.get_ops()) == set(equivalent_label.get_ops()), f"CASE functional equivalence but different ops: {top_discarded_label} and {equivalent_label}"

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


def manage_logical_equivalence(candidate_label, candidate_mask, candidate_iou, concept_masks, formulas_info, equivalents_removed):
    assert isinstance(candidate_iou, float)
    device = candidate_mask.device
    relevant_info = formulas_info.copy()
    # Add the equivalent formulas that we have removed from the beam with the same iou
    if candidate_iou in equivalents_removed:
        additional_equivalents = equivalents_removed[candidate_iou]
        for add_equivalent in additional_equivalents:
            if add_equivalent not in relevant_info:
                mask_add_equivalent = mask_utils.get_formula_mask(
                    add_equivalent, concept_masks, device=device
                )
                relevant_info[add_equivalent] = (mask_add_equivalent, candidate_iou)
    equivalent_labels = extract_functional_equivalents(candidate_mask, candidate_iou, relevant_info)
    to_remove = set()
    to_add = {}
    combinations = []
    if len(equivalent_labels) == 0:
        # In this case we do not have any functional equivalent formula in the beam, we can add the candidate formula without removing any formula from the beam
        to_add[candidate_label] = (candidate_iou, candidate_label, candidate_mask)
        return to_add, to_remove, combinations
    
    #assert len(equivalent_labels) == 1, f"Multiple functional equivalent formulas in the beam for the candidate label {candidate_label} with iou {candidate_iou}: {equivalent_labels}"
    #print(f"Candidate label: {candidate_label}, candidate iou: {candidate_iou}")
    #print(f"Equivalent label: {equivalent_labels[0]}, equivalent iou: {formulas_info[equivalent_labels[0]][1]}")
    # In all the other case we need to manage the functional equivalence
    for equivalent_label in equivalent_labels:
        # We need to understand the type
        type_1_or_2 = share_same_concepts(candidate_label, equivalent_label)
        if type_1_or_2:
            #print("CASE type 1 or 2 equivalence: the candidate and the equivalent share the same concepts")
            #assert set(candidate_label.get_ops()) == set(equivalent_label.get_ops()), f"CASE functional equivalence but different ops: {candidate_label} and {equivalent_label}"
            # We can replace the equivalent node if it not verified by data and the candidate is verified by data   
            if not is_verified(equivalent_label, concept_masks, device=device) and is_verified(candidate_label, concept_masks, device=device):
                # Replace the equivalent formula in the beam with the candidate formula that is verified by data
                to_remove.add(equivalent_label)
                to_add[candidate_label] = (candidate_iou, candidate_label, candidate_mask) # Note: this can happen multiple time but the candidate label is always the same
        else:
            #print("CASE type 3 equivalence: the candidate and the equivalent do not share the same concepts")
            # In this case, we have different concepts and it means the current data do not support an unambiguous explanation at this length
            # Therefore, we should remove the formula from the beam and add it to the set of equivalent labels
            to_remove.add(equivalent_label)
            if len(equivalent_label) < len(candidate_label):
                # In this case, the equivalent label comes from the previous beam.
                # If the current formula is verified, we can add it to the beam because it is more specific than the previous one and in general better
                to_add[candidate_label] = (candidate_iou, candidate_label, candidate_mask)
            else:
                # We try to combine them and to suggest them to the next length, if any
                combinations.extend(combine_formulas(candidate_label, equivalent_label))
    if len(to_remove) > 0:
        # Remove from to remove the labels already removed before
        to_remove = set(to_remove) - set(equivalents_removed.get(candidate_iou, []))
    return to_add, to_remove, combinations


def beam_search_functional_aware(
    search_space,
    *,
    masks,
    beam_masks,
    bitmaps,
    beam_limit,
    previous_beam=None,
    use_logic_equivalence=True,
):
    """Perform the beam search on the search space.

    Args:
        search_space (list): A list of formulas.
        masks (dict): A dictionary of concept masks. Each mask is a tensor of
            shape (N, H, W).
        beam_masks (dict): A dictionary of labal masks of the formulas in the
        current beam. Each mask is a tensor of shape (N, H, W).
        bitmaps (torch.Tensor): A tensor of shape (N, H, W) where N is the
            number of sample.
        beam_limit (int): The beam size.
        previous_beam (dict): A dictionary of the beam formulas and their iou.

    Returns:
        current_beam_formulas (list): A list of formulas.
        current_beam_iou (list): A list of iou.
        visited_indices (int): The number of visited indices.
    """
    if previous_beam is None:
        previous_beam = {}
    current_beam = []
    current_beam_info = {}
    for (_, _, label, _), iou in previous_beam.items():
        label_mask = mask_utils.get_formula_mask(
            label, masks, beam_masks, bitmaps.device
        )
        current_beam_info[label] = (label_mask, iou)

    minimum = 0
    visited_indices = 0
    best_formula = None
    discarded_nodes = []
    beam_combinations = []

    # Init beam with previous best
    for node, v in previous_beam.items():
        if not current_beam or len(current_beam) < beam_limit:
            heapq.heappush(current_beam, node)
            minimum = current_beam[0][0]
        elif v > minimum:
            heapq.heapreplace(current_beam, node)
            minimum = current_beam[0][0]
    
    minimum = current_beam[0][0] if current_beam else 0
    equivalents_removed = {}
    recent_nodes = set()
    recent_eiou = -1

    #OR((103 AND (NOT 2)), 101)
    test_formula = F.Or(F.And(F.Leaf(103), F.Not(F.Leaf(2))), F.Leaf(101))

    while len(search_space) > 0:
        node = heapq.heappop(search_space)
        e_iou = -node[0]

        if current_beam and len(current_beam) >= beam_limit and e_iou < minimum:
            break
        
        candidate_formula = node[2]

        # # Memory Mechanism to avoid duplicates
        if e_iou == recent_eiou:
            if candidate_formula in recent_nodes:
                # Already visited. This could happen due to the history
                continue
            else:
                recent_nodes.add(candidate_formula)
        else:
            recent_nodes = set()
            recent_nodes.add(candidate_formula)
            recent_eiou = e_iou

        # skip equivalent formulas of the current beam
        if use_logic_equivalence and best_formula and hash(candidate_formula) == hash(best_formula):
            continue

        masks_formula = mask_utils.get_formula_mask(
            candidate_formula, masks, beam_masks, bitmaps.device
        )
        iou = metrics.iou(
            masks_formula, bitmaps
        )
        visited_indices += 1
        node = (iou.item(), node[1], node[2], None)

        if current_beam is None or len(current_beam) < beam_limit:
            to_add, to_remove, new_combinations = manage_logical_equivalence(candidate_formula, masks_formula, iou.item(), masks, current_beam_info, equivalents_removed)
            beam_combinations.extend(new_combinations)
            if  candidate_formula not in to_add and len(to_remove) > 0:
                # We keep track of these cases to avoid type 3 cases sneaking into the beam because there are no equivalent labels
                if iou.item() not in equivalents_removed:
                    equivalents_removed[iou.item()] = []
                equivalents_removed[iou.item()].append(candidate_formula)
            # if len(to_add) > 1 or len(to_remove) > 0:
            #     print("The beam is not full yet")
            #     print(f"To add: {[label for label in to_add.keys()]}")
            #     print(f"To remove: {to_remove}")
            #     print(f"Len current beam: {len(current_beam)}")
            #     print(f"New combinations: {new_combinations}")
            #     print(f"*****************************")
            for label_to_remove  in to_remove:
                equivalent_node = (iou.item(), node[1], label_to_remove, None)
                # Removal
                current_beam.remove(equivalent_node) # From beam 
                if label_to_remove == test_formula:
                    print("Removing the test formula from the beam place 5")
                    print(label_to_remove)
                    print(current_beam_info)
                del current_beam_info[label_to_remove] # From beam info
                heapq.heapify(current_beam) # Reorder beam after removal
            
            for label_to_add, (iou_to_add, label_to_add, mask_to_add) in to_add.items():
                # Addition
                heapq.heappush(current_beam, (iou_to_add, node[1], label_to_add, None)) # To beam
                if label_to_add == test_formula:
                    print("Adding the test formula to the beam place 4")
                    print(label_to_add)
                    print(current_beam_info)
                current_beam_info[label_to_add] = (mask_to_add, iou_to_add) # To beam info

            # Update minimum
            minimum = current_beam[0][0]

            # At this stage there are no discarded nodes so there is no need to do anything else
            #assert len(discarded_nodes) == 0, f"There should be no discarded nodes at this stage, but we have {len(discarded_nodes)} discarded nodes."

        elif iou > minimum:
            to_add, to_remove, new_combinations = manage_logical_equivalence(candidate_formula, masks_formula, iou.item(), masks, current_beam_info, equivalents_removed)
            beam_combinations.extend(new_combinations)
            if candidate_formula not in to_add and len(to_remove) > 0:
                # We keep track of these cases to avoid type 3 cases sneaking into the beam because there are no equivalent labels
                if iou.item() not in equivalents_removed:
                    equivalents_removed[iou.item()] = []
                equivalents_removed[iou.item()].append(candidate_formula)
            # if len(to_add) > 1 or len(to_remove) > 0:
            #     print("The beam is full")
            #     print(f"To add: {[label for label in to_add.keys()]}")
            #     print(f"To remove: {to_remove}")
            #     print(f"Len current beam: {len(current_beam)}")
            #     print(f"New combinations: {new_combinations}")
            #     print(f"*****************************")
            for label_to_remove  in to_remove:
                equivalent_node = (iou.item(), node[1], label_to_remove, None)
                if label_to_remove == test_formula:
                    print("Removing the test formula from the beam place 1")
                    print(label_to_remove)
                    print(current_beam_info)
                # Removal
                current_beam.remove(equivalent_node) # From beam 
                heapq.heapify(current_beam) # We need to heapify after removal
                del current_beam_info[label_to_remove] # From beam info
            
            for label_to_add, (iou_to_add, label_to_add, mask_to_add) in to_add.items():
                if label_to_add == test_formula:
                    print("Adding the test formula to the beam place 2")
                    print(label_to_add)
                    print(current_beam_info)
                # Addition
                heapq.heappush(current_beam, (iou_to_add, node[1], label_to_add, None)) # To beam
                current_beam_info[label_to_add] = (mask_to_add, iou_to_add) # To beam info

            # In this case we need to manage the case where the beam now is not full
            if len(current_beam) < beam_limit:
                # We can add the best candidate from the discarded nodes
                to_add_from_discarded, discarded_nodes, new_combinations = add_from_discarded_nodes(discarded_nodes, masks, current_beam_info, beam_limit - len(current_beam), bitmaps.device)
                beam_combinations.extend(new_combinations)
                
                # if len(to_add_from_discarded.keys()) > 0:
                #     print("Added from discarded nodes to fill the beam after removal")
                #     print(f"To add from discarded: {[label for label in to_add_from_discarded.keys()]}")
                #     print(f"Len current beam: {len(current_beam)}")
                #     print(f"New combinations: {new_combinations}")
                #     print(f"*****************************")

                for label_to_add, (iou_to_add, label_to_add, mask_to_add) in to_add_from_discarded.items():
                    if label_to_add == test_formula:
                        print("Adding the test formula from discarded nodes to fill the beam after removal place 3")
                        print(label_to_add)
                        print(current_beam_info)
                    heapq.heappush(current_beam, (iou_to_add, node[1], label_to_add, None)) # To beam
                    current_beam_info[label_to_add] = (mask_to_add, iou_to_add) # To beam info

            # Remove the minimum nodes until we are under the beam limit
            while len(current_beam) > beam_limit:
                # if current_beam[0][0] > minimum:
                #     # We should not remove nodes that are better than the minimum
                #     raise ValueError(f"We should not remove nodes that are better than the minimum, but we have a node with iou {current_beam[0][0]} that is better than the minimum {minimum}")
                #     break
                removed_node = heapq.heappop(current_beam)
                removed_label = removed_node[2]
                del current_beam_info[removed_label]
                # Add the removed node to the discarded nodes
                heapq.heappush(discarded_nodes, (-removed_node[0], removed_node[1], removed_node[2], removed_node[3]))

            # Update minimum
            minimum = current_beam[0][0]
        else:
            # Add the node to the discarded nodes list
            discarded_nodes.append((-iou.item(), node[1], node[2], node[3]))
    return [node for node in current_beam], visited_indices, beam_combinations

def explore_beam_frontier_compound(
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
    beam_size=5,
    length=3,
    labels=None,
    constraints=None,
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

    max_length = length
    top_k_combo = beam_size
 
    # Extract first beam and candidate concepts
    candidate_labels = [F.Leaf(c) for c in range(len(masks))]
    
    iou_atoms = {k: search_utils.analyze_final_node(k, masks, bitmaps) for k in candidate_labels}

    iou_atoms = Counter(iou_atoms)
    non_iou_labels =  [lab.val for lab, iou in iou_atoms.items() if iou > 0]

    first_beam_num = min (len(iou_atoms), beam_size)
    # added str(lab) instead of None to break equivalence and replicate bug
    beam_atoms = {
        (iou, 'INDIVIDUAL', lab, None): iou
        for lab, iou in iou_atoms.most_common(first_beam_num)
        if iou > 0
    }

    leaf_mapping = {F.Leaf(c): c for c in range(len(labels))}

    # Collect the history of best candidates at every length
    history_beam = {}

    # Update the history
    history_beam[1] = [ lab for lab, iou in iou_atoms.most_common(top_k_combo)  if iou > 0] # Store the top_k_combo best atoms in the history beam
    history_combinations = combine_beam_candidates(history_beam[1]) # Creaate the combinations for the next iterations
    history_beam = update_history(history_beam, history_combinations, current_length=1, max_length=max_length) # Add the combinations of the initial beam to the history

    # print("Top 30")
    # i = 1
    # for value,iou in iou_atoms.most_common(top_k_combo):
    #     label_mask = utils.sparse_to_torch(mask_utils.get_formula_mask(
    #         value, masks
    #     ))
    #     print(f"Concept {value}: {F.get_formula_str(value, labels)}-  Mask Sum {label_mask.sum().item()}, Bitmaps: {bitmaps.sum().item()} IoU {iou}")
    #     if i == beam_size:
    #         print("---------------- END OF BEAM ----------------")
    #     i += 1
    # Beam Search
    total_visited = 0
    expanded_nodes = 0
    estimated_nodes = 0
    beam_masks = {}
    beam = beam_atoms.copy()
    
    for index_loop in range(1, max_length):
        previous_beam_length = index_loop
        current_beam_length = index_loop + 1
    
        beam_candidates = []
        for node in beam.keys():
            # Expand only nodes that were not expanded before
            if len(node[2]) == previous_beam_length:
                # Expand the node to get the next frontier
                #assert constraints is None
                next_frontier = beam_expand_compound(node, candidate_labels=candidate_labels, non_zero_labels=non_iou_labels, max_length=max_length, constraints=constraints)
                expanded_nodes += 1
                beam_candidates.extend(next_frontier)
        
        # # Add nodes by combining current frontier with history
        history_candidates = history_beam.get(current_beam_length, [])
        beam_candidates.extend(history_candidates)
        
        # Compute the estimation for the beam candidates
        estimated_nodes += len(beam_candidates)
        
        beam_estimations, _ = search_utils.update_frontier(
            past_frontier=None, new_nodes=beam_candidates,
            label_mapping=label_mapping, heuristic='sample',
            heuristic_info=heuristic_info, max_improvement=max_improvement,
            num_hits=num_hits, max_size_mask=max_size_mask,
            length=max_length, global_min_threshold=0.0, disjoint_info=disjoint_info
        )

        # Beam Search
        next_beam, visited_nodes, beam_combinations = beam_search_functional_aware(
            beam_estimations,
            masks=masks,
            previous_beam=beam,
            beam_masks=beam_masks,
            bitmaps=bitmaps,
            beam_limit=max(beam_size, top_k_combo)
        )
        # Add the combinations of the current beam to the history
        history_beam = update_history(history_beam, beam_combinations, current_length=current_beam_length, max_length=max_length) # Add the combinations of the initial beam to the history
        
        total_visited = total_visited + visited_nodes

        for node in next_beam:
            if len(node[2]) == current_beam_length:
                beam.update({(node[0], 'INDIVIDUAL', node[2], None): node[0]})
        
        beam = dict(Counter(beam).most_common(beam_size))
        
        # Update History
        history_beam[current_beam_length] = [label for _, _, label, _ in next_beam if len(label) == current_beam_length]
        if current_beam_length < max_length:
            combinations_within_beam = combine_beam_candidates(history_beam[current_beam_length]) # Creaate the combinations for the next iterations
            history_combinations = combine_with_history(history_beam[current_beam_length], history_beam) # Create the combinations of the current beam with the previous history
            total_combinations = combinations_within_beam + history_combinations
            # Update the history
            history_beam = update_history(history_beam, total_combinations, current_length=current_beam_length, max_length=max_length) # Add the combinations of the initial beam to the history
        
        beam_masks, (heuristic_info, label_mapping) = search_utils.get_beam_info(beam.keys(), masks, beam_masks, heuristic_info, leaf_mapping, bitmaps, max_length)

       
        
        # print("Top 30")
        # i = 1
        # for iou, _, value, _ in sorted(next_beam, key=lambda x: x[0], reverse=True):
        #     label_mask = utils.sparse_to_torch(mask_utils.get_formula_mask(
        #         value, masks
        #     ))
        #     print(f"Concept {value}: {F.get_formula_str(value, labels)}-  Mask Sum {label_mask.sum().item()}, Bitmaps: {bitmaps.sum().item()} IoU {iou}")
        #     if i == beam_size:
        #         print("---------------- END OF BEAM ----------------")
        #     i += 1
        # print("Next beam")
        # for (_, _, value, _), iou in beam.items():
        #     label_mask = utils.sparse_to_torch(mask_utils.get_formula_mask(
        #         value, masks
        #     ))
        #     print(f"Concept {value}: {F.get_formula_str(value, labels)}-  Mask Sum {label_mask.sum().item()}, Bitmaps: {bitmaps.sum().item()} IoU {iou}")
    best_node, best_iou = Counter(beam).most_common(1)[0]
    best_label = best_node[2]
    return best_label, best_iou, total_visited, expanded_nodes, estimated_nodes