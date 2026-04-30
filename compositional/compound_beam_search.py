from collections import Counter
import heapq

from . import formula as F
from compositional import mask_utils, metrics, search_utils
from . import non_incremental_utils


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
            if len(candidate_formula) > 2:
                if next_op == "AND" or next_op == "NOT":
                    or_compounds = non_incremental_utils.get_or_compounds(label)
                    if len(or_compounds) > 1:
                        # len() == 0 is already covered by incremental
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
                    and_compounds = non_incremental_utils.get_and_compounds(label)
                    if len(and_compounds) > 1:
                        # len() == 0 is already covered by incremental
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

def manage_logical_equivalence_old(candidate_label, candidate_mask, candidate_iou, concept_masks, formulas_info, equivalents_removed, block_type_3=True):
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
    equivalent_labels = non_incremental_utils.extract_functional_equivalents(candidate_mask, candidate_iou, relevant_info)
    to_remove = set()
    to_add = {}
    combinations = []
    if len(equivalent_labels) == 0:
        # In this case we do not have any functional equivalent formula in the beam, we can add the candidate formula without removing any formula from the beam
        to_add[candidate_label] = (candidate_iou, candidate_label, candidate_mask)
        return to_add, to_remove, combinations
    # In all the other case we need to manage the functional equivalence
    for equivalent_label in equivalent_labels:
        # We need to understand the type
        type_1_or_2 = non_incremental_utils.share_same_concepts(candidate_label, equivalent_label)
        if type_1_or_2:
            # We can replace the equivalent node if it not verified by data and the candidate is verified by data   
            if not non_incremental_utils.is_verified(equivalent_label, concept_masks, device=device) and non_incremental_utils.is_verified(candidate_label, concept_masks, device=device):
                # Replace the equivalent formula in the beam with the candidate formula that is verified by data
                to_remove.add(equivalent_label)
                to_add[candidate_label] = (candidate_iou, candidate_label, candidate_mask) # Note: this can happen multiple time but the candidate label is always the same
        else:
            # In this case, we have different concepts and it means the current data do not support an unambiguous explanation at this length
            # Therefore, we should remove the formula from the beam and add it to the set of equivalent labels
            if block_type_3:
                print(" QUI NON DVREBBE 4")
                to_remove.add(equivalent_label)
                if len(equivalent_label) < len(candidate_label):
                    if equivalent_label != candidate_label.left:
                        # In this case, the equivalent label comes from the previous beam.
                        # If the current formula is verified, we can add it to the beam because it is more specific than the previous one and in general better
                        to_add[candidate_label] = (candidate_iou, candidate_label, candidate_mask)
                else:
                    
                    # We try to combine them and to suggest them to the next length, if any
                    combinations.extend(non_incremental_utils.combine_formulas(candidate_label, equivalent_label))
            else:
                if len(equivalent_label) < len(candidate_label) and equivalent_label == candidate_label.left:
                    # 
                    continue
                # In this case we add both the candidate label and the equivalent label and include also the combinations
                to_add[candidate_label] = (candidate_iou, candidate_label, candidate_mask)
                combinations.extend(non_incremental_utils.combine_formulas(candidate_label, equivalent_label))
    if len(to_remove) > 0:
        remove_pre = to_remove.copy()
        # Remove from to remove the labels already removed before
        to_remove = set(to_remove) - set(equivalents_removed.get(candidate_iou, []))
        if len(remove_pre) != len(to_remove):
            print(" QUI NON DVREBBE 5")
    return to_add, to_remove, combinations


def beam_search_functional_aware_old(
    search_space,
    *,
    masks,
    beam_masks,
    bitmaps,
    beam_limit,
    previous_beam=None,
    use_logic_equivalence=True,
    block_type_3=False,
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
        block_type_3 (bool): Whether to block type 3 equivalences.

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
    print("OLD", block_type_3 )
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
    while len(search_space) > 0:
        node = heapq.heappop(search_space)
        e_iou = -node[0]

        if current_beam and len(current_beam) >= beam_limit and e_iou <= minimum:
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
        node = (iou, node[1], node[2], None)

        if current_beam is None or len(current_beam) < beam_limit:
            to_add, to_remove, new_combinations = manage_logical_equivalence_old(candidate_formula, masks_formula, iou, masks, current_beam_info, equivalents_removed, block_type_3=block_type_3)
            beam_combinations.extend(new_combinations)
            if  candidate_formula not in to_add and len(to_remove) > 0:
                print(" QUI NON DVREBBE 1")
                # We keep track of these cases to avoid type 3 cases sneaking into the beam because there are no equivalent labels
                if iou not in equivalents_removed:
                    equivalents_removed[iou] = []
                equivalents_removed[iou].append(candidate_formula)
            for label_to_remove  in to_remove:
                equivalent_node = (iou, node[1], label_to_remove, None)
                # Removal
                current_beam.remove(equivalent_node) # From beam 
                del current_beam_info[label_to_remove] # From beam info
                heapq.heapify(current_beam) # Reorder beam after removal
            
            for label_to_add, (iou_to_add, label_to_add, mask_to_add) in to_add.items():
                # Addition
                heapq.heappush(current_beam, (iou_to_add, node[1], label_to_add, None)) # To beam
                current_beam_info[label_to_add] = (mask_to_add, iou_to_add) # To beam info

            # Update minimum
            minimum = current_beam[0][0]

        elif iou > minimum:
            to_add, to_remove, new_combinations = manage_logical_equivalence_old(candidate_formula, masks_formula, iou, masks, current_beam_info, equivalents_removed, block_type_3=block_type_3)
            beam_combinations.extend(new_combinations)
            if candidate_formula not in to_add and len(to_remove) > 0:
                print(" QUI NON DVREBBE 2")
                # We keep track of these cases to avoid type 3 cases sneaking into the beam because there are no equivalent labels
                if iou not in equivalents_removed:
                    equivalents_removed[iou] = []
                equivalents_removed[iou].append(candidate_formula)
            for label_to_remove  in to_remove:
                equivalent_node = (iou, node[1], label_to_remove, None)
                # Removal
                current_beam.remove(equivalent_node) # From beam 
                heapq.heapify(current_beam) # We need to heapify after removal
                del current_beam_info[label_to_remove] # From beam info
            
            for label_to_add, (iou_to_add, label_to_add, mask_to_add) in to_add.items():
                # Addition
                heapq.heappush(current_beam, (iou_to_add, node[1], label_to_add, None)) # To beam
                current_beam_info[label_to_add] = (mask_to_add, iou_to_add) # To beam info

            # In this case we need to manage the case where the beam now is not full
            if len(current_beam) < beam_limit:
                print(" QUI NON DVREBBE 3")
                # We can add the best candidate from the discarded nodes
                to_add_from_discarded, discarded_nodes, new_combinations = non_incremental_utils.add_from_discarded_nodes(discarded_nodes, masks, current_beam_info, beam_limit - len(current_beam), bitmaps.device)
                beam_combinations.extend(new_combinations)

                for label_to_add, (iou_to_add, label_to_add, mask_to_add) in to_add_from_discarded.items():
                    heapq.heappush(current_beam, (iou_to_add, node[1], label_to_add, None)) # To beam
                    current_beam_info[label_to_add] = (mask_to_add, iou_to_add) # To beam info

            # Remove the minimum nodes until we are under the beam limit
            while len(current_beam) > beam_limit:
                #TODO
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
            discarded_nodes.append((-iou, node[1], node[2], node[3]))
    return [node for node in current_beam], visited_indices, beam_combinations



def manage_logical_equivalence_loose(candidate_label, candidate_mask, candidate_iou, concept_masks, formulas_info):
    device = candidate_mask.device

    equivalent_labels = non_incremental_utils.extract_functional_equivalents(candidate_mask, candidate_iou, formulas_info)
    to_remove = set()
    to_add = {}
    combinations = []
    if len(equivalent_labels) == 0:
        # In this case we do not have any functional equivalent formula in the beam, we can add the candidate formula without removing any formula from the beam
        to_add[candidate_label] = (candidate_iou, candidate_label, candidate_mask)
        return to_add, to_remove, combinations
    # In all the other case we need to manage the functional equivalence
    for equivalent_label in equivalent_labels:
        # We need to understand the type
        type_1_or_2 = non_incremental_utils.share_same_concepts(candidate_label, equivalent_label)
        if type_1_or_2:
            # We can replace the equivalent node if it not verified by data and the candidate is verified by data   
            if not non_incremental_utils.is_verified(equivalent_label, concept_masks, device=device) and non_incremental_utils.is_verified(candidate_label, concept_masks, device=device):
                # Replace the equivalent formula in the beam with the candidate formula that is verified by data
                to_remove.add(equivalent_label)
                to_add[candidate_label] = (candidate_iou, candidate_label, candidate_mask) # Note: this can happen multiple time but the candidate label is always the same
        else:
            # In this case, we have different concepts and it means the current data do not support an unambiguous explanation at this length
            # Therefore, we should remove the formula from the beam and add it to the set of equivalent labels
            if len(equivalent_label) < len(candidate_label) and equivalent_label == candidate_label.left:
                # 
                continue
            # In this case we add both the candidate label and the equivalent label and include also the combinations
            to_add[candidate_label] = (candidate_iou, candidate_label, candidate_mask)
            combinations.extend(non_incremental_utils.combine_formulas(candidate_label, equivalent_label))
    return to_add, to_remove, combinations

def manage_logical_equivalence_strict(candidate_label, candidate_mask, candidate_iou, concept_masks, formulas_info, equivalents_removed):
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
    equivalent_labels = non_incremental_utils.extract_functional_equivalents(candidate_mask, candidate_iou, relevant_info)
    to_remove = set()
    to_add = {}
    combinations = []
    if len(equivalent_labels) == 0:
        # In this case we do not have any functional equivalent formula in the beam, we can add the candidate formula without removing any formula from the beam
        to_add[candidate_label] = (candidate_iou, candidate_label, candidate_mask)
        return to_add, to_remove, combinations
    # In all the other case we need to manage the functional equivalence
    for equivalent_label in equivalent_labels:
        # We need to understand the type
        type_1_or_2 = non_incremental_utils.share_same_concepts(candidate_label, equivalent_label)
        if type_1_or_2:
            # We can replace the equivalent node if it not verified by data and the candidate is verified by data   
            if not non_incremental_utils.is_verified(equivalent_label, concept_masks, device=device) and non_incremental_utils.is_verified(candidate_label, concept_masks, device=device):
                # Replace the equivalent formula in the beam with the candidate formula that is verified by data
                to_remove.add(equivalent_label)
                to_add[candidate_label] = (candidate_iou, candidate_label, candidate_mask) # Note: this can happen multiple time but the candidate label is always the same
        else:
            # In this case, we have different concepts and it means the current data do not support an unambiguous explanation at this length
            # Therefore, we should remove the formula from the beam and add it to the set of equivalent labels
            to_remove.add(equivalent_label)
            if len(equivalent_label) < len(candidate_label):
                if equivalent_label == candidate_label.left:
                    continue
                else:
                    # In this case, the equivalent label comes from the previous beam.
                    # If the current formula is verified, we can add it to the beam because it is more specific than the previous one and in general better
                    to_add[candidate_label] = (candidate_iou, candidate_label, candidate_mask)
            else:
                # We try to combine them and to suggest them to the next length, if any
                combinations.extend(non_incremental_utils.combine_formulas(candidate_label, equivalent_label))
    if len(to_remove) > 0:
        # Remove from to remove the labels already removed before
        to_remove = set(to_remove) - set(equivalents_removed.get(candidate_iou, []))
    return to_add, to_remove, combinations



def beam_search_functional_aware_strict(
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
    while len(search_space) > 0:
        node = heapq.heappop(search_space)
        e_iou = -node[0]

        if current_beam and len(current_beam) >= beam_limit and e_iou <= minimum:
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
        node = (iou, node[1], node[2], None)

        if current_beam is None or len(current_beam) < beam_limit:
            to_add, to_remove, new_combinations = manage_logical_equivalence_strict(candidate_formula, masks_formula, iou, masks, current_beam_info, equivalents_removed)
            beam_combinations.extend(new_combinations)
            if  candidate_formula not in to_add and len(to_remove) > 0:
                # We keep track of these cases to avoid type 3 cases sneaking into the beam because there are no equivalent labels
                if iou not in equivalents_removed:
                    equivalents_removed[iou] = []
                equivalents_removed[iou].append(candidate_formula)
            for label_to_remove  in to_remove:
                equivalent_node = (iou, node[1], label_to_remove, None)
                # Removal
                current_beam.remove(equivalent_node) # From beam 
                del current_beam_info[label_to_remove] # From beam info
                heapq.heapify(current_beam) # Reorder beam after removal
            
            for label_to_add, (iou_to_add, label_to_add, mask_to_add) in to_add.items():
                # Addition
                heapq.heappush(current_beam, (iou_to_add, node[1], label_to_add, None)) # To beam
                current_beam_info[label_to_add] = (mask_to_add, iou_to_add) # To beam info

            # Update minimum
            minimum = current_beam[0][0]

        elif iou > minimum:
            to_add, to_remove, new_combinations = manage_logical_equivalence_strict(candidate_formula, masks_formula, iou, masks, current_beam_info, equivalents_removed)
            beam_combinations.extend(new_combinations)
            if candidate_formula not in to_add and len(to_remove) > 0:
                # We keep track of these cases to avoid type 3 cases sneaking into the beam because there are no equivalent labels
                if iou not in equivalents_removed:
                    equivalents_removed[iou] = []
                equivalents_removed[iou].append(candidate_formula)
            for label_to_remove  in to_remove:
                equivalent_node = (iou, node[1], label_to_remove, None)
                # Removal
                current_beam.remove(equivalent_node) # From beam 
                heapq.heapify(current_beam) # We need to heapify after removal
                del current_beam_info[label_to_remove] # From beam info
            
            for label_to_add, (iou_to_add, label_to_add, mask_to_add) in to_add.items():
                # Addition
                heapq.heappush(current_beam, (iou_to_add, node[1], label_to_add, None)) # To beam
                current_beam_info[label_to_add] = (mask_to_add, iou_to_add) # To beam info

            # In this case we need to manage the case where the beam now is not full
            if len(current_beam) < beam_limit:
                # We can add the best candidate from the discarded nodes
                to_add_from_discarded, discarded_nodes, new_combinations = non_incremental_utils.add_from_discarded_nodes(discarded_nodes, masks, current_beam_info, beam_limit - len(current_beam), bitmaps.device)
                beam_combinations.extend(new_combinations)

                for label_to_add, (iou_to_add, label_to_add, mask_to_add) in to_add_from_discarded.items():
                    heapq.heappush(current_beam, (iou_to_add, node[1], label_to_add, None)) # To beam
                    current_beam_info[label_to_add] = (mask_to_add, iou_to_add) # To beam info

            # Remove the minimum nodes until we are under the beam limit
            while len(current_beam) > beam_limit:
                #TODO
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
            discarded_nodes.append((-iou, node[1], node[2], node[3]))
    return [node for node in current_beam], visited_indices, beam_combinations


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
    beam_combinations = []

    #print("QUI")

    # Init beam with previous best
    for node, v in previous_beam.items():
        if not current_beam or len(current_beam) < beam_limit:
            heapq.heappush(current_beam, node)
            minimum = current_beam[0][0]
        elif v > minimum:
            heapq.heapreplace(current_beam, node)
            minimum = current_beam[0][0]
    
    minimum = current_beam[0][0] if current_beam else 0
    recent_nodes = set()
    recent_eiou = -1
    while len(search_space) > 0:
        node = heapq.heappop(search_space)
        e_iou = -node[0]

        if current_beam and len(current_beam) >= beam_limit and e_iou <= minimum:
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
        node = (iou, node[1], node[2], None)

        if current_beam is None or len(current_beam) < beam_limit:
            to_add, to_remove, new_combinations = manage_logical_equivalence_loose(candidate_formula, masks_formula, iou, masks, current_beam_info)
            beam_combinations.extend(new_combinations)
            for label_to_remove  in to_remove:
                equivalent_node = (iou, node[1], label_to_remove, None)
                # Removal
                current_beam.remove(equivalent_node) # From beam 
                del current_beam_info[label_to_remove] # From beam info
                heapq.heapify(current_beam) # Reorder beam after removal
            
            for label_to_add, (iou_to_add, label_to_add, mask_to_add) in to_add.items():
                # Addition
                heapq.heappush(current_beam, (iou_to_add, node[1], label_to_add, None)) # To beam
                current_beam_info[label_to_add] = (mask_to_add, iou_to_add) # To beam info

            # Update minimum
            minimum = current_beam[0][0]

        elif iou > minimum:
            to_add, to_remove, new_combinations = manage_logical_equivalence_loose(candidate_formula, masks_formula, iou, masks, current_beam_info)
            beam_combinations.extend(new_combinations)
            for label_to_remove  in to_remove:
                equivalent_node = (iou, node[1], label_to_remove, None)
                # Removal
                current_beam.remove(equivalent_node) # From beam 
                heapq.heapify(current_beam) # We need to heapify after removal
                del current_beam_info[label_to_remove] # From beam info
            
            for label_to_add, (iou_to_add, label_to_add, mask_to_add) in to_add.items():
                # Addition
                heapq.heappush(current_beam, (iou_to_add, node[1], label_to_add, None)) # To beam
                current_beam_info[label_to_add] = (mask_to_add, iou_to_add) # To beam info


            # Remove the minimum nodes until we are under the beam limit
            while len(current_beam) > beam_limit:
                #TODO
                # if current_beam[0][0] > minimum:
                #     # We should not remove nodes that are better than the minimum
                #     raise ValueError(f"We should not remove nodes that are better than the minimum, but we have a node with iou {current_beam[0][0]} that is better than the minimum {minimum}")
                #     break
                removed_node = heapq.heappop(current_beam)
                removed_label = removed_node[2]
                del current_beam_info[removed_label]
                # Add the removed node to the discarded nodes

            # Update minimum
            minimum = current_beam[0][0]
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
    block_type_3=True,
    first_beam_size=None
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

    beam_function = beam_search_functional_aware_strict if block_type_3 else beam_search_functional_aware

    #beam_function = beam_search_functional_aware_old
    max_length = length
    top_k_combo = beam_size
 
    # Extract first beam and candidate concepts
    candidate_labels = [F.Leaf(c) for c in range(len(masks))]
    
    iou_atoms = {k: search_utils.analyze_final_node(k, masks, bitmaps) for k in candidate_labels}

    iou_atoms = Counter(iou_atoms)
    non_iou_labels =  [lab.val for lab, iou in iou_atoms.items() if iou > 0]

    first_beam_num = min (len(iou_atoms), beam_size if first_beam_size is None else first_beam_size)
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
    history_combinations = non_incremental_utils.combine_beam_candidates(history_beam[1]) # Creaate the combinations for the next iterations
    history_beam = non_incremental_utils.update_history(history_beam, history_combinations, current_length=1, max_length=max_length) # Add the combinations of the initial beam to the history

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
        next_beam, visited_nodes, beam_combinations = beam_function(
            beam_estimations,
            masks=masks,
            previous_beam=beam,
            beam_masks=beam_masks,
            bitmaps=bitmaps,
            beam_limit=max(beam_size, top_k_combo),
        )
        # Add the combinations of the current beam to the history
        history_beam = non_incremental_utils.update_history(history_beam, beam_combinations, current_length=current_beam_length, max_length=max_length) # Add the combinations of the initial beam to the history
        
        total_visited = total_visited + visited_nodes

        for node in next_beam:
            if len(node[2]) == current_beam_length:
                beam.update({(node[0], 'INDIVIDUAL', node[2], None): node[0]})
        
        beam = dict(Counter(beam).most_common(beam_size))
        
        # Update History
        history_beam[current_beam_length] = [label for _, _, label, _ in next_beam if len(label) == current_beam_length]
        if current_beam_length < max_length:
            combinations_within_beam = non_incremental_utils.combine_beam_candidates(history_beam[current_beam_length]) # Creaate the combinations for the next iterations
            history_combinations = non_incremental_utils.combine_with_history(history_beam[current_beam_length], history_beam) # Create the combinations of the current beam with the previous history
            total_combinations = combinations_within_beam + history_combinations
            # Update the history
            history_beam = non_incremental_utils.update_history(history_beam, total_combinations, current_length=current_beam_length, max_length=max_length) # Add the combinations of the initial beam to the history
        
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
