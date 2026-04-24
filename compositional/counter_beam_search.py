from collections import Counter
import heapq
import queue as Q
import torch

from . import formula as F
from . import utils
from compositional import mask_utils, metrics

def explore_counter_beam_frontier(
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
    counter_variant=False,
    diff_threshold=0.1
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

    # Extract first beam and candidate concepts
    candidate_labels = [F.Leaf(c) for c in range(len(masks))]
    
    # # Zero the masks that are always positive. These are uninformative tokens since we don't have the counter example
    # for concept in candidate_labels:
    #     label_mask = masks[concept.val]
    #     if (torch.sum(label_mask) == label_mask.numel()):
    #         print("Zeroing concept ", concept.val, " - ", labels[concept.val], " since it is always active.")
    #         masks[concept.val] = torch.zeros_like(label_mask)
    
    adv_bitmaps = ~bitmaps

    diff_atoms = {}
    iou_atoms = {}
    non_zero_labels = []
    for k in candidate_labels:
        concept_mask = mask_utils.get_formula_mask(k, masks).to(bitmaps.device)
        concept_iou = metrics.iou(concept_mask, bitmaps).item()
        adv_iou = metrics.iou(concept_mask, adv_bitmaps).item()
        diff = metrics.diff_ratio_iou_adv_iou(concept_iou, adv_iou)
        if concept_iou > 0 and diff > diff_threshold:
            iou_atoms[k] = concept_iou
            diff_atoms[k] = diff
        if concept_iou > 0:
            non_zero_labels.append(k.val)

    first_beam_num = min (len(iou_atoms), beam_size)

    iou_atoms = Counter(iou_atoms)
    beam_atoms = {
        (iou, 'INDIVIDUAL', lab, None): iou
        for lab, iou in iou_atoms.most_common(first_beam_num)
    }

    # No concepts satisfying the consttrtaintta
    if len(beam_atoms) == 0:
        return None, 0.0, 0, 0, 0
    
    # for (_, _, value, _), iou in beam_atoms.items():
    #     # label_mask = utils.sparse_to_torch(mask_utils.get_formula_mask(
    #     #     value, masks
    #     # )).to(bitmaps.device)
    #     # print(bitmaps.device)
    #     # adv_iou = metrics.iou(label_mask, adv_bitmaps).item()
    #     # print(f"Concept {value} - {labels[value.val]} -  ({value.val - 4087}): Mask Sum {label_mask.sum().item()}, Bitmaps: {bitmaps.sum().item()} IoU {iou} Adv IoU {adv_iou} Diff {diff}")
    #     print(f"Concept {value} - {labels[value.val]} -  ({value.val}): IoU {iou} Diff {diff_atoms[value]}")
    # exit()
    minimum_threshold = min(beam_atoms.values())

    
    # Beam Search
    leaf_mapping = {F.Leaf(c): c for c in range(len(labels))}
    total_visited = 0
    expanded_nodes = 0
    estimated_nodes = 0
    beam_masks = {}
    beam = beam_atoms.copy()
    for index_loop in range(1, length):
        beam_candidates = []
        for node in beam.keys():
            # Expand only nodes that were not expanded before
            if len(node[2]) == index_loop:
                # Expand the node to get the next frontier
                next_frontier = beam_expand_node(node, candidate_labels=candidate_labels, non_zero_labels=non_zero_labels, max_length=length, constraints=constraints)
                expanded_nodes += 1
                beam_candidates.extend(next_frontier)
        # Remove duplicates like in MMESH
        beam_candidates = list(set(beam_candidates))
        
        estimated_nodes += len(beam_candidates)
        # Compute the estimation for the next frontier
        beam_estimations, _ = update_frontier(
            past_frontier=None, new_nodes=beam_candidates,
            label_mapping=label_mapping, heuristic='sample',
            heuristic_info=heuristic_info, max_improvement=max_improvement,
            num_hits=num_hits, max_size_mask=max_size_mask,
            length=length, global_min_threshold=minimum_threshold, disjoint_info=disjoint_info
        )
        next_beam, visited_nodes = counter_beam_search(
            beam_estimations,
            masks=masks,
            previous_beam=beam,
            beam_masks=beam_masks,
            bitmaps=bitmaps,
            beam_limit=beam_size,
            diff_threshold=diff_threshold,
            concept_diff_dict=diff_atoms
        )
        
        total_visited = total_visited + visited_nodes
        for node in next_beam:
            if len(node[2]) == index_loop + 1:
                beam.update({(node[0], 'INDIVIDUAL', node[2], str(node[2])): node[0]})
        beam = dict(Counter(beam).most_common(beam_size))
        # print(f"After loop {index_loop}, beam is:")
        # print(analyze_beam(beam_atoms.keys(), masks, bitmaps))
        beam_masks, (heuristic_info, label_mapping) = get_beam_info(beam.keys(), masks, beam_masks, heuristic_info, leaf_mapping, bitmaps, length)
        # print("Next beam")
        # for (_, _, value, _), iou in beam.items():
        #     label_mask = utils.sparse_to_torch(mask_utils.get_formula_mask(
        #         value, masks
        #     ))
        #     print(f"Concept {value}: {F.get_formula_str(value, labels)}-  Mask Sum {label_mask.sum().item()}, Bitmaps: {bitmaps.sum().item()} IoU {iou}")
        # exit()
    best_node, best_iou = Counter(beam).most_common(1)[0]
    best_label = best_node[2]
    return best_label, best_iou, total_visited, expanded_nodes, estimated_nodes

def get_counter_type(concept_iou, adv_iou, tau=0.00):
    threshold_interpretable = 0.04
    threshold_diff = 0.1
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

def counter_analyze_final_node(label_node, masks, bitmaps):
    label_mask = utils.sparse_to_torch(mask_utils.get_formula_mask(
            label_node, masks
        )).to(bitmaps.device)
    iou = metrics.counter_iou(label_mask=label_mask, bitmaps=bitmaps, counter_bitmaps=~bitmaps)
    return iou

def beam_expand_node_compound(frontier_node, *, candidate_labels, non_zero_labels, max_length, constraints=None):
    _, _, label, _ = frontier_node
    vals_formula = set(label.get_vals())
    next_frontier = []
      
    for candidate_term in candidate_labels:

        if candidate_term.val in vals_formula:
            continue
        for next_op in ["OR", "AND", "NOT"]:
            # A zero term cannot improve AND or OR label 
            if next_op != "NOT" and candidate_term.val not in non_zero_labels:
                continue
            # Build the candidate formula based on the next operation
            if next_op == "OR":
                candidate_formula = F.Or(label, candidate_term)
                next_frontier.append(candidate_formula)
            elif next_op == "AND" or next_op == "NOT":
                next_frontier.extend(chain_compounds(label, candidate_term, next_op))
            else:
                raise ValueError(f"Unknown operation {next_op}")


    return next_frontier

def get_last_compound(formula):
    if isinstance(formula, F.Leaf) or isinstance(formula, F.Not) or isinstance(formula, F.And):
        return formula
    else:
        return get_last_compound(formula.right)
   

def chain_compounds(formula, to_add, op):
    if op == "NOT":
        to_add = F.Not(to_add)
    if len(formula) == 1:
        return [F.And(formula, to_add)]
    
    last_compound = get_last_compound(formula)
    if len(last_compound) == len(formula):
        # We can directly attach it because we have a single compound
        return [F.And(formula, to_add)]
    else:
        assert isinstance(formula, F.Or) # All the other cases should be captured byh previous code
        last_compound = get_last_compound(formula)
        candidate_extending_compound = F.And(last_compound, to_add)
        formula_1 = F.Or(formula.left, candidate_extending_compound)
        formula_2 = F.And(formula, to_add)
        return [formula_1, formula_2]
    
def counter_beam_search(
    search_space,
    *,
    masks,
    beam_masks,
    bitmaps,
    beam_limit,
    previous_beam=None,
    use_logic_equivalence=True,
    diff_threshold=0.1,
    concept_diff_dict=None
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
    current_beam = Q.PriorityQueue(beam_limit)
    minimum = 0
    visited_indices = 0
    best_formula = None
    counter_bitmaps = ~bitmaps
    # Init beam with previous best
    for node, v in previous_beam.items():
        if not current_beam.full():
            current_beam.put(node)
            minimum = current_beam.queue[0][0]
        elif v > minimum:
            current_beam.get()
            current_beam.put(node)
            minimum = current_beam.queue[0][0]
    if current_beam.empty():
        minimum = 0
    else:
        minimum = current_beam.queue[0][0]
    
    while len(search_space) > 0:
        node = heapq.heappop(search_space)
        e_iou = -node[0]
        if current_beam.full() and e_iou < minimum:
            break
        candidate_formula = node[2]

        # skip equivalent formulas of the current beam
        if use_logic_equivalence and best_formula and hash(candidate_formula) == hash(best_formula):
            continue

        masks_formula = mask_utils.get_formula_mask(
            candidate_formula, masks, beam_masks
        ).to(bitmaps.device)

        iou = metrics.iou(
            masks_formula, bitmaps
        )
        adv_iou = metrics.iou(
            masks_formula, counter_bitmaps
        )
        diff = metrics.diff_ratio_iou_adv_iou(iou, adv_iou)
        # Impose constraint
        admissible = True
        if len(candidate_formula) > 1:
            if diff < diff_threshold:
                admissible = False
            else:
                last_op = candidate_formula.op
                if last_op == "OR": 
                    diff_concept = concept_diff_dict.get(candidate_formula.right, 0)
                    if diff_concept < diff_threshold:
                        print(f"Admissibility constraint not satisfied for formula {candidate_formula} because of the last added concept {candidate_formula.right} with IoU {iou} and diff {diff_concept}.")      
                        admissible = False

        visited_indices += 1
        if admissible:
            node = (iou.item(), node[1], node[2], node[3], diff)
            if not current_beam.full():
                current_beam.put(node)
                minimum = current_beam.queue[0][0]
            elif iou > minimum:
                current_beam.get()
                current_beam.put(node)
                minimum = current_beam.queue[0][0]
    return list(current_beam.queue), visited_indices