from collections import Counter
import heapq
import queue as Q

from . import formula as F
from . import utils
from compositional import mask_utils, metrics
from . import search_utils, compound_beam_search, counter_beam_search


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


def perform_exhaustive_heuristic_search(
    heuristic_info,
    disjoint_info,
    masks,
    bitmaps,
    num_hits,
    *,
    beam_size=5,
    max_size_mask,
    length=3,
    labels=None,
    beam_variant=None,
    constraints=None,
    counter_variant=False,
    diff_threshold=0.1,
    block_type_3=True,
    first_beam_size=None,
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
    label_mapping = {F.Leaf(c): c for c in range(len(masks))}

    if beam_variant == 'old':
        best_label, best_iou, tot_visited, tot_expanded, tot_estimated = old_explore_beam_frontier(
                    heuristic_info,
                    disjoint_info,
                    0.0,
                    label_mapping,
                    None,
                    masks,
                    bitmaps,
                    num_hits,
                    max_size_mask=max_size_mask,
                    beam_size=beam_size,
                    length=length,
                    labels=labels,
                    first_beam_size=first_beam_size
        )
    elif beam_variant == 'new':
        best_label, best_iou, tot_visited, tot_expanded, tot_estimated = counter_beam_search.explore_counter_beam_frontier(
                        heuristic_info,
                        disjoint_info,
                        0.0,
                        label_mapping,
                        None,
                        masks,
                        bitmaps,
                        num_hits,
                        max_size_mask=max_size_mask,
                        beam_size=beam_size,
                        length=length,
                        labels=labels,
                        constraints=constraints,
                        counter_variant=counter_variant,
                        diff_threshold=diff_threshold,
                        first_beam_size=first_beam_size
        )
    elif beam_variant == 'compound':
        best_label, best_iou, tot_visited, tot_expanded, tot_estimated = compound_beam_search.explore_beam_frontier_compound(
                        heuristic_info,
                        disjoint_info,
                        0.0,
                        label_mapping,
                        None,
                        masks,
                        bitmaps,
                        num_hits,
                        max_size_mask=max_size_mask,
                        beam_size=beam_size,
                        length=length,
                        labels=labels,
                        constraints=constraints,
                        block_type_3=block_type_3,
                        first_beam_size=first_beam_size
        )
    elif beam_variant == 'baseline':
        best_label, best_iou, tot_visited, tot_expanded, tot_estimated = baseline_explore_beam_frontier(
                        heuristic_info,
                        disjoint_info,
                        0.0,
                        label_mapping,
                        None,
                        masks,
                        bitmaps,
                        num_hits,
                        max_size_mask=max_size_mask,
                        beam_size=beam_size,
                        length=length,
                        labels=labels,
                        constraints=constraints,
                        first_beam_size=first_beam_size
        )
    else:
        raise ValueError(f"Unknown beam variant {beam_variant}")
    return best_label, best_iou, tot_visited, tot_expanded, tot_estimated


def beam_search(
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
    current_beam = Q.PriorityQueue(beam_limit)
    minimum = 0
    visited_indices = 0
    best_formula = None
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
        visited_indices += 1
        node = (iou, node[1], node[2], node[3])

        if not current_beam.full():
            current_beam.put(node)
            minimum = current_beam.queue[0][0]
        elif iou > minimum:
            current_beam.get()
            current_beam.put(node)
            minimum = current_beam.queue[0][0]
    return list(current_beam.queue), visited_indices


def baseline_explore_beam_frontier(
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
    first_beam_size=None,
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
    
    iou_atoms = {k: search_utils.analyze_final_node(k, masks, bitmaps) for k in candidate_labels}

    iou_atoms = Counter(iou_atoms)
    non_iou_labels =  [lab.val for lab, iou in iou_atoms.items() if iou > 0]

    first_beam_num = min (len(iou_atoms), beam_size if first_beam_size is None else first_beam_size)
    beam_atoms = {
        (iou, 'INDIVIDUAL', lab, None): iou
        for lab, iou in iou_atoms.most_common(first_beam_num)
        if iou > 0
    }
    # print("first beam:")
    # print(analyze_beam(beam_atoms.keys(), masks, bitmaps))
    leaf_mapping = {F.Leaf(c): c for c in range(len(labels))}

    # for (_, _, value, _), iou in beam_atoms.items():
    #     label_mask = utils.sparse_to_torch(mask_utils.get_formula_mask(
    #         value, masks
    #     ))
    #     print(f"Concept {value} - {labels[value.val]} : Mask Sum {label_mask.sum().item()}, Bitmaps: {bitmaps.sum().item()} IoU {iou}")
    # exit()
 
    minimum_threshold = min(beam_atoms.values())
    # Beam Search
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
                next_frontier = beam_expand_node(node, candidate_labels=candidate_labels, non_zero_labels=non_iou_labels, max_length=length, constraints=constraints)
                expanded_nodes += 1
                beam_candidates.extend(next_frontier)
        # Remove duplicates like in MMESH
        beam_candidates = list(set(beam_candidates))
        
        # Compute the estimation for the next frontier
        estimated_nodes += len(beam_candidates)
        beam_estimations, _ = search_utils.update_frontier(
            past_frontier=None, new_nodes=beam_candidates,
            label_mapping=label_mapping, heuristic='sample',
            heuristic_info=heuristic_info, max_improvement=max_improvement,
            num_hits=num_hits, max_size_mask=max_size_mask,
            length=length, global_min_threshold=minimum_threshold, disjoint_info=disjoint_info
        )
        next_beam, visited_nodes = beam_search(
            beam_estimations,
            masks=masks,
            previous_beam=beam,
            beam_masks=beam_masks,
            bitmaps=bitmaps,
            beam_limit=beam_size
        )
        
        total_visited = total_visited + visited_nodes
        for node in next_beam:
            if len(node[2]) == index_loop + 1:
                beam.update({(node[0], 'INDIVIDUAL', node[2], str(node[2])): node[0]})
        beam = dict(Counter(beam).most_common(beam_size))
        beam_masks, (heuristic_info, label_mapping) = search_utils.get_beam_info(beam.keys(), masks, beam_masks, heuristic_info, leaf_mapping, bitmaps, length)
        # print("Next beam")
        # for (_, _, value, _), iou in beam.items():
        #     label_mask = utils.sparse_to_torch(mask_utils.get_formula_mask(
        #         value, masks
        #     ))
        #     print(f"Concept {value}: {F.get_formula_str(value, labels)}-  Mask Sum {label_mask.sum().item()}, Bitmaps: {bitmaps.sum().item()} IoU {iou}")
    best_node, best_iou = Counter(beam).most_common(1)[0]
    best_label = best_node[2]
    return best_label, best_iou, total_visited, expanded_nodes, estimated_nodes


def old_explore_beam_frontier(
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
    candidate_labels = [F.Leaf(c) for c in range(len(masks))]
    
    iou_atoms = {k: search_utils.analyze_final_node(k, masks, bitmaps) for k in candidate_labels}

    iou_atoms = Counter(iou_atoms)
    first_beam_num = min (len(iou_atoms), beam_size)
    # added str(lab) instead of None to break equivalence and replicate bug
    beam_atoms = {
        (iou, 'INDIVIDUAL', lab, str(lab)): iou
        for lab, iou in iou_atoms.most_common(first_beam_num)
        if iou > 0
    }
    # for (_, _, value, _), iou in beam_atoms.items():
    #     label_mask = utils.sparse_to_torch(mask_utils.get_formula_mask(
    #         value, masks
    #     ))
    #     print(f"Concept {value} - {labels[value.val]} : Mask Sum {label_mask.sum().item()}, Bitmaps: {bitmaps.sum().item()} IoU {iou}")
    non_iou_labels =  [lab for lab, iou in iou_atoms.items() if iou > 0]
    non_iou_labels_vals = [lab.val for lab, iou in iou_atoms.items() if iou > 0]
    leaf_mapping = {c: c.val for c in non_iou_labels}

    label_mapping.update(leaf_mapping)
 
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
                next_frontier = beam_expand_node(node, candidate_labels=non_iou_labels, non_zero_labels=non_iou_labels_vals, max_length=length)
                expanded_nodes += 1
                beam_candidates.extend(next_frontier)
        estimated_nodes += len(beam_candidates)
        # Compute the estimation for the next frontier
        beam_estimations, _ = search_utils.update_frontier(
            past_frontier=None, new_nodes=beam_candidates,
            label_mapping=label_mapping, heuristic='sample',
            heuristic_info=heuristic_info, max_improvement=max_improvement,
            num_hits=num_hits, max_size_mask=max_size_mask,
            length=length, global_min_threshold=minimum_threshold, disjoint_info=disjoint_info
        )
        next_beam, visited_nodes = beam_search(
            beam_estimations,
            masks=masks,
            previous_beam=beam,
            beam_masks=beam_masks,
            bitmaps=bitmaps,
            beam_limit=beam_size,
            use_logic_equivalence=False
        )
        total_visited = total_visited + visited_nodes
        for node in next_beam:
            if len(node[2]) == index_loop + 1:
                beam.update({(node[0], 'INDIVIDUAL', node[2], str(node[2])): node[0]})
        beam = dict(Counter(beam).most_common(beam_size))
        beam_masks, (heuristic_info, label_mapping) = search_utils.get_beam_info(beam.keys(), masks, beam_masks, heuristic_info, leaf_mapping, bitmaps, length)
    best_node, best_iou = Counter(beam).most_common(1)[0]
    best_label = best_node[2]
    return best_label, best_iou, total_visited, expanded_nodes, estimated_nodes
