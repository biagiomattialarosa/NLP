

while top_equivalent:
    top_equivalent = False
    top_discarded_node = None
    # Select the top discarded node that is not functionally equivalent to the current beam nodes

    while not discarded_nodes.empty() and top_discarded_node is None:
        top_discarded_node = discarded_nodes.get()
        mask_top_discarded = mask_utils.get_formula_mask(
            top_discarded_node[2], masks, beam_masks
        ).to(bitmaps.device)

        if fail_check_beam_functional_equivalence(current_beam_masks, mask_top_discarded)[0]:
            top_discarded_node = None
    top_discarded_node = discarded_nodes.get() if not discarded_nodes.empty() else None

    # Check that it is not functionally equivalent to other discarded notes
    if top_discarded_node is not None:
        top_iou = top_discarded_node[0]
        next_node = discarded_nodes.get()
        next_iou = next_node[0] if next_node is not None else None
        different = []
        while top_iou == next_iou and not discarded_nodes.empty(): # Nodes with different iou cannot be functionally equivalent
            mask_next_node = mask_utils.get_formula_mask(
                next_node[2], masks, beam_masks
            ).to(bitmaps.device)
            if fail_check_functional_equivalence(mask_next_node, mask_top_discarded):
                top_equivalent = True
            else:
                different.append(next_node) # Useful for later
            next_node = discarded_nodes.get()
            next_iou = next_node[0] if next_node is not None else None
        # Restore discarded nods different from the equivalent ones
        for different_node in different:
            discarded_nodes.put(different_node)
        different = []

# Add the top discarded node in place to the removed equivalent
if top_discarded_node is not None:
    current_beam.put(top_discarded_node)
    current_beam_masks[top_discarded_node[2]] = mask_top_discarded
    minimum = current_beam.queue[0][0]