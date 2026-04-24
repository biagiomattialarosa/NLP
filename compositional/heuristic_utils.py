import torch

TOP_INDEX_SAMPLE = 0
BOTTOM_INDEX_SAMPLE = 1
TOP_INDEX_SUM = 2
BOTTOM_INDEX_SUM = 3

INDEX_INDIVIDUAL = 0
INDEX_OR = 1
INDEX_AND = 2
INDEX_NOT = 3

INDEX_NODE_IOU_ESTI = 0
INDEX_NODE_NEXT_OP = 1
INDEX_NODE_LABEL = 2
INDEX_NODE_OPS = 3

INDEX_TUPLE_MAX = 0
INDEX_TUPLE_MIN = 1
INDEX_TUPLE_SAMPLE = 0
INDEX_TUPLE_SUM = 1

QUANTITIES = ['common_intersection', 'unique_intersection', 'common_extras', 'unique_extras', 'common_uncovered', 'unique_uncovered']

def compute_quantities_vector(concept_mask, bitmaps, common_elements, unique_elements, neuron_common, neuron_unique):
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
    num_elem = torch.numel(torch.ones_like(bitmaps))
    # Choose the dtype based on the number of elements
    if num_elem < 2**16:
        dtype = torch.int16
    elif num_elem < 2**32:
        dtype = torch.int32
    else:
        dtype = torch.int64
    if concept_mask.device != bitmaps.device:
        concept_mask = concept_mask.to(bitmaps.device)
    intersection = (concept_mask & bitmaps)
    if len(intersection.shape) == 2:
        unique_intersection = (intersection & unique_elements).sum(
            dim=1, dtype=dtype).to('cpu').numpy()
        common_intersection = (intersection & common_elements).sum(
            dim=1, dtype=dtype).to('cpu').numpy()
        extras = (concept_mask & (~bitmaps))
        common_extras = (extras & common_elements).sum(
            dim=1, dtype=dtype).to('cpu').numpy()
        unique_extras = (extras & unique_elements).sum(
            dim=1, dtype=dtype).to('cpu').numpy()
    elif len(intersection.shape) == 1:
        unique_intersection = ((intersection & unique_elements).to(
            dtype)).to('cpu').numpy()
        common_intersection = ((intersection & common_elements).to(
            dtype)).to('cpu').numpy()
        extras = (concept_mask & (~bitmaps))
        common_extras = ((extras & common_elements).to(
            dtype)).to('cpu').numpy()
        unique_extras = ((extras & unique_elements).to(
            dtype)).to('cpu').numpy()
    else:
        raise ValueError(
            f"Unexpected shape for intersection: {intersection.shape}")
    common_uncovered = neuron_common - common_intersection
    unique_uncovered = neuron_unique - unique_intersection

    # Old uncovered calculation (not used anymore)
    # uncovered = ((~concept_mask) & bitmaps_coverable).sum(
    #     dim=1, dtype=torch.int32).to('cpu').numpy()
    return (common_intersection, unique_intersection, common_extras, unique_extras, common_uncovered, unique_uncovered)

def get_concept_info(concept_quantities):
        common_intersection, unique_intersection, common_extras, unique_extras, common_uncovered, unique_uncovered = concept_quantities
        common_intersection_sum = common_intersection.sum()
        unique_intersection_sum = unique_intersection.sum()
        common_extras_sum = common_extras.sum()
        unique_extras_sum = unique_extras.sum()
        common_uncovered_sum = common_uncovered.sum()
        unique_uncovered_sum = unique_uncovered.sum()
        # Min and Max are equal since they concept quantities are assumed to be exact
        tuple_common_intersection = ((common_intersection, common_intersection_sum), (common_intersection, common_intersection_sum))
        tuple_unique_intersection = ((unique_intersection, unique_intersection_sum), (unique_intersection, unique_intersection_sum))
        tuple_common_extras = ((common_extras, common_extras_sum), (common_extras, common_extras_sum))
        tuple_unique_extras = ((unique_extras, unique_extras_sum), (unique_extras, unique_extras_sum))
        tuple_common_uncovered = ((common_uncovered, common_uncovered_sum), (common_uncovered, common_uncovered_sum))
        tuple_unique_uncovered = ((unique_uncovered, unique_uncovered_sum), (unique_uncovered, unique_uncovered_sum))
        info = (tuple_common_intersection, tuple_unique_intersection, tuple_common_extras, tuple_unique_extras, tuple_common_uncovered, tuple_unique_uncovered)
        return info

def get_quantity(*, concepts_quantities, quantity_name, quantity_type, quantity_scope, label=None, label_mapping=None):
    """
    Returns the maximum and minimum values of a quantity.
    
    Args:
        quantity (tuple): A tuple containing the quantity.
        
    Returns:
        tuple: Maximum and minimum values of the quantity.
    """
    if quantity_type not in ['max', 'min']:
        raise ValueError(f"Unknown quantity type: {quantity_type}")
    if quantity_scope not in ['sample', 'sum']:
        raise ValueError(f"Unknown quantity scope: {quantity_scope}")
    if quantity_name not in QUANTITIES:
        raise ValueError(f"Unknown quantity name: {quantity_name}")

    index_quantity = QUANTITIES.index(quantity_name)
    index_type = INDEX_TUPLE_MAX if quantity_type == 'max' else INDEX_TUPLE_MIN
    index_scope = INDEX_TUPLE_SAMPLE if quantity_scope == 'sample' else INDEX_TUPLE_SUM
    if label_mapping is None:
        assert label is None
        quantity = concepts_quantities[index_quantity][index_type][index_scope]
    else:
        index_label = label_mapping[label]
        quantity = concepts_quantities[index_label][index_quantity][index_type][index_scope]

    return quantity