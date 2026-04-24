import numpy as np

from . import formula as F
from . import heuristic_utils
from .heuristic_utils import TOP_INDEX_SUM, BOTTOM_INDEX_SUM


def estimate_disjoint_label_info(label, left_quantities, right_quantities, neuron_quantities, max_size_mask, disjoint_masks):

    """
    Estimate the label information for a given label based on its left and right quantities.

    Args:
        label (F.Formula): The label for which to estimate the label information.
        left_quantities (tuple): Quantities from the left child of the label.
        right_quantities (tuple): Quantities from the right child of the label.

    Returns:
        tuple: Estimated quantities for the label.
    """

    # Left quantities
    max_left_unique_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities, quantity_name='unique_intersection', quantity_type='max', quantity_scope='sum')
    min_left_unique_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities, quantity_name='unique_intersection', quantity_type='min', quantity_scope='sum')
    max_left_common_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities, quantity_name='common_intersection', quantity_type='max', quantity_scope='sum')
    min_left_common_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities, quantity_name='common_intersection', quantity_type='min', quantity_scope='sum')
    max_left_common_extras_sum = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities, quantity_name='common_extras', quantity_type='max', quantity_scope='sum')
    min_left_common_extras_sum = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities, quantity_name='common_extras', quantity_type='min', quantity_scope='sum')
    max_left_unique_extras_sum = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities, quantity_name='unique_extras', quantity_type='max', quantity_scope='sum')
    min_left_unique_extras_sum = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities, quantity_name='unique_extras', quantity_type='min', quantity_scope='sum')
    max_left_common_uncovered_sum = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities, quantity_name='common_uncovered', quantity_type='max', quantity_scope='sum')
    min_left_common_uncovered_sum = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities, quantity_name='common_uncovered', quantity_type='min', quantity_scope='sum')
    max_left_unique_uncovered_sum = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities, quantity_name='unique_uncovered', quantity_type='max', quantity_scope='sum')
    min_left_unique_uncovered_sum = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities, quantity_name='unique_uncovered', quantity_type='min', quantity_scope='sum')

    # Right quantities
    max_right_unique_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities, quantity_name='unique_intersection', quantity_type='max', quantity_scope='sum')
    min_right_unique_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities, quantity_name='unique_intersection', quantity_type='min', quantity_scope='sum')
    max_right_common_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities, quantity_name='common_intersection', quantity_type='max', quantity_scope='sum')
    min_right_common_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities, quantity_name='common_intersection', quantity_type='min', quantity_scope='sum') 
    max_right_common_extras_sum = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities, quantity_name='common_extras', quantity_type='max', quantity_scope='sum')
    min_right_common_extras_sum = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities, quantity_name='common_extras', quantity_type='min', quantity_scope='sum')
    max_right_unique_extras_sum = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities, quantity_name='unique_extras', quantity_type='max', quantity_scope='sum')
    min_right_unique_extras_sum = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities, quantity_name='unique_extras', quantity_type='min', quantity_scope='sum')
    max_right_common_uncovered_sum = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities, quantity_name='common_uncovered', quantity_type='max', quantity_scope='sum') 
    min_right_common_uncovered_sum = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities, quantity_name='common_uncovered', quantity_type='min', quantity_scope='sum')
    max_right_unique_uncovered_sum = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities, quantity_name='unique_uncovered', quantity_type='max', quantity_scope='sum')
    min_right_unique_uncovered_sum = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities, quantity_name='unique_uncovered', quantity_type='min', quantity_scope='sum')

    if isinstance(label, F.Or):
        max_unique_intersection_sum = max_left_unique_intersection_sum + max_right_unique_intersection_sum
        min_unique_intersection_sum = min_left_unique_intersection_sum + min_right_unique_intersection_sum
        max_common_intersection_sum = max_left_common_intersection_sum + max_right_common_intersection_sum
        min_common_intersection_sum = min_left_common_intersection_sum + min_right_common_intersection_sum
        if ((max_unique_intersection_sum <= max_left_unique_intersection_sum) and 
                    (max_common_intersection_sum <= max_left_common_intersection_sum) and 
                    (min_unique_intersection_sum <= min_left_unique_intersection_sum) and 
                    (min_common_intersection_sum <= min_left_common_intersection_sum)) or \
                   ((max_unique_intersection_sum <= max_right_unique_intersection_sum) and 
                    (max_common_intersection_sum <= max_right_common_intersection_sum) and 
                    (min_unique_intersection_sum <= min_right_unique_intersection_sum) and 
                    (min_common_intersection_sum <= min_right_common_intersection_sum)):
            # If one of the two side does not change to the intersection, we can discard this formula
            return None

        min_common_uncovered_sum = max(min(min_left_common_uncovered_sum - max_right_common_intersection_sum, min_right_common_uncovered_sum - max_left_common_intersection_sum), 0)
        max_common_uncovered_sum = max(min(max_left_common_uncovered_sum - min_right_common_intersection_sum, max_right_common_uncovered_sum - min_left_common_intersection_sum), 0)
        min_unique_uncovered_sum = max(min(min_left_unique_uncovered_sum - max_right_unique_intersection_sum, min_right_unique_uncovered_sum - max_left_unique_intersection_sum), 0)
        max_unique_uncovered_sum = max(min(max_left_unique_uncovered_sum - min_right_unique_intersection_sum, max_right_unique_uncovered_sum - min_left_unique_intersection_sum), 0)

        min_unique_extras_sum = min_left_unique_extras_sum + min_right_unique_extras_sum
        max_unique_extras_sum = max_left_unique_extras_sum + max_right_unique_extras_sum


        min_common_extras_sum = min_left_common_extras_sum + min_right_common_extras_sum
        max_common_extras_sum = max_left_common_extras_sum + max_right_common_extras_sum
        


        return ((None, max_common_intersection_sum), (None, min_common_intersection_sum)), ((None, max_unique_intersection_sum), (None, min_unique_intersection_sum)), \
        ((None, max_common_extras_sum), (None, min_common_extras_sum)), \
        ((None, max_unique_extras_sum), (None, min_unique_extras_sum)), \
        ((None, max_common_uncovered_sum), (None, min_common_uncovered_sum)), \
        ((None, max_unique_uncovered_sum), (None, min_unique_uncovered_sum))
    elif isinstance(label, F.And) and isinstance(label.right, F.Not):
        # Since they are disjoint, there is not a counter example of their presence together
        # Everything would end up the same of the left label (see commentetd part below)
        return None

    elif isinstance(label, F.And):
        # AND of disjoint labels is zero by definition
            return None
    else:
        raise ValueError(f"Unknown label type: {type(label)}")



def estimate_label_info(label, left_quantities, right_quantities, neuron_quantities, max_size_mask):

    """
    Estimate the label information for a given label based on its left and right quantities.

    Args:
        label (F.Formula): The label for which to estimate the label information.
        left_quantities (tuple): Quantities from the left child of the label.
        right_quantities (tuple): Quantities from the right child of the label.

    Returns:
        tuple: Estimated quantities for the label.
    """

    neuron_unique_tuple, neuron_common_tuple, neuron_coverable_mask_tuple, neuron_coverable_tuple, neuron_sum_tuple, common_space_extras_tuple, unique_space_extras_tuple = neuron_quantities
    neuron_coverable, neuron_coverable_sum = neuron_coverable_mask_tuple
    _ , neuron_hits = neuron_sum_tuple
    _, neuron_common_sum = neuron_common_tuple
    _, neuron_unique_sum = neuron_unique_tuple
    _, common_space_extras_sum = common_space_extras_tuple
    _, unique_space_extras_sum = unique_space_extras_tuple

    tot_size = max_size_mask*len(neuron_coverable)
    _, common_space_extras_sum = common_space_extras_tuple
    _, unique_space_extras_sum = unique_space_extras_tuple

    # Left quantities
    max_left_unique_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities, quantity_name='unique_intersection', quantity_type='max', quantity_scope='sum')
    min_left_unique_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities, quantity_name='unique_intersection', quantity_type='min', quantity_scope='sum')
    max_left_common_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities, quantity_name='common_intersection', quantity_type='max', quantity_scope='sum')
    min_left_common_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities, quantity_name='common_intersection', quantity_type='min', quantity_scope='sum')
    max_left_common_extras_sum = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities, quantity_name='common_extras', quantity_type='max', quantity_scope='sum')
    min_left_common_extras_sum = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities, quantity_name='common_extras', quantity_type='min', quantity_scope='sum')
    max_left_unique_extras_sum = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities, quantity_name='unique_extras', quantity_type='max', quantity_scope='sum')
    min_left_unique_extras_sum = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities, quantity_name='unique_extras', quantity_type='min', quantity_scope='sum')
    max_left_common_uncovered_sum = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities, quantity_name='common_uncovered', quantity_type='max', quantity_scope='sum')
    min_left_common_uncovered_sum = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities, quantity_name='common_uncovered', quantity_type='min', quantity_scope='sum')
    max_left_unique_uncovered_sum = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities, quantity_name='unique_uncovered', quantity_type='max', quantity_scope='sum')
    min_left_unique_uncovered_sum = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities, quantity_name='unique_uncovered', quantity_type='min', quantity_scope='sum')

    # Right quantities
    max_right_unique_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities, quantity_name='unique_intersection', quantity_type='max', quantity_scope='sum')
    min_right_unique_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities, quantity_name='unique_intersection', quantity_type='min', quantity_scope='sum')
    max_right_common_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities, quantity_name='common_intersection', quantity_type='max', quantity_scope='sum')
    min_right_common_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities, quantity_name='common_intersection', quantity_type='min', quantity_scope='sum') 
    max_right_common_extras_sum = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities, quantity_name='common_extras', quantity_type='max', quantity_scope='sum')
    min_right_common_extras_sum = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities, quantity_name='common_extras', quantity_type='min', quantity_scope='sum')
    max_right_unique_extras_sum = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities, quantity_name='unique_extras', quantity_type='max', quantity_scope='sum')
    min_right_unique_extras_sum = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities, quantity_name='unique_extras', quantity_type='min', quantity_scope='sum')
    max_right_common_uncovered_sum = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities, quantity_name='common_uncovered', quantity_type='max', quantity_scope='sum') 
    min_right_common_uncovered_sum = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities, quantity_name='common_uncovered', quantity_type='min', quantity_scope='sum')
    max_right_unique_uncovered_sum = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities, quantity_name='unique_uncovered', quantity_type='max', quantity_scope='sum')
    min_right_unique_uncovered_sum = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities, quantity_name='unique_uncovered', quantity_type='min', quantity_scope='sum')

    if isinstance(label, F.Or):

        max_unique_intersection_sum = max_left_unique_intersection_sum + max_right_unique_intersection_sum
        min_unique_intersection_sum = min_left_unique_intersection_sum + min_right_unique_intersection_sum

        min_unique_extras_sum = min_left_unique_extras_sum + min_right_unique_extras_sum
        max_unique_extras_sum = max_left_unique_extras_sum + max_right_unique_extras_sum
        
        min_common_intersection_sum = max(min_left_common_intersection_sum, min_right_common_intersection_sum)
        max_common_intersection_sum = min(max(
            max_left_common_intersection_sum + min(max_left_common_uncovered_sum, max_right_common_intersection_sum),
            max_right_common_intersection_sum + min(max_right_common_uncovered_sum, max_left_common_intersection_sum)),
            neuron_common_sum
        )  

        if ((max_unique_intersection_sum <= max_left_unique_intersection_sum and \
            min_unique_intersection_sum <= min_left_unique_intersection_sum and \
            max_common_intersection_sum <= max_left_common_intersection_sum and \
            min_common_intersection_sum <= min_left_common_intersection_sum) or \
            (max_unique_intersection_sum <= max_right_unique_intersection_sum and \
            min_unique_intersection_sum <= min_right_unique_intersection_sum and \
            max_common_intersection_sum <= max_right_common_intersection_sum and \
            min_common_intersection_sum <= min_right_common_intersection_sum)):
            # If one of the two side does not change to the intersection, we can discard this formula
            return None

        min_common_uncovered_sum = max(min(min_left_common_uncovered_sum - max_right_common_intersection_sum, min_right_common_uncovered_sum - max_left_common_intersection_sum), 0)
        max_common_uncovered_sum = min(max_left_common_uncovered_sum, max_right_common_uncovered_sum)
        min_unique_uncovered_sum = max(min(min_left_unique_uncovered_sum - max_right_unique_intersection_sum, min_right_unique_uncovered_sum - max_left_unique_intersection_sum), 0)
        max_unique_uncovered_sum = max(min(max_left_unique_uncovered_sum - min_right_unique_intersection_sum, max_right_unique_uncovered_sum - min_left_unique_intersection_sum), 0)

        min_common_extras_sum = max(min_left_common_extras_sum, min_right_common_extras_sum)
        max_common_extras_sum = min(common_space_extras_sum,
                                    max_left_common_extras_sum + max_right_common_extras_sum) # min(max_size_mask - N^u - N^c - E_min^u(L) - E_min^u(c), E_max^c(L) + E_max^c(c))
       

    elif isinstance(label, F.And) and isinstance(label.right, F.Not):

        max_unique_intersection_sum = max_left_unique_intersection_sum
        min_unique_intersection_sum = min_left_unique_intersection_sum

        #min_common_intersection_sum = max(min_left_common_intersection_sum + min_right_common_uncovered_sum - neuron_common_sum, 0)
        max_common_intersection_sum = min(max_right_common_uncovered_sum, max_left_common_intersection_sum) # min(U_max^c, I_max^c(L))
        min_common_intersection_sum = max(min_left_common_intersection_sum - min(max_left_common_intersection_sum, max_right_common_intersection_sum), 0)
        #max_common_intersection_sum = max(max_left_common_intersection_sum - max(0, min_left_common_intersection_sum + min_right_common_intersection_sum - neuron_common_sum), 0)

        #min_common_uncovered_sum = max(min_left_common_uncovered_sum, min_right_common_intersection_sum) # max(U_min^L, I_min^c(c) + I_min^u(c))
        #max_common_uncovered_sum = neuron_common_sum - max(0, min_left_common_intersection_sum + min_right_common_uncovered_sum - neuron_common_sum) # N^u + N^c - max(0, I_min^c(L) + I_min^c(c) - N^u - N^c)
        min_common_uncovered_sum = min(min_left_common_uncovered_sum + max(min_left_common_intersection_sum + min_right_common_intersection_sum - neuron_common_sum, 0), neuron_common_sum)
        max_common_uncovered_sum = min(max_left_common_uncovered_sum + min(max_left_common_intersection_sum, max_right_common_intersection_sum), neuron_common_sum)

        min_unique_uncovered_sum = min_left_unique_uncovered_sum
        max_unique_uncovered_sum = max_left_unique_uncovered_sum

        min_unique_extras_sum = min_left_unique_extras_sum
        max_unique_extras_sum = max_left_unique_extras_sum

        # min_common_extras_sum = max(min_left_common_extras_sum + (common_space_extras_sum - max_right_common_extras_sum) - common_space_extras_sum, 0) 
        # max_common_extras_sum = max_left_common_extras_sum
        min_common_extras_sum = min_left_common_extras_sum - min(min_left_common_extras_sum, max_right_common_extras_sum)
        max_common_extras_sum = max_left_common_extras_sum - max(0, min_left_common_extras_sum + min_right_common_extras_sum - common_space_extras_sum) # E_max^c(L) - max(0, E_min^c(L) + E_min^c(c) - E_max^c(c))

        # if (min_common_extras_sum >= min_left_common_extras_sum and \
        #     max_common_intersection_sum <= max_left_common_intersection_sum and \
        #     min_common_intersection_sum <= min_left_common_intersection_sum):
        #     # If one of the two side does not contribute to the intersection, we can discard this formula
        #     return None
        if (max_left_common_extras_sum == 0 and \
            min_left_common_extras_sum == 0):
                return None
        
    elif isinstance(label, F.And):
        # The and always zeroes the unique elements
        min_unique_extras_sum = 0
        max_unique_extras_sum = 0

        min_common_extras_sum = max(min_left_common_extras_sum + min_right_common_extras_sum - common_space_extras_sum, 0)
    
        max_common_extras_sum = min(max_left_common_extras_sum, max_right_common_extras_sum)
        
        if (max_left_unique_extras_sum == 0 and \
            min_left_unique_extras_sum == 0 and \
            max_left_common_extras_sum == 0 and \
            min_left_common_extras_sum == 0):
                return None
        # max_common_extras_sum >= max_left_common_extras_sum and \
        # min_common_extras_sum >= min_left_common_extras_sum) or \
        # (max_right_unique_extras_sum == 0 and \
        # min_right_unique_extras_sum == 0 and \
        # max_common_extras_sum >= max_right_common_extras_sum and \
        # min_common_extras_sum >= min_right_common_extras_sum):
        # # if (max_left_unique_extras_sum == 0 and \
        # #     min_left_unique_extras_sum == 0 and \
        # # max_common_extras_sum >= max_left_common_extras_sum and \
        # # min_common_extras_sum >= min_left_common_extras_sum) or \
        # # (max_right_unique_extras_sum == 0 and \
        # # min_right_unique_extras_sum == 0 and \
        # # max_common_extras_sum >= max_right_common_extras_sum and \
        # # min_common_extras_sum >= min_right_common_extras_sum):
        #     print("AND ZERO UNIQUE", label)
        #     print((max_left_unique_extras_sum == 0 and \
        #     min_left_unique_extras_sum == 0 and \
        # max_common_extras_sum >= max_left_common_extras_sum and \
        # min_common_extras_sum >= min_left_common_extras_sum))
        #     print(max_common_extras_sum, max_left_common_extras_sum)
        #     print(min_common_extras_sum, min_left_common_extras_sum) 
        #     print(min_left_common_extras_sum, min_right_common_extras_sum, common_space_extras_sum)

        #     # If one of the two side does not contribute to the intersection, we can discard this formula
        #     return None

        # if (max_left_unique_extras_sum == 0 and \
        #     min_left_unique_extras_sum == 0 and \
        # max_common_extras_sum >= max_left_common_extras_sum and \
        # min_common_extras_sum >= min_left_common_extras_sum) or \
        # (max_right_unique_extras_sum == 0 and \
        # min_right_unique_extras_sum == 0 and \
        # max_common_extras_sum >= max_right_common_extras_sum and \
        # min_common_extras_sum >= min_right_common_extras_sum):
        # # if (max_left_unique_extras_sum == 0 and \
        # #     min_left_unique_extras_sum == 0 and \
        # # max_common_extras_sum >= max_left_common_extras_sum and \
        # # min_common_extras_sum >= min_left_common_extras_sum) or \
        # # (max_right_unique_extras_sum == 0 and \
        # # min_right_unique_extras_sum == 0 and \
        # # max_common_extras_sum >= max_right_common_extras_sum and \
        # # min_common_extras_sum >= min_right_common_extras_sum):
        #     print("AND ZERO UNIQUE", label)
        #     print((max_left_unique_extras_sum == 0 and \
        #     min_left_unique_extras_sum == 0 and \
        # max_common_extras_sum >= max_left_common_extras_sum and \
        # min_common_extras_sum >= min_left_common_extras_sum))
        #     print(max_common_extras_sum, max_left_common_extras_sum)
        #     print(min_common_extras_sum, min_left_common_extras_sum) 
        #     print(min_left_common_extras_sum, min_right_common_extras_sum, common_space_extras_sum)

        #     # If one of the two side does not contribute to the intersection, we can discard this formula
        #     return None
        
        max_unique_intersection_sum = 0
        min_unique_intersection_sum = 0
        
        min_common_intersection_sum = max(min_left_common_intersection_sum + min_right_common_intersection_sum - neuron_common_sum, 0)
        
        # This will select always the smaller concept
        max_common_intersection_sum = min(max_left_common_intersection_sum, max_right_common_intersection_sum)
        
        min_common_uncovered_sum = max(min_left_common_uncovered_sum, min_right_common_uncovered_sum) # max(U_min^L, U_min^c)
        # max_common_uncovered_sum = max(
        #     neuron_common_sum - max(
        #         0, min_left_common_intersection_sum + min_right_common_intersection_sum - neuron_common_sum), 0) # N^u + N^c - max(0, I_min^c(L) + I_min^c(c) - N^u - N^c)
        max_common_uncovered_sum = min(max_left_common_uncovered_sum + max_right_common_uncovered_sum - max(0, min_left_common_uncovered_sum + min_right_common_uncovered_sum - neuron_common_sum), neuron_common_sum)
        # AND cannot preserve unique, so the unique uncovered always increases
        #min_unique_uncovered_sum = min(max_left_unique_uncovered_sum + max_right_unique_intersection_sum, max_right_unique_uncovered_sum + max_left_unique_intersection_sum, neuron_coverable_sum)
        min_unique_uncovered_sum = max(min_left_unique_uncovered_sum, min_right_unique_uncovered_sum) # max(U_min^L, U_min^c)
        max_unique_uncovered_sum = min(max_left_unique_uncovered_sum + max_right_unique_uncovered_sum, neuron_unique_sum) 
    return ((None, max_common_intersection_sum), (None, min_common_intersection_sum)), ((None, max_unique_intersection_sum), (None, min_unique_intersection_sum)), \
        ((None, max_common_extras_sum), (None, min_common_extras_sum)), \
        ((None, max_unique_extras_sum), (None, min_unique_extras_sum)), \
        ((None, max_common_uncovered_sum), (None, min_common_uncovered_sum)), \
        ((None, max_unique_uncovered_sum), (None, min_unique_uncovered_sum))



def individual_estimation(label, concept_quantities, neuron_quantities, max_improvement, *, num_hits, max_size_mask, max_length, minimum_threshold=0):
    neuron_unique_tuple, neuron_common_tuple, neuron_coverable_mask_tuple, neuron_coverable_tuple, neuron_sum_tuple, common_space_extras_tuple, unique_space_extras_tuple = neuron_quantities
    _, neuron_hits = neuron_sum_tuple
    neuron_coverable , neuron_coverable_sum = neuron_coverable_tuple
    tot_size = max_size_mask * len(neuron_coverable)
    _, common_space_extras_sum = common_space_extras_tuple
    _, unique_space_extras_sum = unique_space_extras_tuple
    
    # Unpack max and min quantities
    max_common_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='common_intersection', quantity_type='max', quantity_scope='sum')
    min_common_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='common_intersection', quantity_type='min', quantity_scope='sum')
    max_unique_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='unique_intersection', quantity_type='max', quantity_scope='sum')
    min_unique_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='unique_intersection', quantity_type='min', quantity_scope='sum')
    max_common_extras_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='common_extras', quantity_type='max', quantity_scope='sum')
    min_common_extras_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='common_extras', quantity_type='min', quantity_scope='sum')
    max_unique_extras_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='unique_extras', quantity_type='max', quantity_scope='sum')
    min_unique_extras_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='unique_extras', quantity_type='min', quantity_scope='sum')

    #max_label_intersection = max_common_intersection + max_unique_intersection
    max_label_common_intersection_sum =  min(max_common_intersection_sum + max_unique_intersection_sum, neuron_coverable_sum)
    if max_label_common_intersection_sum == 0:
        return 0.0, 0.0

    # Max IoU
    min_union = num_hits + min_unique_extras_sum + min_common_extras_sum
    max_iou = max_label_common_intersection_sum / min_union

    # Save computation
    if max_iou == 0 or max_iou < minimum_threshold:
        return 0.0, 0.0
    
    # Min IoU
    min_intersection = min(min_common_intersection_sum + min_unique_intersection_sum, neuron_coverable_sum)
    max_union = num_hits + min(max_unique_extras_sum + max_common_extras_sum, common_space_extras_sum + unique_space_extras_sum)
    min_iou = min_intersection / max_union
    return max_iou, min_iou


def or_chain_estimation(label, concept_quantities, neuron_quantities, max_improvement, *, num_hits, max_size_mask, max_length, minimum_threshold=0):
    # Aux variables
    label_len = len(label)
    k = max_length - label_len # Missing length   

    neuron_unique_tuple, neuron_common_tuple, neuron_coverable_mask_tuple, neuron_coverable_tuple, neuron_sum_tuple, common_space_extras_tuple, unique_space_extras_tuple = neuron_quantities
    _, neuron_common_sum = neuron_common_tuple
    _, neuron_unique_sum = neuron_unique_tuple
    _, neuron_hits = neuron_sum_tuple
    neuron_coverable, neuron_coverable_sum = neuron_coverable_tuple
    tot_size = max_size_mask * len(neuron_coverable)
    # Unpack improvement information
    improv_quantities, cumsum_quantities = max_improvement
    cumsum_unique_intersection, cumsum_unique_extras = cumsum_quantities
    improv_common_intersection, improv_unique_intersection, improv_common_extras, improv_unique_extras, _, _ = improv_quantities


    # # Unpack max and min quantities
    neuron_coverable_sum = neuron_coverable_tuple[1]

    max_common_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='common_intersection', quantity_type='max', quantity_scope='sum')
    min_common_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='common_intersection', quantity_type='min', quantity_scope='sum')
    max_unique_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='unique_intersection', quantity_type='max', quantity_scope='sum')
    min_unique_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='unique_intersection', quantity_type='min', quantity_scope='sum')
    max_common_extras_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='common_extras', quantity_type='max', quantity_scope='sum')
    min_common_extras_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='common_extras', quantity_type='min', quantity_scope='sum')
    max_unique_extras_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='unique_extras', quantity_type='max', quantity_scope='sum')
    min_unique_extras_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='unique_extras', quantity_type='min', quantity_scope='sum')

    neuron_coverable = neuron_coverable_tuple[0]
    # We discard the labels that cannot increase the IoU
    if max_common_intersection_sum + max_unique_intersection_sum == 0:
        return 0.0, 0.0
    
    #max_label_intersection = max_common_intersection + max_unique_intersection
    top_k_common_intersection_sum = improv_common_intersection[k][TOP_INDEX_SUM]
    top_k_unique_intersection_sum = improv_unique_intersection[k][TOP_INDEX_SUM]
    if isinstance(label, F.Leaf) or isinstance(label, F.Or):
        last_val = label.get_vals()[0]
        previous_index = last_val - 1
        if previous_index >= 0:
            top_k_unique_intersection_sum  = min(top_k_unique_intersection_sum, neuron_coverable_sum - cumsum_unique_intersection[previous_index])
    top_k_extras_sum = improv_common_extras[k][TOP_INDEX_SUM] + improv_unique_extras[k][TOP_INDEX_SUM]
    bottom_1_intersection_sum = improv_common_intersection[0][BOTTOM_INDEX_SUM] + improv_unique_intersection[0][BOTTOM_INDEX_SUM]
    bottom_1_extras_sum = improv_common_extras[0][BOTTOM_INDEX_SUM] + improv_unique_extras[0][BOTTOM_INDEX_SUM]

    tot_size = max_size_mask*len(neuron_coverable)

    # Max IoU
    min_union = min(num_hits + min_common_extras_sum + min_unique_extras_sum + max(0, bottom_1_extras_sum - max_common_extras_sum - max_unique_extras_sum), tot_size)
    max_intersection = min(
        min(max_common_intersection_sum + top_k_common_intersection_sum, neuron_common_sum)
        + min(max_unique_intersection_sum + top_k_unique_intersection_sum, neuron_unique_sum),
        neuron_coverable_sum)
    
    max_iou = max_intersection / min_union

    # Save computation
    if max_iou == 0 or max_iou < minimum_threshold:
        return 0.0, 0.0
    
    # Min IoU
    min_intersection = max(min(min_common_intersection_sum + min_unique_intersection_sum, neuron_coverable_sum), bottom_1_intersection_sum)
    max_union = min(num_hits + max_common_extras_sum + max_unique_extras_sum + top_k_extras_sum, tot_size)

    min_iou = min_intersection / max_union
    return max_iou, min_iou


def and_chain_estimation(label, concept_quantities, neuron_quantities, max_improvement, *, num_hits, max_size_mask, max_length, minimum_threshold=0):
    # Aux variables
    # Unpack improvement information
    improv_quantities, cum_sum_unique_extras = max_improvement
    improv_common_intersection, improv_unique_intersection, improv_common_extras, improv_unique_extras, _, _ = improv_quantities
    neuron_unique_tuple, neuron_common_tuple, neuron_coverable_mask_tuple, neuron_coverable_tuple, neuron_sum_tuple, common_space_extras_tuple, unique_space_extras_tuple = neuron_quantities
    _, neuron_hits = neuron_sum_tuple
    neuron_coverable , neuron_coverable_sum = neuron_coverable_tuple
    tot_size = max_size_mask * len(neuron_coverable)
    _, common_space_extras_sum = common_space_extras_tuple
    # # Unpack max and min quantities
    neuron_coverable = neuron_coverable_tuple[0]
    neuron_sum = neuron_sum_tuple[0]
    
    max_common_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='common_intersection', quantity_type='max', quantity_scope='sum')
    min_common_extras_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='common_extras', quantity_type='min', quantity_scope='sum')
    min_unique_extras_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='unique_extras', quantity_type='min', quantity_scope='sum')
    

    # We discard the labels that cannot increase the IoU
    if max_common_intersection_sum == 0:
        return 0.0, 0.0

    top_1_common_intersection_sum = improv_common_intersection[0][TOP_INDEX_SUM]
    bottom_1_common_extras_sum = improv_common_extras[0][BOTTOM_INDEX_SUM]
    bottom_1_common_intersection_sum = improv_common_intersection[0][BOTTOM_INDEX_SUM]
    tot_size = max_size_mask * len(neuron_coverable)

    # MaxIoU

    max_intersection = min(max_common_intersection_sum, top_1_common_intersection_sum)
    min_union = min(num_hits + max(0, min_common_extras_sum  + bottom_1_common_extras_sum - common_space_extras_sum), tot_size)


    max_iou = max_intersection / min_union

    # Save computation
    if max_iou == 0 or max_iou < minimum_threshold:
        return 0.0, 0.0
    if bottom_1_common_intersection_sum == 0:
        return max_iou, 0.0
    return max_iou, 0.0

def and_not_chain_estimation(label, concept_quantities, neuron_quantities, max_improvement, *, num_hits, max_size_mask, max_length, minimum_threshold=0):
    # Aux variables
    # Unpack improvement information
    improv_quantities, cum_sum_quantities = max_improvement
    improv_common_intersection, improv_unique_intersection, improv_common_extras, improv_unique_extras, improv_common_uncovered, improv_unique_uncovered = improv_quantities
    neuron_unique_tuple, neuron_common_tuple, neuron_coverable_mask_tuple, neuron_coverable_tuple, neuron_sum_tuple, common_space_extras_tuple, unique_space_extras_tuple = neuron_quantities
    _, neuron_hits = neuron_sum_tuple
    neuron_coverable , neuron_coverable_sum = neuron_coverable_tuple

    if isinstance(label, F.Leaf) or (isinstance(label, F.And) and isinstance(label.right, F.Not)):
        last_val = label.get_vals()[0]
        previous_index = last_val - 1
        cumsum_unique_intersection, cumsum_unique_extras = cum_sum_quantities
        if previous_index > 0:
            neuron_coverable_sum  = neuron_coverable_sum - cumsum_unique_intersection[previous_index]

    tot_size = max_size_mask * len(neuron_coverable)
    _, common_space_extras_sum = common_space_extras_tuple
    _, unique_space_extras_sum = unique_space_extras_tuple
    # # Unpack max and min quantities
    max_common_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='common_intersection', quantity_type='max', quantity_scope='sum')
    max_unique_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='unique_intersection', quantity_type='max', quantity_scope='sum')
    min_unique_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='unique_intersection', quantity_type='min', quantity_scope='sum')
    max_common_extras_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='common_extras', quantity_type='max', quantity_scope='sum')
    max_unique_extras_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='unique_extras', quantity_type='max', quantity_scope='sum')
    min_unique_extras_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='unique_extras', quantity_type='min', quantity_scope='sum')

    # We discard the labels that cannot increase the IoU
    if max_common_intersection_sum + max_unique_intersection_sum == 0:
        return 0.0, 0.0
    

    # Max IoU
    max_intersection = min(max_unique_intersection_sum + max_common_intersection_sum, neuron_coverable_sum)
    min_union = num_hits + min_unique_extras_sum
    max_iou = max_intersection / min_union

    # Save computation
    if max_iou == 0 or max_iou < minimum_threshold:
        return 0.0, 0.0
    
    # Min IoU
    min_intersection = min_unique_intersection_sum
    max_union = num_hits + min(max_unique_extras_sum + max_common_extras_sum, common_space_extras_sum + unique_space_extras_sum)
    min_iou = min_intersection / max_union


    return max_iou, min_iou

def comb_and_or_chain_estimation(label, concept_quantities, neuron_quantities, max_improvement, *, num_hits, max_size_mask, max_length, minimum_threshold=0):
    # Aux variables
    label_len = len(label)
    k = max_length - label_len # Missing length
    neuron_unique_tuple, neuron_common_tuple, neuron_coverable_mask_tuple, neuron_coverable_tuple, neuron_sum_tuple, common_space_extras_tuple, unique_space_extras_tuple = neuron_quantities
    _, neuron_common_sum = neuron_common_tuple
    _, neuron_hits = neuron_sum_tuple
    neuron_coverable , neuron_coverable_sum = neuron_coverable_tuple
    tot_size = max_size_mask * len(neuron_coverable)

    # Unpack improvement information
    improv_quantities, cum_sum_unique_extras = max_improvement
    improv_common_intersection, improv_unique_intersection, improv_common_extras, improv_unique_extras, _, _ = improv_quantities

    max_common_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='common_intersection', quantity_type='max', quantity_scope='sum')
    min_unique_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='unique_intersection', quantity_type='min', quantity_scope='sum')

    #zero_vector = np.zeros_like(max_common_intersection)
    top_k_common_intersection_sum = improv_common_intersection[k][TOP_INDEX_SUM]
    top_k_unique_intersection_sum = improv_unique_intersection[k][TOP_INDEX_SUM]

    bottom_1_common_intersection_sum = improv_common_intersection[0][BOTTOM_INDEX_SUM]

    if max_common_intersection_sum == 0:
        return 0.0, 0.0 
    
    # Max IoU

    max_intersection = min(
        min(max_common_intersection_sum + top_k_common_intersection_sum, neuron_common_sum) \
         + top_k_unique_intersection_sum,
           neuron_coverable_sum - min_unique_intersection_sum)
    min_union = num_hits
 

    max_iou = max_intersection / min_union

    # Save computation
    if max_iou == 0 or max_iou < minimum_threshold:
        return 0.0, 0.0
    if bottom_1_common_intersection_sum == 0:
        return max_iou, 0.0
    return max_iou, 0.0


def comb_or_andnot_chain_estimation(label, concept_quantities, neuron_quantities, max_improvement, *, num_hits, max_size_mask, max_length, minimum_threshold=0):
    # Aux variables
    label_len = len(label)
    k = max_length - label_len # Missing length

    neuron_unique_tuple, neuron_common_tuple, neuron_coverable_mask_tuple, neuron_coverable_tuple, neuron_sum_tuple, common_space_extras_tuple, unique_space_extras_tuple = neuron_quantities
    _, neuron_common_sum = neuron_common_tuple
    _, neuron_unique_sum = neuron_unique_tuple
    _, neuron_hits = neuron_sum_tuple
    neuron_coverable , neuron_coverable_sum = neuron_coverable_tuple
    tot_size = max_size_mask * len(neuron_coverable)

    # Unpack improvement information
    improv_quantities, cum_sum_unique_extras = max_improvement
    improv_common_intersection, improv_unique_intersection, improv_common_extras, improv_unique_extras, _, _ = improv_quantities

    # # Unpack max and min quantities

    max_common_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='common_intersection', quantity_type='max', quantity_scope='sum')
    max_unique_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='unique_intersection', quantity_type='max', quantity_scope='sum')
    min_unique_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='unique_intersection', quantity_type='min', quantity_scope='sum')
    max_common_extras_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='common_extras', quantity_type='max', quantity_scope='sum')
    max_unique_extras_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='unique_extras', quantity_type='max', quantity_scope='sum')
    min_unique_extras_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='unique_extras', quantity_type='min', quantity_scope='sum')
    

    neuron_coverable = neuron_coverable_tuple[0]

    # We discard the labels that cannot increase the IoU
    if max_common_intersection_sum + max_unique_intersection_sum == 0:
        return 0.0, 0.0

    top_k_common_intersection_sum = improv_common_intersection[k][TOP_INDEX_SUM]
    top_k_unique_intersection_sum = improv_unique_intersection[k][TOP_INDEX_SUM]
    #
    top_k_minus_1_extras_sum = improv_unique_extras[k-1][TOP_INDEX_SUM] + improv_common_extras[k-1][TOP_INDEX_SUM]
    tot_size = max_size_mask * len(neuron_coverable)
    # Max IoU
    max_intersection = min(min(max_common_intersection_sum  + top_k_common_intersection_sum, neuron_common_sum)  + min(max_unique_intersection_sum + top_k_unique_intersection_sum, neuron_unique_sum), neuron_coverable_sum)
    min_union = num_hits + min_unique_extras_sum
    max_iou = max_intersection / min_union
    
    # Save computation
    if max_iou == 0 or max_iou < minimum_threshold:
        return 0.0, 0.0
   
    # Min IoU
    min_intersection = min_unique_intersection_sum
    if min_intersection == 0:
        return max_iou, 0.0
    max_union = min(num_hits + max_unique_extras_sum + max_common_extras_sum + top_k_minus_1_extras_sum, tot_size)
    min_iou = min_intersection / max_union    

    return max_iou, min_iou


