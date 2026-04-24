import numpy as np

from . import formula as F
from . import heuristic_utils
from .heuristic_utils import TOP_INDEX_SAMPLE, BOTTOM_INDEX_SAMPLE, TOP_INDEX_SUM, BOTTOM_INDEX_SUM


def estimate_disjoint_label_info(label, left_quantities, right_quantities, neuron_unique, neuron_common, neuron_coverable, neuron_sum, max_size_mask, disjoint_masks, debug=False):

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
    left_common_intersection, left_unique_intersection, left_common_extras, left_unique_extras, left_common_uncovered, left_unique_uncovered = left_quantities_sample

    max_left_common_intersection_tuple, min_left_common_intersection_tuple = heuristic_utils.get_max_min_quantity(left_common_intersection)
    max_left_unique_intersection_tuple, min_left_unique_intersection_tuple = heuristic_utils.get_max_min_quantity(left_unique_intersection)
    max_left_common_extras_tuple, min_left_common_extras_tuple = heuristic_utils.get_max_min_quantity(left_common_extras)
    max_left_unique_extras_tuple, min_left_unique_extras_tuple = heuristic_utils.get_max_min_quantity(left_unique_extras)
    max_left_common_uncovered_tuple, min_left_common_uncovered_tuple = heuristic_utils.get_max_min_quantity(left_common_uncovered)
    max_left_unique_uncovered_tuple, min_left_unique_uncovered_tuple = heuristic_utils.get_max_min_quantity(left_unique_uncovered)
    
    max_left_common_intersection = max_left_common_intersection_tuple[0]

    max_left_common_intersection_sum = max_left_common_intersection_tuple[1]
    min_left_common_intersection_sum = min_left_common_intersection_tuple[1]
    max_left_unique_intersection_sum = max_left_unique_intersection_tuple[1]
    min_left_unique_intersection_sum = min_left_unique_intersection_tuple[1]
    max_left_common_extras_sum = max_left_common_extras_tuple[1]
    min_left_common_extras_sum = min_left_common_extras_tuple[1]
    max_left_unique_extras_sum = max_left_unique_extras_tuple[1]
    min_left_unique_extras_sum = min_left_unique_extras_tuple[1]
    max_left_common_uncovered_sum = max_left_common_uncovered_tuple[1]
    min_left_common_uncovered_sum = min_left_common_uncovered_tuple[1]
    max_left_unique_uncovered_sum = max_left_unique_uncovered_tuple[1]
    min_left_unique_uncovered_sum = min_left_unique_uncovered_tuple[1]

    right_common_intersection, right_unique_intersection, right_common_extras, right_unique_extras, right_common_uncovered, right_unique_uncovered = right_quantities_sample
    max_right_common_intersection_tuple, min_right_common_intersection_tuple = heuristic_utils.get_max_min_quantity(right_common_intersection)
    max_right_unique_intersection_tuple, min_right_unique_intersection_tuple = heuristic_utils.get_max_min_quantity(right_unique_intersection)
    max_right_common_extras_tuple, min_right_common_extras_tuple = heuristic_utils.get_max_min_quantity(right_common_extras)
    max_right_unique_extras_tuple, min_right_unique_extras_tuple = heuristic_utils.get_max_min_quantity(right_unique_extras)
    max_right_common_uncovered_tuple, min_right_common_uncovered_tuple = heuristic_utils.get_max_min_quantity(right_common_uncovered)
    max_right_unique_uncovered_tuple, min_right_unique_uncovered_tuple = heuristic_utils.get_max_min_quantity(right_unique_uncovered)
    max_right_common_intersection_sum = max_right_common_intersection_tuple[1]
    min_right_common_intersection_sum = min_right_common_intersection_tuple[1]
    max_right_unique_intersection_sum = max_right_unique_intersection_tuple[1]
    min_right_unique_intersection_sum = min_right_unique_intersection_tuple[1]
    max_right_common_extras_sum = max_right_common_extras_tuple[1]
    min_right_common_extras_sum = min_right_common_extras_tuple[1]
    max_right_unique_extras_sum = max_right_unique_extras_tuple[1]
    min_right_unique_extras_sum = min_right_unique_extras_tuple[1]
    max_right_common_uncovered_sum = max_right_common_uncovered_tuple[1]
    min_right_common_uncovered_sum = min_right_common_uncovered_tuple[1]
    max_right_unique_uncovered_sum = max_right_unique_uncovered_tuple[1]
    min_right_unique_uncovered_sum = min_right_unique_uncovered_tuple[1]

    max_left_common_intersection = max_left_common_intersection_tuple[0]
    max_right_common_intersection = max_right_common_intersection_tuple[0]
    max_right_common_uncovered = max_right_common_uncovered_tuple[0]
    max_left_common_uncovered = max_left_common_uncovered_tuple[0]
    min_left_common_uncovered = min_left_common_uncovered_tuple[0]
    min_right_common_uncovered = min_right_common_uncovered_tuple[0]
    

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
            #raise ValueError(f"Discarding formula: {label} because it does not change the intersection")
            return None, None, None, None, None, None

        min_common_uncovered_sum = max(min(min_left_common_uncovered_sum - max_right_common_intersection_sum, min_right_common_uncovered_sum - max_left_common_uncovered_sum), 0)
        max_common_uncovered_sum = max(min(max_left_common_uncovered_sum - min_right_common_intersection_sum, max_right_common_uncovered_sum - min_left_common_uncovered_sum), 0)
        min_unique_uncovered_sum = max(min(min_left_unique_uncovered_sum - max_right_unique_intersection_sum, min_right_unique_uncovered_sum - max_left_unique_uncovered_sum), 0)
        max_unique_uncovered_sum = max(min(max_left_unique_uncovered_sum - min_right_unique_intersection_sum, max_right_unique_uncovered_sum - min_left_unique_uncovered_sum), 0)

        min_unique_extras_sum = min_left_unique_extras_sum + min_right_unique_extras_sum
        max_unique_extras_sum = max_left_unique_extras_sum + max_right_unique_extras_sum


        min_common_extras_sum = max(min_left_common_extras_sum, min_right_common_extras_sum)
        max_common_extras_sum = max_left_common_extras_sum + max_right_common_extras_sum

        max_common_intersection = max_left_common_intersection + max_right_common_intersection

        max_common_uncovered = np.clip(np.minimum(max_left_common_uncovered - min_right_common_intersection, max_right_common_uncovered - min_left_common_intersection), a_min=0, a_max=None) # min(U_max^c, I_max^c(L))
        
        return ((max_common_intersection, max_common_intersection_sum), (None, min_common_intersection_sum)), ((None, max_unique_intersection_sum), (None, min_unique_intersection_sum)), \
        ((None, max_common_extras_sum), (None, min_common_extras_sum)), \
        ((None, max_unique_extras_sum), (None, min_unique_extras_sum)), \
        ((None, max_common_uncovered_sum), (None, min_common_uncovered_sum)), \
        ((None, max_unique_uncovered_sum), (None, min_unique_uncovered_sum))
    elif isinstance(label, F.And) and isinstance(label.right, F.Not):
        # Since they are disjoint, there is not a counter example of their presence together
        # Everything would end up the same of the left label (see commentetd part below)
        return None, None, None, None, None, None

    elif isinstance(label, F.And):
        # AND of disjoint labels is zero by definition
            return None, None, None, None, None, None
    else:
        raise ValueError(f"Unknown label type: {type(label)}")



def estimate_label_info(label, left_quantities, right_quantities, neuron_unique, neuron_common, neuron_coverable, neuron_sum, max_size_mask, neuron_coverable_sum, neuron_sum_sum, debug=False):

    """
    Estimate the label information for a given label based on its left and right quantities.

    Args:
        label (F.Formula): The label for which to estimate the label information.
        left_quantities (tuple): Quantities from the left child of the label.
        right_quantities (tuple): Quantities from the right child of the label.

    Returns:
        tuple: Estimated quantities for the label.
    """
    if debug:
        print(f"DEBUG: True")
    left_quantities_sample = left_quantities
    right_quantities_sample = right_quantities
    left_common_intersection, left_unique_intersection, left_common_extras, left_unique_extras, left_common_uncovered, left_unique_uncovered = left_quantities_sample
    
    max_left_common_intersection_tuple, min_left_common_intersection_tuple = heuristic_utils.get_max_min_quantity(left_common_intersection)
    max_left_unique_intersection_tuple, min_left_unique_intersection_tuple = heuristic_utils.get_max_min_quantity(left_unique_intersection)
    max_left_common_extras_tuple, min_left_common_extras_tuple = heuristic_utils.get_max_min_quantity(left_common_extras)
    max_left_unique_extras_tuple, min_left_unique_extras_tuple = heuristic_utils.get_max_min_quantity(left_unique_extras)
    max_left_common_uncovered_tuple, min_left_common_uncovered_tuple = heuristic_utils.get_max_min_quantity(left_common_uncovered)
    max_left_unique_uncovered_tuple, min_left_unique_uncovered_tuple = heuristic_utils.get_max_min_quantity(left_unique_uncovered)

    # Sum of the left quantities
    max_left_common_intersection_sum = max_left_common_intersection_tuple[1]
    min_left_common_intersection_sum = min_left_common_intersection_tuple[1]
    max_left_unique_intersection_sum = max_left_unique_intersection_tuple[1]
    min_left_unique_intersection_sum = min_left_unique_intersection_tuple[1]
    max_left_common_extras_sum = max_left_common_extras_tuple[1]
    min_left_common_extras_sum = min_left_common_extras_tuple[1]
    max_left_unique_extras_sum = max_left_unique_extras_tuple[1]
    min_left_unique_extras_sum = min_left_unique_extras_tuple[1]
    max_left_common_uncovered_sum = max_left_common_uncovered_tuple[1]
    min_left_common_uncovered_sum = min_left_common_uncovered_tuple[1]
    max_left_unique_uncovered_sum = max_left_unique_uncovered_tuple[1]
    min_left_unique_uncovered_sum = min_left_unique_uncovered_tuple[1]

    right_common_intersection, right_unique_intersection, right_common_extras, right_unique_extras, right_common_uncovered, right_unique_uncovered = right_quantities_sample
    max_right_common_intersection_tuple, min_right_common_intersection_tuple = heuristic_utils.get_max_min_quantity(right_common_intersection)
    max_right_unique_intersection_tuple, min_right_unique_intersection_tuple = heuristic_utils.get_max_min_quantity(right_unique_intersection)
    max_right_common_extras_tuple, min_right_common_extras_tuple = heuristic_utils.get_max_min_quantity(right_common_extras)
    max_right_unique_extras_tuple, min_right_unique_extras_tuple = heuristic_utils.get_max_min_quantity(right_unique_extras)
    max_right_common_uncovered_tuple, min_right_common_uncovered_tuple = heuristic_utils.get_max_min_quantity(right_common_uncovered)
    max_right_unique_uncovered_tuple, min_right_unique_uncovered_tuple = heuristic_utils.get_max_min_quantity(right_unique_uncovered)

    # Sum of the right quantities
    max_right_common_intersection_sum = max_right_common_intersection_tuple[1]
    min_right_common_intersection_sum = min_right_common_intersection_tuple[1]
    max_right_unique_intersection_sum = max_right_unique_intersection_tuple[1]
    min_right_unique_intersection_sum = min_right_unique_intersection_tuple[1]
    max_right_common_extras_sum = max_right_common_extras_tuple[1]
    min_right_common_extras_sum = min_right_common_extras_tuple[1]
    max_right_unique_extras_sum = max_right_unique_extras_tuple[1]
    min_right_unique_extras_sum = min_right_unique_extras_tuple[1]
    max_right_common_uncovered_sum = max_right_common_uncovered_tuple[1]
    min_right_common_uncovered_sum = min_right_common_uncovered_tuple[1]
    max_right_unique_uncovered_sum = max_right_unique_uncovered_tuple[1]
    min_right_unique_uncovered_sum = min_right_unique_uncovered_tuple[1]

    tot_size = max_size_mask*len(neuron_coverable)
    neuron_coverable_sum = neuron_coverable_sum
    max_left_common_intersection = max_left_common_intersection_tuple[0]
    max_right_common_intersection = max_right_common_intersection_tuple[0]
    max_right_common_uncovered = max_right_common_uncovered_tuple[0]
    min_right_common_uncovered = min_right_common_uncovered_tuple[0]
    min_right_common_intersection = min_right_common_intersection_tuple[0]
    min_left_common_intersection = min_left_common_intersection_tuple[0]
    max_left_common_uncovered_tuple, min_left_common_uncovered_tuple = heuristic_utils.get_max_min_quantity(left_common_uncovered)
    min_left_common_uncovered = min_left_common_uncovered_tuple[0]
    max_left_common_uncovered = max_left_common_uncovered_tuple[0]
    if isinstance(label, F.Or):

        max_common_intersection = max_left_common_intersection + max_right_common_intersection
        max_unique_intersection_sum = max_left_unique_intersection_sum + max_right_unique_intersection_sum
        min_unique_intersection_sum = min_left_unique_intersection_sum + min_right_unique_intersection_sum

        min_unique_extras_sum = min_left_unique_extras_sum + min_right_unique_extras_sum
        max_unique_extras_sum = max_left_unique_extras_sum + max_right_unique_extras_sum
        

        min_common_intersection_sum = max(min_left_common_intersection_sum, min_right_common_intersection_sum)
        max_common_intersection_sum = min(max(
            max_left_common_intersection_sum + min(max_left_common_uncovered_sum, max_right_common_intersection_sum),
            max_right_common_intersection_sum + min(max_right_common_uncovered_sum, max_left_common_intersection_sum)),
            neuron_coverable_sum - min_left_unique_intersection_sum - min_right_unique_intersection_sum
        )  
        max_common_intersection = np.minimum(
            
                max_left_common_intersection +
                max_right_common_intersection,
                neuron_coverable)
        max_common_intersection_sum = max_common_intersection.sum()
        if ((max_unique_intersection_sum <= max_left_unique_intersection_sum and \
            min_unique_intersection_sum <= min_left_unique_intersection_sum and \
            max_common_intersection_sum <= max_left_common_intersection_sum and \
            min_common_intersection_sum <= min_left_common_intersection_sum) or \
            (max_unique_intersection_sum <= max_right_unique_intersection_sum and \
            min_unique_intersection_sum <= min_right_unique_intersection_sum and \
            max_common_intersection_sum <= max_right_common_intersection_sum and \
            min_common_intersection_sum <= min_right_common_intersection_sum)):
            # If one of the two side does not change to the intersection, we can discard this formula
            return None, None, None, None, None, None

        min_common_uncovered_sum = max(min(min_left_common_uncovered_sum - max_right_common_intersection_sum, min_right_common_uncovered_sum - max_left_common_intersection_sum), 0)
        max_common_uncovered_sum = min(max_left_common_uncovered_sum, max_right_common_intersection_sum, 0)
        min_unique_uncovered_sum = max(min(min_left_unique_uncovered_sum - max_right_unique_intersection_sum, min_right_unique_uncovered_sum - max_left_unique_intersection_sum), 0)
        max_unique_uncovered_sum = max(min(max_left_unique_uncovered_sum - min_right_unique_intersection_sum, max_right_unique_uncovered_sum - min_left_unique_intersection_sum), 0)

        min_common_extras_sum = max(min_left_common_extras_sum, min_right_common_extras_sum)
        max_common_extras_sum = min(tot_size - neuron_sum_sum - min_left_unique_extras_sum - min_right_unique_extras_sum,
                                    max_left_common_extras_sum + max_right_common_extras_sum) # min(max_size_mask - N^u - N^c - E_min^u(L) - E_min^u(c), E_max^c(L) + E_max^c(c))
        if debug:
            print(f'OR components: Min Common Intersection: {min_common_intersection_sum}, Max Common Intersection: {max_common_intersection_sum}')
            print(f'Min Unique Intersection: {min_unique_intersection_sum}, Max Unique Intersection: {max_unique_intersection_sum}')
            print(f'Min Common Extras: {min_common_extras_sum}, Max Common Extras: {max_common_extras_sum}')
            print(f'Min Common Uncovered: {min_common_uncovered_sum}, Max Common Uncovered: {max_common_uncovered_sum}')
            print(f'Min Unique Uncovered: {min_unique_uncovered_sum}, Max Unique Uncovered: {max_unique_uncovered_sum}')
        min_common_uncovered = np.clip(
            min_left_common_uncovered - max_right_common_uncovered, a_min=0, a_max=None
        ) # max(0, I_min^c(L) + U_min^c - N^u - N^c)
        max_common_uncovered = np.minimum(
            max_left_common_uncovered, max_right_common_uncovered
        )
    elif isinstance(label, F.And) and isinstance(label.right, F.Not):

        max_unique_intersection_sum = max_left_unique_intersection_sum
        min_unique_intersection_sum = min_left_unique_intersection_sum

        min_common_intersection_sum = max(min_left_common_intersection_sum + min_right_common_uncovered_sum - neuron_sum_sum, 0)
        #max_common_intersection_sum = min(max_right_common_uncovered_sum, max_left_common_intersection_sum) # min(U_max^c, I_max^c(L))
        max_common_intersection = np.minimum(
            max_size_mask - min_right_common_intersection, max_left_common_intersection
        ) # min(U_max^c, I_max^c(L))
        max_common_intersection_sum = max_common_intersection.sum()
        min_common_uncovered_sum = max(min_left_common_uncovered_sum, min_right_common_intersection_sum) # max(U_min^L, I_min^c(c) + I_min^u(c))
        max_common_uncovered_sum = neuron_sum_sum - max(0, min_left_common_uncovered_sum + min_right_common_intersection_sum - neuron_sum_sum) # N^u + N^c - max(0, I_min^c(L) + I_min^c(c) - N^u - N^c)
        min_unique_uncovered_sum = min_left_unique_uncovered_sum 
        max_unique_uncovered_sum = max_left_unique_uncovered_sum

        min_unique_extras_sum = min_left_unique_extras_sum
        max_unique_extras_sum = max_left_unique_extras_sum

        min_common_extras_sum = max(min_left_common_extras_sum + (tot_size - neuron_sum_sum - max_right_unique_extras_sum - max_right_common_extras_sum) - (tot_size - neuron_sum_sum), 0)
        max_common_extras_sum = max_left_common_extras_sum

        if (min_common_extras_sum >= min_left_common_extras_sum and \
            max_common_intersection_sum <= max_left_common_intersection_sum and \
            min_common_intersection_sum <= min_left_common_intersection_sum):
            # If one of the two side does not contribute to the intersection, we can discard this formula
            if debug:
                print(f"Discarding formula: {label} because it does not change the intersection")
            return None, None, None, None, None, None

        if debug:
            print(f'NOT components: Min Common Intersection: {min_common_intersection_sum}, Max Common Intersection: {max_common_intersection_sum}')
            #print(f'Min Unique Intersection: {min_unique_intersection_sum}, Max Unique Intersection: {max_unique_intersection_sum}')
            print(f'Min Common Extras: {min_common_extras_sum}, Max Common Extras: {max_common_extras_sum}')
            print(f'Min Common Uncovered: {min_common_uncovered_sum}, Max Common Uncovered: {max_common_uncovered_sum}')
            print(f'Min Unique Uncovered: {min_unique_uncovered_sum}, Max Unique Uncovered: {max_unique_uncovered_sum}')
            print(max_right_common_uncovered_sum, max_left_common_intersection_sum)
            print(min_left_common_extras_sum, tot_size - neuron_sum_sum - max_right_unique_extras_sum - max_right_common_extras_sum, tot_size - neuron_sum_sum)
            #print(min_right_common_uncovered_sum, neuron_sum_sum, min_right_unique_uncovered_sum)
        min_common_uncovered = np.maximum(
            min_left_common_uncovered, min_right_common_intersection
        ) # max(U_min^c(L), U_min^c(c))
        max_common_uncovered = neuron_sum - np.clip(
            min_left_common_uncovered + min_right_common_intersection - neuron_sum, a_min=0, a_max=None
        ) # N^u + N^c - max(0, I_max^c(L) + U_max^c - N^u - N^c)

    elif isinstance(label, F.And):
        # The and always zeroes the unique elements
        min_unique_extras_sum = 0
        max_unique_extras_sum = 0
        max_unique_intersection_sum = 0
        min_unique_intersection_sum = 0

        min_common_extras_sum = max(min_left_common_extras_sum + min_right_common_extras_sum - (tot_size - neuron_sum_sum - max_left_unique_extras_sum - max_right_unique_extras_sum), 0)
    
        max_common_extras_sum = min(max_left_common_extras_sum, max_right_common_extras_sum)
        
        
        if (max_left_unique_extras_sum == 0 and \
            min_left_unique_extras_sum == 0 and \
        max_common_extras_sum >= max_left_common_extras_sum and \
        min_common_extras_sum >= min_left_common_extras_sum) or \
        (max_right_unique_extras_sum == 0 and \
        min_right_unique_extras_sum == 0 and \
        max_common_extras_sum >= max_right_common_extras_sum and \
        min_common_extras_sum >= min_right_common_extras_sum):
            # If one of the two side does not contribute to the intersection, we can discard this formula
            return None, None, None, None, None, None
        

        
        min_common_intersection_sum = max(min_left_common_intersection_sum + min_right_common_intersection_sum - neuron_coverable_sum, 0)
        
        # This will select always the smaller concept
        #max_common_intersection_sum = min(max_left_common_intersection_sum, max_right_common_intersection_sum)
        max_common_intersection = np.minimum(
            max_left_common_intersection, max_right_common_intersection) # min(I_max^c(L), I_max^c(c))
        max_common_intersection_sum = max_common_intersection.sum()
        min_common_uncovered_sum = max(min_left_common_uncovered_sum, min_right_common_uncovered_sum) # max(U_min^L, U_min^c)
        max_common_uncovered_sum = max(neuron_sum_sum - max(0, min_left_common_intersection_sum + min_right_common_intersection_sum - neuron_coverable_sum), 0) # N^u + N^c - max(0, I_min^c(L) + I_min^c(c) - N^u - N^c)
        # AND cannot preserve unique, so the unique uncovered always increases
        #min_unique_uncovered_sum = min(max_left_unique_uncovered_sum + max_right_unique_intersection_sum, max_right_unique_uncovered_sum + max_left_unique_intersection_sum, neuron_coverable_sum)
        min_unique_uncovered_sum = max(min_left_unique_uncovered_sum, min_right_unique_uncovered_sum) # max(U_min^L, U_min^c)
        max_unique_uncovered_sum = min(max_left_unique_uncovered_sum + max_right_unique_uncovered_sum, neuron_sum_sum) 
        

        if debug:
            print(f'AND components: Min Common Intersection: {min_common_intersection_sum}, Max Common Intersection: {max_common_intersection_sum}')
            print(f'Min Unique Intersection: {min_unique_intersection_sum}, Max Unique Intersection: {max_unique_intersection_sum}')
            print(f'Min Common Extras: {min_common_extras_sum}, Max Common Extras: {max_common_extras_sum}')
            print(f'Min Common Uncovered: {min_common_uncovered_sum}, Max Common Uncovered: {max_common_uncovered_sum}')
            print(f'Min Unique Uncovered: {min_unique_uncovered_sum}, Max Unique Uncovered: {max_unique_uncovered_sum}')
            print(max_left_common_intersection_sum, max_right_common_intersection_sum)
        min_common_uncovered = np.maximum(
            min_left_common_uncovered, min_right_common_uncovered
        )
        max_common_uncovered = neuron_coverable - np.clip(
            min_left_common_intersection + min_right_common_intersection - neuron_coverable, a_min=0, a_max=None
        )
    return ((max_common_intersection, max_common_intersection_sum), (None, min_common_intersection_sum)), ((None, max_unique_intersection_sum), (None, min_unique_intersection_sum)), \
        ((None, max_common_extras_sum), (None, min_common_extras_sum)), \
        ((None, max_unique_extras_sum), (None, min_unique_extras_sum)), \
        ((None, max_common_uncovered_sum), (None, min_common_uncovered_sum)), \
        ((None, max_unique_uncovered_sum), (None, min_unique_uncovered_sum))



def individual_estimation(label, common_intersection, unique_intersection, common_extras, unique_extras, common_uncovered, unique_uncovered, neuron_unique_tuple, neuron_common_tuple, neuron_coverable_tuple, neuron_sum_tuple, max_improvement, *, num_hits, max_size_mask, max_length, minimum_threshold=0, debug=False):
    # Unpack max and min quantities
    max_common_intersection_tuple, min_common_intersection_tuple = heuristic_utils.get_max_min_quantity(common_intersection)
    max_unique_intersection_tuple, min_unique_intersection_tuple = heuristic_utils.get_max_min_quantity(unique_intersection)
    max_common_extras_tuple, min_common_extras_tuple = heuristic_utils.get_max_min_quantity(common_extras)
    max_unique_extras_tuple, min_unique_extras_tuple = heuristic_utils.get_max_min_quantity(unique_extras)
    max_common_intersection_sum = max_common_intersection_tuple[1]
    min_common_intersection_sum = min_common_intersection_tuple[1]
    max_unique_intersection_sum = max_unique_intersection_tuple[1]
    min_unique_intersection_sum = min_unique_intersection_tuple[1]
    max_common_extras_sum = max_common_extras_tuple[1]
    min_common_extras_sum = min_common_extras_tuple[1]
    max_unique_extras_sum = max_unique_extras_tuple[1]
    min_unique_extras_sum = min_unique_extras_tuple[1]


    #max_label_intersection = max_common_intersection + max_unique_intersection
    max_label_common_intersection_sum =  max_common_intersection_sum + max_unique_intersection_sum
    if max_label_common_intersection_sum == 0:
        return 0.0, 0.0

    # Max IoU
    min_union = num_hits + min_unique_extras_sum + min_common_extras_sum
    max_iou = max_label_common_intersection_sum / min_union

    # Save computation
    if max_iou == 0 or max_iou < minimum_threshold:
        return 0.0, 0.0
    
    # Min IoU
    min_intersection = min_common_intersection_sum + min_unique_intersection_sum
    max_union = num_hits + max_unique_extras_sum + max_common_extras_sum
    min_iou = min_intersection / max_union

    return max_iou, min_iou


def or_chain_estimation(label, common_intersection, unique_intersection, common_extras, unique_extras, common_uncovered, unique_uncovered, neuron_unique_tuple, neuron_common_tuple, neuron_coverable_tuple, neuron_sum_tuple, max_improvement, *, num_hits, max_size_mask, max_length, minimum_threshold=0, debug=False, operation_type='sum'):
    # Aux variables
    label_len = len(label)
    k = max_length - label_len # Missing length   

    # Unpack improvement information
    improv_common_intersection, improv_unique_intersection, improv_common_extras, improv_unique_extras, _, _ = max_improvement


    # Unpack max and min quantities
    max_common_intersection_tuple, min_common_intersection_tuple = heuristic_utils.get_max_min_quantity(common_intersection)
    max_unique_intersection_tuple, min_unique_intersection_tuple = heuristic_utils.get_max_min_quantity(unique_intersection)
    max_common_extras_tuple, min_common_extras_tuple = heuristic_utils.get_max_min_quantity(common_extras)
    max_unique_extras_tuple, min_unique_extras_tuple = heuristic_utils.get_max_min_quantity(unique_extras)

    # Sum quantities
    max_common_intersection_sum = max_common_intersection_tuple[1]
    min_common_intersection_sum = min_common_intersection_tuple[1]
    max_unique_intersection_sum = max_unique_intersection_tuple[1]
    min_unique_intersection_sum = min_unique_intersection_tuple[1]
    max_common_extras_sum = max_common_extras_tuple[1]
    min_common_extras_sum = min_common_extras_tuple[1]
    max_unique_extras_sum = max_unique_extras_tuple[1]
    min_unique_extras_sum = min_unique_extras_tuple[1]
    neuron_coverable_sum = neuron_coverable_tuple[1]

    # Samples quantities
    neuron_coverable = neuron_coverable_tuple[0]

    # We discard the labels that cannot increase the IoU
    if max_common_intersection_sum + max_unique_intersection_sum == 0:
        return 0.0, 0.0
    
    #max_label_intersection = max_common_intersection + max_unique_intersection
    top_k_intersection_sum = improv_common_intersection[k][TOP_INDEX_SUM] + improv_unique_intersection[k][TOP_INDEX_SUM]
    top_k_extras_sum = improv_common_extras[k][TOP_INDEX_SUM] + improv_unique_extras[k][TOP_INDEX_SUM]
    bottom_1_intersection_sum = improv_common_intersection[0][BOTTOM_INDEX_SUM] + improv_unique_intersection[0][BOTTOM_INDEX_SUM]
    bottom_1_extras_sum = improv_common_extras[0][BOTTOM_INDEX_SUM] + improv_unique_extras[0][BOTTOM_INDEX_SUM]

    tot_size = max_size_mask*len(neuron_coverable)

    # Max IoU

    min_union = min(num_hits + min_common_extras_sum + min_unique_extras_sum + max(0, bottom_1_extras_sum - max_common_extras_sum - max_unique_extras_sum), tot_size)
    max_intersection = min(max_common_intersection_sum + max_unique_intersection_sum + top_k_intersection_sum, neuron_coverable_sum)
    max_iou = max_intersection / min_union

    # Save computation
    if max_iou == 0 or max_iou < minimum_threshold:
        return 0.0, 0.0
    
    # Min IoU
    min_intersection = max(min_common_intersection_sum + min_unique_intersection_sum, bottom_1_intersection_sum)
    max_union = min(num_hits + min_common_extras_sum + min_unique_extras_sum + top_k_extras_sum, tot_size)
    min_iou = min_intersection / max_union
    if debug:
        print(f"Label: {label}, Max IoU: {max_iou}, Min IoU: {min_iou}, Max Intersection: {max_intersection}, Min Union: {min_union}, Min Intersection: {min_intersection}, Max Union: {max_union}")
        print(f"Max Iou Numerator Components: {max_common_intersection_sum}, {max_unique_intersection_sum}")
        print(f"Max Iou Denominator Components: {min_common_extras_sum}, {min_unique_extras_sum}, {max(0, bottom_1_extras_sum - max_common_extras_sum - max_unique_extras_sum)}")
        print(f"Min Iou Numerator Components: {min_common_intersection_sum}, {min_unique_intersection_sum}")
        print(f"Min Iou Denominator Components:  {min_common_extras_sum}, {min_unique_extras_sum}")
    return max_iou, min_iou


def and_chain_estimation(label, common_intersection, unique_intersection, common_extras, unique_extras, common_uncovered, unique_uncovered, neuron_unique_tuple, neuron_common_tuple, neuron_coverable_tuple, neuron_sum_tuple, max_improvement, *, num_hits, max_size_mask, max_length, minimum_threshold=0, debug=False, operation_type='sum'):
    # Aux variables
    # Unpack improvement information
    improv_common_intersection, improv_unique_intersection, improv_common_extras, improv_unique_extras, _, _ = max_improvement

    # Unpack max and min quantities
    max_common_intersection_tuple, min_common_intersection_tuple = heuristic_utils.get_max_min_quantity(common_intersection)
    max_common_extras_tuple, min_common_extras_tuple = heuristic_utils.get_max_min_quantity(common_extras)
    max_unique_extras_tuple, min_unique_extras_tuple = heuristic_utils.get_max_min_quantity(unique_extras)
    
    # Sum quantities
    max_common_intersection_sum = max_common_intersection_tuple[1]
    min_common_extras_sum = min_common_extras_tuple[1]
    min_unique_extras_sum = min_unique_extras_tuple[1]

    # Samples quantities
    neuron_coverable = neuron_coverable_tuple[0]
    max_common_intersection = max_common_intersection_tuple[0]
    min_unique_extras = min_unique_extras_tuple[0]
    min_common_extras = min_common_extras_tuple[0]
    neuron_sum = neuron_sum_tuple[0]

    # We discard the labels that cannot increase the IoU
    if max_common_intersection_sum == 0:
        return 0.0, 0.0

    top_1_common_intersection = improv_common_intersection[0][TOP_INDEX_SAMPLE]
    bottom_1_common_extras_sum = improv_common_extras[0][BOTTOM_INDEX_SUM]
    bottom_1_common_intersection_sum = improv_common_intersection[0][BOTTOM_INDEX_SUM]
    tot_size = max_size_mask * len(neuron_coverable)

    # MaxIoU
    if operation_type == 'sum':
        max_intersection = max_common_intersection_sum
        min_union = min(num_hits + max(0, min_common_extras_sum  + bottom_1_common_extras_sum - (tot_size - num_hits - min_unique_extras_sum) ), tot_size)
    elif operation_type == 'sample':
        max_intersection = np.minimum(max_common_intersection, top_1_common_intersection).sum()
        bottom_1_common_extras = improv_common_extras[0][BOTTOM_INDEX_SAMPLE] 
        min_union = np.clip(neuron_sum + np.clip(
            min_common_extras +  bottom_1_common_extras - (max_size_mask - neuron_sum -min_unique_extras),
                a_min=0, a_max=max_size_mask), a_min=0, a_max=max_size_mask).sum()
    elif operation_type == 'hybrid':
        max_intersection = max_common_intersection_sum
        min_union = min(num_hits + max(0, min_common_extras_sum  + bottom_1_common_extras_sum - (tot_size - num_hits - min_unique_extras_sum) ), tot_size)
        if min_union == tot_size:
            # Search for an alternative lower overestimation
            bottom_1_common_extras = improv_common_extras[0][BOTTOM_INDEX_SAMPLE]
            min_union = np.clip(neuron_sum + np.clip(
                min_common_extras + bottom_1_common_extras - (max_size_mask - neuron_sum - min_unique_extras),
                    a_min=0, a_max=max_size_mask), a_min=0, a_max=max_size_mask).sum()


    max_iou = max_intersection / min_union

    # Save computation
    if max_iou == 0 or max_iou < minimum_threshold:
        return 0.0, 0.0
    if bottom_1_common_intersection_sum == 0:
        return max_iou, 0.0
    return max_iou, 0.0

def and_not_chain_estimation(label, common_intersection, unique_intersection, common_extras, unique_extras, common_uncovered, unique_uncovered, neuron_unique_tuple, neuron_common_tuple, neuron_coverable_tuple, neuron_sum_tuple, max_improvement, *, num_hits, max_size_mask, max_length, minimum_threshold=0, debug=False, operation_type='sum'):
    # Aux variables
    # Unpack improvement information
    improv_common_intersection, improv_unique_intersection, improv_common_extras, improv_unique_extras, improv_common_uncovered, improv_unique_uncovered = max_improvement

    # Unpack max and min quantities
    max_common_intersection_tuple, min_common_intersection_tuple = heuristic_utils.get_max_min_quantity(common_intersection)
    max_unique_intersection_tuple, min_unique_intersection_tuple = heuristic_utils.get_max_min_quantity(unique_intersection)
    max_common_extras_tuple, min_common_extras_tuple = heuristic_utils.get_max_min_quantity(common_extras)
    max_unique_extras_tuple, min_unique_extras_tuple = heuristic_utils.get_max_min_quantity(unique_extras)
    
    # Sum quantities
    max_common_intersection_sum = max_common_intersection_tuple[1]
    max_unique_intersection_sum = max_unique_intersection_tuple[1]
    min_unique_intersection_sum = min_unique_intersection_tuple[1]
    max_common_extras_sum = max_common_extras_tuple[1]
    max_unique_extras_sum = max_unique_extras_tuple[1]
    min_unique_extras_sum = min_unique_extras_tuple[1]

    # We discard the labels that cannot increase the IoU
    if max_common_intersection_sum + max_unique_intersection_sum == 0:
        return 0.0, 0.0
    

    # Max IoU
    if operation_type == 'sum' or operation_type == 'hybrid':
        max_intersection = max_unique_intersection_sum + max_common_intersection_sum
    min_union = num_hits + min_unique_extras_sum
    max_iou = max_intersection / min_union

    # Save computation
    if max_iou == 0 or max_iou < minimum_threshold:
        return 0.0, 0.0
    
    # Min IoU
    min_intersection = min_unique_intersection_sum
    max_union = num_hits + max_unique_extras_sum + max_common_extras_sum
    min_iou = min_intersection / max_union


    return max_iou, min_iou

def comb_and_or_chain_estimation(label, common_intersection, unique_intersection, common_extras, unique_extras, common_uncovered, unique_uncovered, neuron_unique_tuple, neuron_common_tuple, neuron_coverable_tuple, neuron_sum_tuple, max_improvement, *, num_hits, max_size_mask, max_length, minimum_threshold=0, debug=False, operation_type='sum'):
    # Aux variables
    label_len = len(label)
    k = max_length - label_len # Missing length   
    # Unpack improvement information
    improv_common_intersection, improv_unique_intersection, improv_common_extras, improv_unique_extras, _, _ = max_improvement

    # Unpack max and min quantities
    max_common_intersection_tuple, min_common_intersection_tuple = heuristic_utils.get_max_min_quantity(common_intersection)
    max_unique_intersection_tuple, min_unique_intersection_tuple = heuristic_utils.get_max_min_quantity(unique_intersection)

    # Sum quantities
    max_common_intersection_sum = max_common_intersection_tuple[1]
    min_unique_intersection_sum = min_unique_intersection_tuple[1]
    neuron_coverable_sum = neuron_coverable_tuple[1]

    # Samples quantities

    #zero_vector = np.zeros_like(max_common_intersection)
    top_k_common_intersection_sum = improv_common_intersection[k][TOP_INDEX_SUM]
    top_k_unique_intersection_sum = improv_unique_intersection[k][TOP_INDEX_SUM]

    bottom_1_common_intersection_sum = improv_common_intersection[0][BOTTOM_INDEX_SUM]

    if max_common_intersection_sum == 0:
        return 0.0, 0.0 
    
    # Max IoU

    max_intersection = min(max_common_intersection_sum + top_k_common_intersection_sum + top_k_unique_intersection_sum, neuron_coverable_sum - min_unique_intersection_sum)
    min_union = num_hits
 

    max_iou = max_intersection / min_union

    # Save computation
    if max_iou == 0 or max_iou < minimum_threshold:
        return 0.0, 0.0
    if bottom_1_common_intersection_sum == 0:
        return max_iou, 0.0
    # if debug:
    #     print(f"Comb AND-OR Chain")
    #     print(f"Max IoU: {max_iou}")
    #     print(f"Max Intersection: {max_intersection.sum()}, Min Union: {min_union.sum()}")
    return max_iou, 0.0


def comb_or_andnot_chain_estimation(label, common_intersection, unique_intersection, common_extras, unique_extras, common_uncovered, unique_uncovered, neuron_unique_tuple, neuron_common_tuple, neuron_coverable_tuple, neuron_sum_tuple, max_improvement, *, num_hits, max_size_mask, max_length, minimum_threshold=0, debug=False, operation_type='sum'):
    # Aux variables
    label_len = len(label)
    k = max_length - label_len # Missing length   
    # Unpack improvement information
    improv_common_intersection, improv_unique_intersection, improv_common_extras, improv_unique_extras, _, _ = max_improvement

    # Unpack max and min quantities
    max_common_intersection_tuple, min_common_intersection_tuple = heuristic_utils.get_max_min_quantity(common_intersection)
    max_unique_intersection_tuple, min_unique_intersection_tuple = heuristic_utils.get_max_min_quantity(unique_intersection)
    max_common_extras_tuple, min_common_extras_tuple = heuristic_utils.get_max_min_quantity(common_extras)
    max_unique_extras_tuple, min_unique_extras_tuple = heuristic_utils.get_max_min_quantity(unique_extras)

    # Sum quantities
    max_common_intersection_sum = max_common_intersection_tuple[1]
    max_unique_intersection_sum = max_unique_intersection_tuple[1]
    min_unique_intersection_sum = min_unique_intersection_tuple[1]
    max_common_extras_sum = max_common_extras_tuple[1]
    max_unique_extras_sum = max_unique_extras_tuple[1]
    min_unique_extras_sum = min_unique_extras_tuple[1]
    neuron_coverable_sum = neuron_coverable_tuple[1]

    # Samples quantities
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
    max_intersection = min(max_common_intersection_sum + max_unique_intersection_sum + top_k_common_intersection_sum + top_k_unique_intersection_sum, neuron_coverable_sum)
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


