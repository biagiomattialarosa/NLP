import numpy as np

from . import formula as F
from . import heuristic_utils
from .heuristic_utils import TOP_INDEX_SAMPLE, BOTTOM_INDEX_SAMPLE, TOP_INDEX_SUM, BOTTOM_INDEX_SUM

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
    
    max_left_common_intersection =  heuristic_utils.get_quantity(
        concepts_quantities=left_quantities,
        quantity_name='common_intersection',
        quantity_type='max',
        quantity_scope='sample'
    )
    min_left_common_intersection = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities,
        quantity_name='common_intersection',
        quantity_type='min',
        quantity_scope='sample'
    )
    max_left_unique_intersection = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities,
        quantity_name='unique_intersection',
        quantity_type='max',
        quantity_scope='sample'
    )
    min_left_unique_intersection = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities,
        quantity_name='unique_intersection',
        quantity_type='min',
        quantity_scope='sample'
    )
    max_left_common_extras = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities,
        quantity_name='common_extras',
        quantity_type='max',
        quantity_scope='sample'
    )
    min_left_common_extras = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities,
        quantity_name='common_extras',
        quantity_type='min',
        quantity_scope='sample'
    )
    max_left_unique_extras = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities,
        quantity_name='unique_extras',
        quantity_type='max',
        quantity_scope='sample'
    )
    min_left_unique_extras = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities,
        quantity_name='unique_extras',
        quantity_type='min',
        quantity_scope='sample'
    )
    # max_left_common_uncovered = heuristic_utils.get_quantity(
    #     concepts_quantities=left_quantities,
    #     quantity_name='common_uncovered',
    #     quantity_type='max',
    #     quantity_scope='sample'
    # )
    # min_left_common_uncovered = heuristic_utils.get_quantity(
    #     concepts_quantities=left_quantities,
    #     quantity_name='common_uncovered',
    #     quantity_type='min',
    #     quantity_scope='sample'
    # )
    # max_left_unique_uncovered = heuristic_utils.get_quantity(
    #     concepts_quantities=left_quantities,
    #     quantity_name='unique_uncovered',
    #     quantity_type='max',
    #     quantity_scope='sample'
    # )
    # min_left_unique_uncovered = heuristic_utils.get_quantity(
    #     concepts_quantities=left_quantities,
    #     quantity_name='unique_uncovered',
    #     quantity_type='min',
    #     quantity_scope='sample'
    # )

    max_left_unique_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities, quantity_name='unique_intersection', quantity_type='max', quantity_scope='sum')
    min_left_unique_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities, quantity_name='unique_intersection', quantity_type='min', quantity_scope='sum')
    max_left_common_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities, quantity_name='common_intersection', quantity_type='max', quantity_scope='sum')
    min_left_common_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities, quantity_name='common_intersection', quantity_type='min', quantity_scope='sum')


    # right_common_intersection, right_unique_intersection, right_common_extras, right_unique_extras, right_common_uncovered, right_unique_uncovered = right_quantities_sample
    # max_right_common_intersection_tuple, min_right_common_intersection_tuple = heuristic_utils.get_max_min_quantity(right_common_intersection)
    # max_right_unique_intersection_tuple, min_right_unique_intersection_tuple = heuristic_utils.get_max_min_quantity(right_unique_intersection)
    # max_right_common_extras_tuple, min_right_common_extras_tuple = heuristic_utils.get_max_min_quantity(right_common_extras)
    # max_right_unique_extras_tuple, min_right_unique_extras_tuple = heuristic_utils.get_max_min_quantity(right_unique_extras)
    # max_right_common_uncovered_tuple, min_right_common_uncovered_tuple = heuristic_utils.get_max_min_quantity(right_common_uncovered)
    # max_right_unique_uncovered_tuple, min_right_unique_uncovered_tuple = heuristic_utils.get_max_min_quantity(right_unique_uncovered)
    max_right_common_intersection = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities,
        quantity_name='common_intersection',
        quantity_type='max',
        quantity_scope='sample'
    )
    min_right_common_intersection = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities,
        quantity_name='common_intersection',
        quantity_type='min',
        quantity_scope='sample'
    )
    max_right_unique_intersection = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities,
        quantity_name='unique_intersection',
        quantity_type='max',
        quantity_scope='sample'
    )
    min_right_unique_intersection = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities,
        quantity_name='unique_intersection',
        quantity_type='min',
        quantity_scope='sample'
    )
    max_right_common_extras = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities,
        quantity_name='common_extras',
        quantity_type='max',
        quantity_scope='sample'
    )
    min_right_common_extras = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities,
        quantity_name='common_extras',
        quantity_type='min',
        quantity_scope='sample'
    )
    max_right_unique_extras = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities,
        quantity_name='unique_extras',
        quantity_type='max',
        quantity_scope='sample'
    )
    min_right_unique_extras = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities,
        quantity_name='unique_extras',
        quantity_type='min',
        quantity_scope='sample'
    )
    # max_right_common_uncovered = heuristic_utils.get_quantity(
    #     concepts_quantities=right_quantities,
    #     quantity_name='common_uncovered',
    #     quantity_type='max',
    #     quantity_scope='sample'
    # )
    # min_right_common_uncovered = heuristic_utils.get_quantity(
    #     concepts_quantities=right_quantities,
    #     quantity_name='common_uncovered',
    #     quantity_type='min',
    #     quantity_scope='sample'
    # )
    # max_right_unique_uncovered = heuristic_utils.get_quantity(
    #     concepts_quantities=right_quantities,
    #     quantity_name='unique_uncovered',
    #     quantity_type='max',
    #     quantity_scope='sample'
    # )
    # min_right_unique_uncovered = heuristic_utils.get_quantity(
    #     concepts_quantities=right_quantities,
    #     quantity_name='unique_uncovered',
    #     quantity_type='min',
    #     quantity_scope='sample'
    # )

    max_right_unique_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities, quantity_name='unique_intersection', quantity_type='max', quantity_scope='sum')
    min_right_unique_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities, quantity_name='unique_intersection', quantity_type='min', quantity_scope='sum')
    max_right_common_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities, quantity_name='common_intersection', quantity_type='max', quantity_scope='sum')
    min_right_common_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities, quantity_name='common_intersection', quantity_type='min', quantity_scope='sum') 
    
    if (max_right_unique_intersection_sum == 0 and min_right_unique_intersection_sum == 0 and \
        max_right_common_intersection_sum == 0 and min_right_common_intersection_sum == 0) or \
        (max_left_unique_intersection_sum == 0 and min_left_unique_intersection_sum == 0 and \
         max_left_common_intersection_sum == 0 and min_left_common_intersection_sum == 0):
        return None

    if isinstance(label, F.Or):
        max_unique_intersection = max_left_unique_intersection + max_right_unique_intersection # I_max^u(L) + I_max^u(c)
        min_unique_intersection = min_left_unique_intersection + min_right_unique_intersection # I_min^u(L) + I_min^u(c)

        max_common_intersection = max_left_common_intersection + max_right_common_intersection # I_max^c(L) + I_max^c(c)
        min_common_intersection = min_left_common_intersection + min_right_common_intersection # I_min^c(L) + I_min^c(c)

        # min_uncovered  = np.clip(
        #     min_left_common_uncovered + min_left_unique_uncovered - max_right_common_intersection - max_right_unique_intersection, a_min=0, a_max=None) # max(0, U_min^L - I_max^c(c) - I_max^u(c))
        # max_uncovered = np.clip(
        #     max_left_common_uncovered + max_left_unique_uncovered - min_right_common_intersection - min_right_unique_intersection, a_min=0, a_max=None)

        # min_common_uncovered = np.clip(np.minimum(min_left_common_uncovered - max_right_common_intersection, min_right_common_uncovered - max_left_common_intersection), a_min=0, a_max=None) # max(0, I_min^c(L) + U_min^c - N^u - N^c)
        # max_common_uncovered = np.clip(np.minimum(max_left_common_uncovered - min_right_common_intersection, max_right_common_uncovered - min_left_common_intersection), a_min=0, a_max=None) # min(U_max^c, I_max^c(L))
        # min_unique_uncovered = np.clip(np.minimum(min_left_unique_uncovered - max_right_unique_intersection, min_right_unique_uncovered - max_left_unique_intersection), a_min=0, a_max=None) # max(0, I_min^u(L) + U_min^u - N^u - N^c)
        # max_unique_uncovered = np.clip(np.minimum(max_left_unique_uncovered - min_right_unique_intersection, max_right_unique_uncovered - min_left_unique_intersection), a_min=0, a_max=None) # min(U_max^u, I_max^u(L))



        min_unique_extras = min_left_unique_extras + min_right_unique_extras # E_min^u(L) + E_min^u(c)
        max_unique_extras = max_left_unique_extras + max_right_unique_extras # E_max^u(L) + E_max^u(c)
        min_common_extras = min_left_common_extras + min_right_common_extras # E_min^c(L) + E_min^c(c)
        max_common_extras = max_left_common_extras + max_right_common_extras # E_max^c(L) + E_max^c(c)


        return ((max_common_intersection, max_common_intersection.sum()), (min_common_intersection, min_common_intersection.sum())), ((max_unique_intersection, max_unique_intersection.sum()), (min_unique_intersection, min_unique_intersection.sum())), \
        ((max_common_extras, max_common_extras.sum()), (min_common_extras, min_common_extras.sum())), \
        ((max_unique_extras, max_unique_extras.sum()), (min_unique_extras, min_unique_extras.sum())), \
        ((None, None), (None, None)), \
        ((None, None), (None, None))
        #((max_common_uncovered, max_common_uncovered.sum()), (min_common_uncovered, min_common_uncovered.sum())), \
        #((max_unique_uncovered, max_unique_uncovered.sum()), (min_unique_uncovered, min_unique_uncovered.sum()))

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
    neuron_unique_tuple, neuron_common_tuple, neuron_coverable_tuple, neuron_sum_tuple, common_space_extras_tuple, _  = neuron_quantities
    neuron_sum, _ = neuron_sum_tuple
    neuron_common, _ = neuron_common_tuple
    neuron_unique, _ = neuron_unique_tuple
    common_space_extras, _ = common_space_extras_tuple

    # left_quantities_sample = left_quantities
    # right_quantities_sample = right_quantities
    # left_common_intersection, left_unique_intersection, left_common_extras, left_unique_extras, left_common_uncovered, left_unique_uncovered = left_quantities_sample
    # max_left_common_intersection_tuple, min_left_common_intersection_tuple = heuristic_utils.get_max_min_quantity(left_common_intersection)
    # max_left_unique_intersection_tuple, min_left_unique_intersection_tuple = heuristic_utils.get_max_min_quantity(left_unique_intersection)
    # max_left_common_extras_tuple, min_left_common_extras_tuple = heuristic_utils.get_max_min_quantity(left_common_extras)
    # max_left_unique_extras_tuple, min_left_unique_extras_tuple = heuristic_utils.get_max_min_quantity(left_unique_extras)
    # max_left_common_uncovered_tuple, min_left_common_uncovered_tuple = heuristic_utils.get_max_min_quantity(left_common_uncovered)
    # max_left_unique_uncovered_tuple, min_left_unique_uncovered_tuple = heuristic_utils.get_max_min_quantity(left_unique_uncovered)
    max_left_common_intersection = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities,
        quantity_name='common_intersection',
        quantity_type='max',
        quantity_scope='sample'
    )
    min_left_common_intersection = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities,
        quantity_name='common_intersection',
        quantity_type='min',
        quantity_scope='sample'
    )
    max_left_unique_intersection = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities,
        quantity_name='unique_intersection',
        quantity_type='max',
        quantity_scope='sample'
    )
    min_left_unique_intersection = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities,
        quantity_name='unique_intersection',
        quantity_type='min',
        quantity_scope='sample'
    )
    max_left_common_extras = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities,
        quantity_name='common_extras',
        quantity_type='max',
        quantity_scope='sample'
    )
    min_left_common_extras = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities,
        quantity_name='common_extras',
        quantity_type='min',
        quantity_scope='sample'
    )
    max_left_unique_extras = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities,
        quantity_name='unique_extras',
        quantity_type='max',
        quantity_scope='sample'
    )
    min_left_unique_extras = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities,
        quantity_name='unique_extras',
        quantity_type='min',
        quantity_scope='sample'
    )
    # max_left_common_uncovered = heuristic_utils.get_quantity(
    #     concepts_quantities=left_quantities,
    #     quantity_name='common_uncovered',
    #     quantity_type='max',
    #     quantity_scope='sample'
    # )
    # min_left_common_uncovered = heuristic_utils.get_quantity(
    #     concepts_quantities=left_quantities,
    #     quantity_name='common_uncovered',
    #     quantity_type='min',
    #     quantity_scope='sample'
    # )
    # max_left_unique_uncovered = heuristic_utils.get_quantity(
    #     concepts_quantities=left_quantities,
    #     quantity_name='unique_uncovered',
    #     quantity_type='max',
    #     quantity_scope='sample'
    # )
    # min_left_unique_uncovered = heuristic_utils.get_quantity(
    #     concepts_quantities=left_quantities,
    #     quantity_name='unique_uncovered',
    #     quantity_type='min',
    #     quantity_scope='sample'
    # )

    # Sum of the left quantities
    max_left_unique_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities,
        quantity_name='unique_intersection',
        quantity_type='max',
        quantity_scope='sum'
    )
    min_left_unique_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities,
        quantity_name='unique_intersection',
        quantity_type='min',
        quantity_scope='sum'
    )
    max_left_unique_extras_sum = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities,
        quantity_name='unique_extras',
        quantity_type='max',
        quantity_scope='sum'
    )
    min_left_unique_extras_sum = heuristic_utils.get_quantity(
        concepts_quantities=left_quantities,
        quantity_name='unique_extras',
        quantity_type='min',
        quantity_scope='sum'
    )

    # right_common_intersection, right_unique_intersection, right_common_extras, right_unique_extras, right_common_uncovered, right_unique_uncovered = right_quantities_sample
    # max_right_common_intersection_tuple, min_right_common_intersection_tuple = heuristic_utils.get_max_min_quantity(right_common_intersection)
    # max_right_unique_intersection_tuple, min_right_unique_intersection_tuple = heuristic_utils.get_max_min_quantity(right_unique_intersection)
    # max_right_common_extras_tuple, min_right_common_extras_tuple = heuristic_utils.get_max_min_quantity(right_common_extras)
    # max_right_unique_extras_tuple, min_right_unique_extras_tuple = heuristic_utils.get_max_min_quantity(right_unique_extras)
    # max_right_common_uncovered_tuple, min_right_common_uncovered_tuple = heuristic_utils.get_max_min_quantity(right_common_uncovered)
    # max_right_unique_uncovered_tuple, min_right_unique_uncovered_tuple = heuristic_utils.get_max_min_quantity(right_unique_uncovered)
    max_right_common_intersection = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities,
        quantity_name='common_intersection',
        quantity_type='max',
        quantity_scope='sample'
    )
    min_right_common_intersection = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities,
        quantity_name='common_intersection',
        quantity_type='min',
        quantity_scope='sample'
    )
    max_right_unique_intersection = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities,
        quantity_name='unique_intersection',
        quantity_type='max',
        quantity_scope='sample'
    )
    min_right_unique_intersection = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities,
        quantity_name='unique_intersection',
        quantity_type='min',
        quantity_scope='sample'
    )
    max_right_common_extras = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities,
        quantity_name='common_extras',
        quantity_type='max',
        quantity_scope='sample'
    )
    min_right_common_extras = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities,
        quantity_name='common_extras',
        quantity_type='min',
        quantity_scope='sample'
    )
    max_right_unique_extras = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities,
        quantity_name='unique_extras',
        quantity_type='max',
        quantity_scope='sample'
    )
    min_right_unique_extras = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities,
        quantity_name='unique_extras',
        quantity_type='min',
        quantity_scope='sample'
    )
    max_right_common_uncovered = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities,
        quantity_name='common_uncovered',
        quantity_type='max',
        quantity_scope='sample'
    )
    min_right_common_uncovered = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities,
        quantity_name='common_uncovered',
        quantity_type='min',
        quantity_scope='sample'
    )
    # max_right_unique_uncovered = heuristic_utils.get_quantity(
    #     concepts_quantities=right_quantities,
    #     quantity_name='unique_uncovered',
    #     quantity_type='max',
    #     quantity_scope='sample'
    # )
    # min_right_unique_uncovered = heuristic_utils.get_quantity(
    #     concepts_quantities=right_quantities,
    #     quantity_name='unique_uncovered',
    #     quantity_type='min',
    #     quantity_scope='sample'
    # )


  

    # Sum of the right quantities
    max_right_unique_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities,
        quantity_name='unique_intersection',
        quantity_type='max',
        quantity_scope='sum'
    )
    min_right_unique_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities,
        quantity_name='unique_intersection',
        quantity_type='min',
        quantity_scope='sum'
    )
    max_right_unique_extras_sum = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities,
        quantity_name='unique_extras',
        quantity_type='max',
        quantity_scope='sum'
    )
    min_right_unique_extras_sum = heuristic_utils.get_quantity(
        concepts_quantities=right_quantities,
        quantity_name='unique_extras',
        quantity_type='min',
        quantity_scope='sum'
    )
 
    #space_extras = max_size_mask - neuron_sum

    if isinstance(label, F.Or):
        max_unique_intersection = max_left_unique_intersection + max_right_unique_intersection # I_max^u(L) + I_max^u(c)
        min_unique_intersection = min_left_unique_intersection + min_right_unique_intersection # I_min^u(L) + I_min^u(c)

        max_unique_intersection_sum = max_left_unique_intersection_sum + max_right_unique_intersection_sum
        min_unique_intersection_sum = min_left_unique_intersection_sum + min_right_unique_intersection_sum

        

        min_common_intersection = np.maximum(min_left_common_intersection, min_right_common_intersection) # max(I_min^c(L), I_min^c(c))
        # max_common_intersection = np.minimum(
        #     np.maximum(
        #         max_left_common_intersection + np.minimum(
        #             max_left_common_uncovered,
        #             max_right_common_intersection),
        #         max_right_common_intersection + np.minimum(
        #             max_right_common_uncovered,
        #             max_left_common_intersection)),
        #         neuron_common)
        
        max_common_intersection = np.minimum(
                max_left_common_intersection + 
                max_right_common_intersection,
                neuron_common)

        if ((max_unique_intersection_sum <= max_left_unique_intersection_sum and \
            min_unique_intersection_sum <= min_left_unique_intersection_sum and \
            np.all(max_common_intersection <= max_left_common_intersection) and \
            np.all(min_common_intersection <= min_left_common_intersection)) or \
            (max_unique_intersection_sum <= max_right_unique_intersection_sum and \
            min_unique_intersection_sum <= min_right_unique_intersection_sum and \
            np.all(max_common_intersection <= max_right_common_intersection) and \
            np.all(min_common_intersection <= min_right_common_intersection))):
            # If one of the two side does not change to the intersection, we can discard this formula
            return None

        # min_uncovered  = np.clip(
        #     min_left_uncovered - max_right_common_intersection - max_right_unique_intersection, a_min=0, a_max=None) # max(0, U_min^L - I_max^c(c) - I_max^u(c))
        # max_uncovered = np.minimum(
        #     max_left_uncovered, max_right_uncovered) # min(U_max^L, U_max^c)

        # min_common_uncovered = np.minimum(
        #     np.clip(
        #     min_left_common_uncovered - max_right_common_uncovered, a_min=0, a_max=None
        #     ), # max(0, I_min^c(L) + U_min^c - N^u - N^c),
        #     np.clip(
        #     min_right_common_uncovered - max_left_common_uncovered, a_min=0, a_max=None
        #     ) # max(0, I_min^c(c) + U_min^
        # )
        # max_common_uncovered = np.minimum(
        #     max_left_common_uncovered, max_right_common_uncovered
        # )
        # min_unique_uncovered = np.minimum(
        #      np.clip(
        #     min_left_unique_uncovered - max_right_unique_uncovered, a_min=0, a_max=None
        # ), # max(0, I_min^u(L) + U_min^u - N^u - N^c)
        # np.clip(
        #     min_right_unique_uncovered - max_left_unique_uncovered, a_min=0, a_max=None
        # ) # max(0, I_min^u(c) + U_min^u - N^u - N^c)
        #  )

        # # This is because unique elements cannot be shared
        # max_unique_uncovered = np.minimum(np.clip(
        #     max_left_unique_uncovered - min_right_unique_uncovered, a_min=0, a_max=None
        # ),
        # np.clip(
        #     max_right_unique_uncovered - min_left_unique_uncovered, a_min=0, a_max=None
        # )) # min(U_max^u(L), U_max^u(c))

        # min_unique_uncovered = np.minimum(
        #     np.clip(
        #         min_left_unique_uncovered - max_right_unique_intersection, a_min=0, a_max=None
        #     ),
        #     np.clip(
        #         min_right_unique_uncovered - max_left_unique_intersection, a_min=0, a_max=None
        #     ) # This is because unique elements cannot be shared
        # )
        # max_unique_uncovered = np.minimum(
        #     np.clip(
        #         max_left_unique_uncovered - min_right_unique_intersection, a_min=0, a_max=None
        #     ),
        #     np.clip(
        #         max_right_unique_uncovered - min_left_unique_intersection, a_min=0, a_max=None
        #     )
        # )

        # min_common_uncovered = np.minimum(
        #     np.clip(
        #     min_left_common_uncovered - max_right_common_intersection, a_min=0, a_max=None
        #     ), # max(0, I_min^c(L) + U_min^c - N^u - N^c),
        #     np.clip(
        #     min_right_common_uncovered - max_left_common_intersection, a_min=0, a_max=None
        #     ) # max(0, I_min^c(c) + U_min^
        # )
        # max_common_uncovered = np.minimum(
        #     max_left_common_uncovered, max_right_common_uncovered
        # )

        min_unique_extras = min_left_unique_extras + min_right_unique_extras # E_min^u(L) + E_min^u(c)
        max_unique_extras = max_left_unique_extras + max_right_unique_extras # E_max^u(L) + E_max^u(c)
        min_unique_extras_sum = min_left_unique_extras_sum + min_right_unique_extras_sum
        max_unique_extras_sum = max_left_unique_extras_sum + max_right_unique_extras_sum

        min_common_extras = np.maximum(min_left_common_extras, min_right_common_extras) # max(E_min^c(L), E_min^c(c))
        max_common_extras = np.minimum(
            common_space_extras,
            max_left_common_extras + max_right_common_extras
        ) # min(max_size_mask - N^u - N^c - E_min^u(L) - E_min^u(c), E_max^c(L) + E_max^c(c))
    
    elif isinstance(label, F.And) and isinstance(label.right, F.Not):
        #neuron_activation =  neuron_unique + neuron_common
        max_unique_intersection = max_left_unique_intersection
        min_unique_intersection = min_left_unique_intersection

        max_unique_intersection_sum = max_left_unique_intersection_sum
        min_unique_intersection_sum = min_left_unique_intersection_sum
        # min_right_uncovered = min_right_common_uncovered + min_right_unique_uncovered
        # max_right_uncovered = max_right_common_uncovered + max_right_unique_uncovered
        min_common_intersection = np.clip(
            min_left_common_intersection + min_right_common_uncovered - neuron_common, a_min=0, a_max=None
        ) # max(0, I_min^c(L) + U_min^c - N^u - N^c)
        max_common_intersection = np.minimum(
            max_right_common_uncovered, max_left_common_intersection
        ) # min(U_max^c, I_max^c(L))

        # # min_uncovered = np.maximum(
        # #     min_left_uncovered, min_right_common_intersection + min_right_unique_intersection
        # # ) # max(U_min^L, I_min^c(c) + I_min^u(c))
        # # max_uncovered = neuron_coverable - np.clip(
        # #     max_left_common_intersection + max_right_uncovered - neuron_sum, a_min=0, a_max=None
        # # ) # N^u + N^c - max(0, I_max^c(L) + U_max^c - N^u - N^c)
        # min_common_uncovered = np.maximum(
        #     min_left_common_uncovered, min_right_common_intersection
        # ) # max(U_min^c(L), U_min^c(c))
        # max_common_uncovered = neuron_common - np.clip(
        #     min_left_common_uncovered + min_right_common_intersection - neuron_common, a_min=0, a_max=None
        # ) # N^u + N^c - max(0, I_max^c(L) + U_max^c - N^u - N^c)

        # # This is because any unique_intersection of right is already included in the left unique uncovered
        # min_unique_uncovered = min_left_unique_uncovered
        # max_unique_uncovered = max_left_unique_uncovered
        
        min_unique_extras = min_left_unique_extras # E_min^u(L)
        max_unique_extras = max_left_unique_extras # E_max^u(L)
        min_unique_extras_sum = min_left_unique_extras_sum
        max_unique_extras_sum = max_left_unique_extras_sum

        min_common_extras = np.clip(
            min_left_common_extras - max_right_common_extras,
            a_min=0, a_max=None
        )

        max_common_extras = np.minimum(max_left_common_extras, common_space_extras - min_right_common_extras)
        # if ( np.all(min_common_extras >= min_left_common_extras) and \
        #     np.all(max_common_intersection <= max_left_common_intersection) and \
        #     np.all(min_common_intersection <= min_left_common_intersection)):
        #     # If one of the two side does not contribute to the intersection, we can discard this formula
        #     return None

    elif isinstance(label, F.And):
        min_unique_extras = np.zeros_like(min_left_unique_extras)
        max_unique_extras =  np.zeros_like(min_left_unique_extras)
        min_unique_extras_sum = 0
        max_unique_extras_sum = 0

        min_common_extras = np.clip(
            min_left_common_extras + min_right_common_extras - common_space_extras
        , a_min=0, a_max=None
        ) # max(0, E^c(L) + E^c(c) - (max_size_mask - N^u - N^c))
        max_common_extras = np.minimum(
            max_left_common_extras,
            max_right_common_extras
        ) # min(E^c(L), E^c(c))
        # if (max_left_unique_extras_sum == 0 and \
        #     min_left_unique_extras_sum == 0 and \
        # np.all(max_common_extras >= max_left_common_extras) and \
        # np.all(min_common_extras >= min_left_common_extras)) or \
        # (max_right_unique_extras_sum == 0 and \
        # min_right_unique_extras_sum == 0 and \
        # np.all(max_common_extras >= max_right_common_extras) and \
        # np.all(min_common_extras >= min_right_common_extras) ):
        #     # If one of the two side does not contribute to the intersection, we can discard this formula
        #     return None

        max_unique_intersection = np.zeros_like(min_left_unique_intersection)
        min_unique_intersection = np.zeros_like(min_left_unique_intersection)

        max_unique_intersection_sum = 0
        min_unique_intersection_sum = 0
        
        min_common_intersection = np.clip(
            min_left_common_intersection + min_right_common_intersection - neuron_common,
            a_min=0, a_max=None
            ) # max(0, I_min^c(L) + I_min^c(c) - N^u - N^c)
        max_common_intersection = np.minimum(
            max_left_common_intersection, max_right_common_intersection) # min(I_max^c(L), I_max^c(c))
        
        # min_uncovered = np.maximum(
        #     min_left_uncovered, min_right_uncovered) # max(U_min^L, U_min^c)
        # max_uncovered = neuron_coverable - np.clip(
        #     min_left_common_intersection + min_right_common_intersection - neuron_coverable, a_min=0, a_max=None
        # ) # N^u + N^c - max(0, I_min^c(L) + I_min^c(c) - N^u - N^c)
        # min_common_uncovered = np.maximum(
        #     min_left_common_uncovered, min_right_common_uncovered
        # )
        # max_common_uncovered = neuron_common - np.clip(
        #     min_left_common_intersection + min_right_common_intersection - neuron_common, a_min=0, a_max=None
        # )
        # min_unique_uncovered = np.maximum(
        #     min_left_unique_uncovered, min_right_unique_uncovered
        # )
        # max_unique_uncovered = np.maximum(
        #     max_left_unique_uncovered + max_right_unique_uncovered, 
        #     neuron_unique
        # )
        
    else:
        raise ValueError(f"Unknown label type: {type(label)}")

    # assert np.all(np.less_equal(max_common_intersection+max_unique_intersection, neuron_coverable))
    # assert np.all(np.less_equal(min_common_intersection+min_unique_intersection, neuron_coverable))

    return ((max_common_intersection, max_common_intersection.sum()), (min_common_intersection, min_common_intersection.sum())), ((max_unique_intersection, max_unique_intersection_sum), (min_unique_intersection, min_unique_intersection_sum)), \
        ((max_common_extras, max_common_extras.sum()), (min_common_extras, min_common_extras.sum())), \
        ((max_unique_extras, max_unique_extras_sum), (min_unique_extras, min_unique_extras_sum)), \
                ((None, None), (None, None)), \
        ((None, None), (None, None))
        # ((max_common_uncovered, max_common_uncovered.sum()), (min_common_uncovered, min_common_uncovered.sum())), \
        # ((max_unique_uncovered, max_unique_uncovered.sum()), (min_unique_uncovered, min_unique_uncovered.sum()))
        #---> qui errore ((max_uncovered, max_uncovered.sum()), (min_uncovered, min_uncovered.sum()))


def individual_estimation(label, concept_quantities, neuron_quantities, max_improvement, *, num_hits, max_size_mask, max_length, minimum_threshold=0):
    
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


def or_chain_estimation(label, concept_quantities, neuron_quantities, max_improvement, *, num_hits, max_size_mask, max_length, minimum_threshold=0):
    # Aux variables
    label_len = len(label)
    k = max_length - label_len # Missing length
    neuron_unique_tuple, neuron_common_tuple, neuron_coverable_tuple, neuron_sum_tuple, common_space_extras_tuple, unique_space_extras_tuple = neuron_quantities
    common_space_extras, _ = common_space_extras_tuple
    unique_space_extras, _ = unique_space_extras_tuple
    neuron_coverable, _ = neuron_coverable_tuple
    neuron_common, _ = neuron_common_tuple
    neuron_unique, _ = neuron_unique_tuple
    tot_size = max_size_mask * len(neuron_coverable)
    # Unpack improvement information
    improv_quantities, cum_sums = max_improvement

    #(sample_cumsum_unique_intersection, _), (sample_cumsum_unique_extras, _) = cum_sums
    improv_common_intersection, improv_unique_intersection, improv_common_extras, improv_unique_extras, _, _ = improv_quantities

    # Sum quantities
    max_common_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='common_intersection', quantity_type='max', quantity_scope='sum')
    max_unique_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='unique_intersection', quantity_type='max', quantity_scope='sum')
    min_common_extras_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='common_extras', quantity_type='min', quantity_scope='sum')
    # max_unique_extras_sum = heuristic_utils.get_quantity(
    #     concepts_quantities=concept_quantities, quantity_name='unique_extras', quantity_type='max', quantity_scope='sum')
    min_unique_extras_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='unique_extras', quantity_type='min', quantity_scope='sum')
    min_common_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='common_intersection', quantity_type='min', quantity_scope='sum')
    max_unique_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='unique_intersection', quantity_type='max', quantity_scope='sum')
    min_unique_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='unique_intersection', quantity_type='min', quantity_scope='sum')

    # Samples quantities
    max_common_intersection = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='common_intersection', quantity_type='max', quantity_scope='sample')
    min_common_intersection = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='common_intersection', quantity_type='min', quantity_scope='sample')
    max_unique_intersection = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='unique_intersection', quantity_type='max', quantity_scope='sample')
    min_unique_intersection = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='unique_intersection', quantity_type='min', quantity_scope='sample')
    max_common_extras = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='common_extras', quantity_type='max', quantity_scope='sample')
    min_common_extras = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='common_extras', quantity_type='min', quantity_scope='sample')
    max_unique_extras = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='unique_extras', quantity_type='max', quantity_scope='sample')
    min_unique_extras = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='unique_extras', quantity_type='min', quantity_scope='sample')

    neuron_sum = neuron_sum_tuple[0]

    # We discard the labels that cannot increase the IoU
    if max_common_intersection_sum + max_unique_intersection_sum == 0:
        return (0.0, 0.0), (0.0, 0.0)

    # Max IoU

    top_k_common_intersection = improv_common_intersection[k][TOP_INDEX_SAMPLE]
    top_k_unique_intersection = improv_unique_intersection[k][TOP_INDEX_SAMPLE]
    # if isinstance(label, F.Leaf) or isinstance(label, F.Or):
    #     last_val = label.get_vals()[0]
    #     previous_index = last_val - 1
    #     if previous_index >= 0:
    #         top_k_unique_intersection  = np.minimum(top_k_unique_intersection, neuron_coverable - sample_cumsum_unique_intersection[previous_index])
    #max_intersection = np.minimum(max_common_intersection + max_unique_intersection + top_k_intersection, neuron_coverable).sum()
    max_intersection = (
        np.minimum(max_common_intersection + top_k_common_intersection, neuron_common) +
        np.minimum(max_unique_intersection + top_k_unique_intersection, neuron_unique)
       ).sum()

    
    bottom_1_extras_sum = improv_common_extras[0][BOTTOM_INDEX_SUM] + improv_unique_extras[0][BOTTOM_INDEX_SUM]
    if bottom_1_extras_sum > 0:
        bottom_1_extras_common = improv_common_extras[0][BOTTOM_INDEX_SAMPLE] 
        bottom_1_extras_unique = improv_unique_extras[0][BOTTOM_INDEX_SAMPLE]
        min_label_extras = min_common_extras + min_unique_extras
        # min_added_extras = np.clip(bottom_1_extras_common - max_common_extras, a_min=0, a_max=None) + \
        #                np.clip(bottom_1_extras_unique - max_unique_extras, a_min=0, a_max=None)
        min_extras = np.maximum(min_label_extras, bottom_1_extras_common + bottom_1_extras_unique)
        min_union = np.clip(neuron_sum + min_extras, a_min=0, a_max=max_size_mask).sum()
    else:
        min_union = min(num_hits + min_common_extras_sum + min_unique_extras_sum, tot_size)

    bottom_1_intersection_sum = improv_common_intersection[0][BOTTOM_INDEX_SUM] + improv_unique_intersection[0][BOTTOM_INDEX_SUM]
    if bottom_1_intersection_sum > 0:
        bottom_1_intersection = improv_common_intersection[0][BOTTOM_INDEX_SAMPLE] + improv_unique_intersection[0][BOTTOM_INDEX_SAMPLE]
        min_intersection = np.maximum(min_common_intersection + min_unique_intersection, bottom_1_intersection).sum()
    else:
        min_intersection = min_common_intersection_sum + min_unique_intersection_sum
    #min_union = np.clip(neuron_sum + min_label_extras, a_min=0, a_max=max_size_mask).sum()
    #max_label_extras = max_common_extras + max_unique_extras
    #min_label_extras = min_common_extras + min_unique_extras
    #bottom_1_extras = improv_common_extras[0][BOTTOM_INDEX_SAMPLE] + improv_unique_extras[0][BOTTOM_INDEX_SAMPLE]
   
   
   # bottom_1_extras_common = improv_common_extras[0][BOTTOM_INDEX_SAMPLE] 
   # bottom_1_extras_unique = improv_unique_extras[0][BOTTOM_INDEX_SAMPLE]

    
    #min_added_extras = np.clip(bottom_1_extras - max_label_extras, a_min=0, a_max=None)
   # min_added_extras = np.clip(bottom_1_extras_common - max_common_extras, a_min=0, a_max=None) + \
   #                     np.clip(bottom_1_extras_unique - max_unique_extras, a_min=0, a_max=None)

    #min_union = np.clip(neuron_sum + min_label_extras, a_min=0, a_max=max_size_mask).sum()
    #min_union = min(num_hits + min_common_extras_sum + min_unique_extras_sum, tot_size)

    #bottom_1_intersection = improv_common_intersection[0][BOTTOM_INDEX_SAMPLE] + improv_unique_intersection[0][BOTTOM_INDEX_SAMPLE]
    #min_intersection = np.maximum(min_common_intersection + min_unique_intersection, bottom_1_intersection).sum()
    #min_intersection = min_common_intersection_sum + min_unique_intersection_sum
    
    #top_k_extras = improv_common_extras[k][TOP_INDEX_SAMPLE] + improv_unique_extras[k][TOP_INDEX_SAMPLE]
    top_k_common_extras = improv_common_extras[k][TOP_INDEX_SAMPLE]
    top_k_unique_extras = improv_unique_extras[k][TOP_INDEX_SAMPLE]
    max_union = np.clip(neuron_sum + np.minimum(max_common_extras + top_k_common_extras, common_space_extras) + np.minimum(max_unique_extras + top_k_unique_extras, unique_space_extras), a_min=0, a_max=max_size_mask).sum()


    return (max_intersection, min_intersection), (max_union, min_union)



def and_chain_estimation(label, concept_quantities, neuron_quantities, max_improvement, *, num_hits, max_size_mask, max_length, minimum_threshold=0):
    # Aux variables
    # Unpack improvement information
    improv_quantities, cum_sum_unique_extras = max_improvement
    improv_common_intersection, improv_unique_intersection, improv_common_extras, improv_unique_extras, _, _ = improv_quantities
    neuron_unique_tuple, neuron_common_tuple, neuron_coverable_tuple, neuron_sum_tuple, common_space_extras_tuple, _  = neuron_quantities
    # Sum quantities
    max_common_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='common_intersection', quantity_type='max', quantity_scope='sum')

    # Samples quantities
    max_common_intersection = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='common_intersection', quantity_type='max', quantity_scope='sample')
    max_common_extras = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='common_extras', quantity_type='max', quantity_scope='sample')
    neuron_sum = neuron_sum_tuple[0]

    # We discard the labels that cannot increase the IoU
    if max_common_intersection_sum == 0:
        return (0.0, 0.0), (0.0, 0.0)

    top_1_common_intersection = improv_common_intersection[0][TOP_INDEX_SAMPLE]
    top_1_common_extras = improv_common_extras[0][TOP_INDEX_SAMPLE]

    # MaxIoU

    max_intersection = np.minimum(max_common_intersection, top_1_common_intersection).sum()
    min_union = num_hits
    max_union = num_hits + np.minimum(max_common_extras, top_1_common_extras).sum()
    return (max_intersection, 0.0), (max_union, min_union)
   

def and_not_chain_estimation(label, concept_quantities, neuron_quantities, max_improvement, *, num_hits, max_size_mask, max_length, minimum_threshold=0):
    # Aux variables
    # Unpack improvement information
    improv_quantities, cum_sums = max_improvement
    improv_common_intersection, improv_unique_intersection, improv_common_extras, improv_unique_extras, improv_common_uncovered, improv_unique_uncovered = improv_quantities
    neuron_unique_tuple, neuron_common_tuple, neuron_coverable_tuple, neuron_sum_tuple, common_space_extras_tuple, _  = neuron_quantities
    _, neuron_common_sum = neuron_common_tuple
    #(sample_cumsum_unique_intersection, _), (sample_cumsum_unique_extras, _) = cum_sums
    # Unpack max and min quantities
    # max_common_intersection_tuple, min_common_intersection_tuple = heuristic_utils.get_max_min_quantity(common_intersection)
    # max_unique_intersection_tuple, min_unique_intersection_tuple = heuristic_utils.get_max_min_quantity(unique_intersection)
    # max_common_extras_tuple, min_common_extras_tuple = heuristic_utils.get_max_min_quantity(common_extras)
    # max_unique_extras_tuple, min_unique_extras_tuple = heuristic_utils.get_max_min_quantity(unique_extras)
    
    # Sum quantities
    max_common_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='common_intersection', quantity_type='max', quantity_scope='sum')
    max_unique_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='unique_intersection', quantity_type='max', quantity_scope='sum')
    min_unique_intersection_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='unique_intersection', quantity_type='min', quantity_scope='sum')
    min_unique_extras_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='unique_extras', quantity_type='min', quantity_scope='sum')
    max_common_extras_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='common_extras', quantity_type='max', quantity_scope='sum')
    max_unique_extras_sum = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='unique_extras', quantity_type='max', quantity_scope='sum')

    # Samples quantities
    max_common_intersection = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='common_intersection', quantity_type='max', quantity_scope='sample')
    max_unique_intersection = heuristic_utils.get_quantity(
        concepts_quantities=concept_quantities, quantity_name='unique_intersection', quantity_type='max', quantity_scope='sample')
    # max_common_extras = heuristic_utils.get_quantity(
    #     concepts_quantities=concept_quantities, quantity_name='common_extras', quantity_type='max', quantity_scope='sample')
    # max_unique_extras = heuristic_utils.get_quantity(
    #     concepts_quantities=concept_quantities, quantity_name='unique_extras', quantity_type='max', quantity_scope='sample')
    #min_common_extras = heuristic_utils.get_quantity(

    # We discard the labels that cannot increase the IoU
    if max_common_intersection_sum + max_unique_intersection_sum == 0:
        return (0.0, 0.0), (0.0, 0.0)

    #top_1_uncovered = improv_common_uncovered[0][TOP_INDEX_SAMPLE] + improv_unique_uncovered[0][TOP_INDEX_SAMPLE]
    top_1_uncovered_common = improv_common_uncovered[0][TOP_INDEX_SAMPLE] 
    #top_1_common_extras = improv_common_extras[0][TOP_INDEX_SAMPLE]
    #top_1_unique_extras = improv_unique_extras[0][TOP_INDEX_SAMPLE]
    # if isinstance(label, F.Leaf) or (isinstance(label, F.And) and isinstance(label.right, F.Not)):
    #     last_val = label.get_vals()[0]
    #     previous_index = last_val - 1
    #     if previous_index > 0:
    #         top_1_unique_extras  = space_extras_uniques - sample_cumsum_unique_extras[previous_index]
    # Max IoU

    top_1_uncovered_common_sum = improv_common_uncovered[0][TOP_INDEX_SUM]


    if top_1_uncovered_common_sum < neuron_common_sum:
        max_intersection = (max_unique_intersection + np.minimum(
            max_common_intersection, top_1_uncovered_common
        )).sum()
    else:
        max_intersection = max_unique_intersection_sum + max_common_intersection_sum
    min_union = num_hits + min_unique_extras_sum
    # Min IoU
    min_intersection = min_unique_intersection_sum

    #TODO AGGIUNGI STESSA COSA PER PER MAX INTERSECTION
    max_union = num_hits + max_unique_extras_sum + max_common_extras_sum
    return (max_intersection, min_intersection), (max_union, min_union)



