from compositional import formula as F

import unittest


def get_intersection(left_formula, right_formula):
    left_vals = left_formula.get_leaves()
    right_vals = right_formula.get_leaves()
    intersection = set(left_vals) & set(right_vals)
    return intersection

def combine_disjoint_formulas(left_formula, right_formula, op):
    if get_intersection(left_formula, right_formula):
        # case where there is an intersection, we cannot combine the formulas
        return []
    if op == 'AND':
        spanned_formulas = [F.And(left_formula, right_formula)]
    elif op == 'OR':
        spanned_formulas = [F.Or(left_formula, right_formula)]
    else:
        return []
    # elif op == 'NOT':
    #     spanned_formulas = [F.And(left_formula, F.Not(right_formula)), F.And(right_formula, F.Not(left_formula))]
    return spanned_formulas


def apply_distributivity(left_formula, right_formula, op):
    intersection = get_intersection(left_formula, right_formula)
    if len(intersection) == 0:
        # No intersection. This case is managed by combine disjoint
        return None
    
     # Chain using the same common operator
    opposite_op = F.Or if op == F.And else F.And
    leaves = left_formula.get_leaves() + right_formula.get_leaves()
    leaves = sorted(set(leaves), key=lambda x: x.val) #put
    common_terms = [term for term in leaves if term in intersection]
    distributed_formula_left = common_terms[0]
    for term in common_terms[1:]:
        distributed_formula_left = op(distributed_formula_left, term)
    different_terms = [term for term in leaves if term not in intersection]
    distributed_formula_right = different_terms[0]
    for term in different_terms[1:]:
        distributed_formula_right = opposite_op(distributed_formula_right, term)
    distributed_formula = op(distributed_formula_left, distributed_formula_right)
    return distributed_formula

def combine_same_exclusive_op(left_formula, right_formula):
    left_op_list = list(set(left_formula.get_ops()))
    right_op_list = list(set(right_formula.get_ops()))
    if len(left_op_list) != len(right_op_list) or len(left_op_list) != 1:
        # Different number of operators or not exactly one operator, cannot combine
        return None
    left_op = left_op_list[0]
    right_op = right_op_list[0]
    if left_op != right_op:
        # Different operators, cannot combine
        return None
    if left_op == 'NOT':
        # NOT operator, cannot combine
        return None
    
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
        return None
    
    # Chain using the same common operator
    leaves = left_leaves + right_leaves
    vals = sorted(set(leaves), key=lambda x: x.val) # Sort the values to ensure deterministic output
    combined_formula = vals[0] # Note that we have at least one leaf since the leaf nodes would be discarded by the first condition in this function
    for term in vals[1:]:
        combined_formula = op(combined_formula, term)

    # Chain using the opposite operator. Distributivity
    distributed_formula = apply_distributivity(left_formula, right_formula, op)
    return [combined_formula, distributed_formula]

def combine_different_op(left_formula, right_formula):
    left_op_list = list(set(left_formula.get_ops()))
    right_op_list = list(set(right_formula.get_ops()))
    if len(left_op_list) != 1 or len(right_op_list) != 1:
        # Not exactly one operator, cannot combine
        return None
    left_op = left_op_list[0]
    right_op = right_op_list[0]
    if left_op == right_op:
        # Same operators, cannot combine with this function
        return None
    if 'NOT' in (left_op, right_op):
        # NOT operator, cannot combine
        return None
    
    left_leaves = left_formula.get_leaves()
    right_leaves = right_formula.get_leaves()
    common_terms = set(left_leaves) & set(right_leaves)
    if len(common_terms) == 0:
        # No intersection. This case is managed by combine disjoint
        return None
    
    # Here, given (a OR b) and (a AND c) I would like to obtain ((a AND c) OR b) by replacing a with its specialization in the first formula
    or_leaves = left_leaves if left_op == 'OR' else right_leaves
    combined_formula = left_formula if left_op == 'AND' else right_formula # The end formula will be always full preserved
    for term in or_leaves:
        # We aggregate all the disjoint terms in or
        if term not in common_terms:
            combined_formula = F.Or(combined_formula, term)
    return combined_formula 



def combine_beam(left_formula, right_formula):
    intersection = get_intersection(left_formula, right_formula)
    if len(intersection) == 0:
        # Merge formulas in OR
        merged_formula = F.Or(left_formula, right_formula)
        return merged_formula
    else:
        return None


class TestUnify(unittest.TestCase):

    def test_intersection(self):
        a = F.Leaf(1)
        b = F.Leaf(2)
        c = F.Leaf(3)
        d = F.Leaf(4)

        # AND
        f1 = F.And(a, b)
        f2 = F.And(b, c)
        f3 = F.And(c, d)

        # AND NOT
        e = F.Not(a)
        f4 = F.And(e, d) # (Not a) and d
        f5 = F.And(e, c) # (Not a) and c

        # OR
        f6 = F.Or(a, b)
        f7 = F.Or(b, c)
        f8 = F.Or(c, d)

        # Test AND
        self.assertEqual(get_intersection(f1, f2), {b})
        self.assertEqual(get_intersection(f1, f3), set())

        # Test AND NOT
        self.assertEqual(get_intersection(f1, f4), set())
        self.assertEqual(get_intersection(f4, f5), set([F.Not(a)]))

        # Test OR
        self.assertEqual(get_intersection(f6, f7), {b})
        self.assertEqual(get_intersection(f6, f8), set())
        

    def test_combine(self):
        # OR
        a = F.Or(F.Leaf(1), F.Leaf(2))
        b = F.Or(F.Leaf(2), F.Leaf(3))
        c = F.Or(F.Leaf(3), F.Leaf(4))
        #  f1 = (a OR b OR c) f2 = (a AND c AND d)
        a_large = F.Or(F.Or(F.Leaf(1), F.Leaf(2)), F.Leaf(3))
        b_large = F.And(F.And(F.Leaf(1), F.Leaf(3)), F.Leaf(4))

        # AND
        d = F.And(F.Leaf(1), F.Leaf(2))
        e = F.And(F.Leaf(2), F.Leaf(3))
        f = F.And(F.Leaf(3), F.Leaf(4))

        # NOT
        g = F.And(F.Leaf(1), F.Not(F.Leaf(2)))

        # Test OR Disjoint
        self.assertListEqual(combine_disjoint_formulas(a, c, 'OR'), [F.Or(a, c)])

        # Test AND Disjoint
        self.assertListEqual(combine_disjoint_formulas(a, c, 'AND'), [F.And(a, c)])
        
        # Test NOT Disjoint
        #self.assertListEqual(combine_disjoint_formulas(a, c, 'NOT'), [F.And(a, F.Not(c)), F.And(c, F.Not(a))])
        self.assertListEqual(combine_disjoint_formulas(a, c, 'NOT'), [])

        # Test void
        self.assertListEqual(combine_disjoint_formulas(a, b, 'OR'), [])
        self.assertListEqual(combine_disjoint_formulas(a, b, 'AND'), [])
        self.assertListEqual(combine_disjoint_formulas(a, b, 'NOT'), [])

        # Test combine same exclusive op
        self.assertEqual(combine_same_exclusive_op(a, b), [F.Or(F.Or(F.Leaf(1), F.Leaf(2)), F.Leaf(3)), F.Or(F.Leaf(2), F.And(F.Leaf(1), F.Leaf(3)))])
        self.assertEqual(combine_same_exclusive_op(d, e), [F.And(F.And(F.Leaf(1), F.Leaf(2)), F.Leaf(3)), F.And(F.Leaf(2), F.Or(F.Leaf(1), F.Leaf(3)))])
        self.assertEqual(combine_same_exclusive_op(a, d), None) # This is managed by disjoint
        self.assertEqual(combine_same_exclusive_op(a, c), None) # This is managed by disjoin
        self.assertEqual(combine_same_exclusive_op(g, b), None) # NOT operator cannot be combined

        # Test combine different op
        self.assertEqual(combine_different_op(a, e), F.Or(F.And(F.Leaf(2), F.Leaf(3)), F.Leaf(1)))
        self.assertEqual(combine_different_op(a_large, b_large), F.Or(b_large, F.Leaf(2)))


if __name__ == '__main__':
    unittest.main()

