from compositional import formula as F


def split_compounds(formula):
    if isinstance(formula, F.Leaf) or isinstance(formula, F.Not) or isinstance(formula, F.And):
        return [formula]
    else:
        left_compounds = split_compounds(formula.left)
        right_compounds = split_compounds(formula.right)
        return left_compounds + right_compounds

            
def get_last_compound(formula):
    if isinstance(formula, F.Leaf) or isinstance(formula, F.Not) or isinstance(formula, F.And):
        return formula
    else:
        return get_last_compound(formula.right)



def next(formula, to_add, operator):
    if operator == "NOT":
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


def rewrite_in_compound_form(formula):
    if isinstance(formula, F.Leaf) or isinstance(formula, F.Not) or isinstance(formula, F.And):
        return formula
    else:
        compounds = split_compounds(formula)
        # Chain the compounds together using OR

    
a = F.Leaf(1)
b = F.Leaf(2)
c = F.Leaf(3)
d = F.Leaf(4)
e = F.Leaf(5)

f1 = F.Or(a, b)
f2 = F.And(c, d)
f3 = F.Or(f1, c)
f4 = F.And(f1, c)
f5 = F.Or(f4, e)
f6 = F.Or(f1, f2)
f7 = F.And (e, d)
f8 = F.Or(f6, f7)
f9 = F.And(f6, f7)

assert split_compounds(f1) == [a, b]
assert split_compounds(f2) == [f2]
assert split_compounds(f3) == [a, b, c]
assert split_compounds(f4) == [f4]
assert split_compounds(f5) == [f4, e]
assert split_compounds(f6) == [a, b, f2]
assert split_compounds(f7) == [f7]
assert split_compounds(f8) == [a, b, f2, f7]
assert split_compounds(f9) == [f9]

print(f1)
print("Compounds in f1:", split_compounds(f1))
print("Last compound in f1:", get_last_compound(f1))
print(f2)
print("Compounds in f2:", split_compounds(f2))
print("Last compound in f2:", get_last_compound(f2))
print(f3)
print("Compounds in f3:", split_compounds(f3))
print("Last compound in f3:", get_last_compound(f3))
print(f4)
print("Compounds in f4:", split_compounds(f4))
print("Last compound in f4:", get_last_compound(f4))
print(f5)
print("Compounds in f5:", split_compounds(f5))
print("Last compound in f5:", get_last_compound(f5))

print(f6)
print("Compounds in f6:", split_compounds(f6))
print("Last compound in f6:", get_last_compound(f6))
print(f7)
print("Compounds in f7:", split_compounds(f7))
print("Last compound in f7:", get_last_compound(f7))
print(f8)
print("Compounds in f8:", split_compounds(f8))
print("Last compound in f8:", get_last_compound(f8))
print(f9)
print("Compounds in f9:", split_compounds(f9))
print("Last compound in f9:", get_last_compound(f9))



to_add = F.Leaf(6)
operator_and = 'AND'
operator_or = 'OR'
print(f"F1: {f1}")
for new_formula in next(f1, to_add, operator_and):
    print(f"New formula: {new_formula}")
for new_formula in next(f1, to_add, operator_or):
    print(f"New formula: {new_formula}")
print(f"F2: {f2}")
for new_formula in next(f2, to_add, operator_and):
    print(f"New formula: {new_formula}")
for new_formula in next(f2, to_add, operator_or):
    print(f"New formula: {new_formula}")
print(f"F3: {f3}")
for new_formula in next(f3, to_add, operator_and):
    print(f"New formula: {new_formula}")
for new_formula in next(f3, to_add, operator_or):
    print(f"New formula: {new_formula}")
print(f"F4: {f4}")
for new_formula in next(f4, to_add, operator_and):
    print(f"New formula: {new_formula}")
for new_formula in next(f4, to_add, operator_or):
    print(f"New formula: {new_formula}")
print(f"F5: {f5}")
for new_formula in next(f5, to_add, operator_and):
    print(f"New formula: {new_formula}")
for new_formula in next(f5, to_add, operator_or):
    print(f"New formula: {new_formula}")
print(f"F6: {f6}")
for new_formula in next(f6, to_add, operator_and):
    print(f"New formula: {new_formula}")
for new_formula in next(f6, to_add, operator_or):
    print(f"New formula: {new_formula}")
print(f"F7: {f7}")
for new_formula in next(f7, to_add, operator_and):
    print(f"New formula: {new_formula}")
for new_formula in next(f7, to_add, operator_or):
    print(f"New formula: {new_formula}")
print(f"F8: {f8}")
for new_formula in next(f8, to_add, operator_and):
    print(f"New formula: {new_formula}")
for new_formula in next(f8, to_add, operator_or):
    print(f"New formula: {new_formula}")
print(f"F9: {f9}")
for new_formula in next(f9, to_add, operator_and):
    print(f"New formula: {new_formula}")
for new_formula in next(f9, to_add, operator_or):
    print(f"New formula: {new_formula}")


# def merge(f1, f2):
#     vals_f1 = set(f1.get_vals())
#     vals_f2 = set(f2.get_vals())
#     intersection = vals_f1.intersection(vals_f2)
#     if len(intersection) > 0:
#         if isinstance(f1, F.And) and isinstance(f2, F.And):
#             if len(f1) == 2 and len(f2) == 2:
#                 # Apply distributivity


#         print(f"Formulas {f1} and {f2} can be merged because they share the following values: {intersection}")

def unify(*, fixed_term, term_to_attach):
    ### Given two formulas, if they do not share any value, we can merge them by attaching one to the other.
    vals_fixed = set(fixed_term.get_vals())
    vals_to_attach = set(term_to_attach.get_vals())
    intersection = vals_fixed.intersection(vals_to_attach)
    if len(intersection) == 0:
        return None
    else:
        # Merge fixed term (a AND b) and term to attach (a AND C) to (a AND (b OR C))
        print(f"Formulas {fixed_term} and {term_to_attach} can be merged because they share the following values: {intersection}")
        if isinstance(fixed_term, F.And) and isinstance(term_to_attach, F.And):
            # The second condition ensures that we do not repeat terms

            if fixed_term.left == term_to_attach.left:
                common_term = fixed_term.left
                different_terms = (fixed_term.right, term_to_attach.right)
            elif fixed_term.left == term_to_attach.right:
                common_term = fixed_term.left
                different_terms = (fixed_term.right, term_to_attach.left)
            elif fixed_term.right == term_to_attach.left:
                common_term = fixed_term.right
                different_terms = (fixed_term.left, term_to_attach.right)
            elif fixed_term.right == term_to_attach.right:
                common_term = fixed_term.right
                different_terms = (fixed_term.left, term_to_attach.left)        
            if len(common_term) == len(intersection):
                new_formula = F.And(common_term, F.Or(different_terms[0], different_terms[1]))
                print(f"Formulas {fixed_term} and {term_to_attach} can be merged because they share the common term {common_term}. New formula after merging: {new_formula}")
                return new_formula

        # elif isinstance(fixed_term, F.Or) and isinstance(term_to_attach, F.Or):
        #     if fixed_term.left == term_to_attach.left and len(fixed_term.left) == len(intersection):
        #         new_formula = F.Or(fixed_term.left, F.Or(fixed_term.right, term_to_attach.right))
        #         print(f"Formulas {fixed_term} and {term_to_attach} can be merged because they share the common term {fixed_term.left}. New formula after merging: {new_formula}")
        #         return new_formula
        #     elif fixed_term.left == term_to_attach.right and len(fixed_term.left) == len(intersection):
        #         new_formula = F.Or(fixed_term.left, F.Or(fixed_term.right, term_to_attach.left))
        #         print(f"Formulas {fixed_term} and {term_to_attach} can be merged because they share the common term {fixed_term.left}. New formula after merging: {new_formula}")
        #         return new_formula
        #     if fixed_term.right == term_to_attach.left and len(fixed_term.right) == len(intersection):
        #         new_formula = F.Or(fixed_term.right, F.Or(fixed_term.left, term_to_attach.right))
        #         print(f"Formulas {fixed_term} and {term_to_attach} can be merged because they share the common term {fixed_term.right}. New formula after merging: {new_formula}")
        #         return new_formula
        #     elif fixed_term.right == term_to_attach.right and len(fixed_term.right) == len(intersection):
        #         new_formula = F.Or(fixed_term.right, F.Or(fixed_term.left, term_to_attach.left))
        #         print(f"Formulas {fixed_term} and {term_to_attach} can be merged because they share the common term {fixed_term.right}. New formula after merging: {new_formula}")
        #         return new_formula


def merge_beam(*, fixed_term, term_to_attach):
    ### Given two formulas, if they do not share any value, we can merge them by attaching one to the other.
    vals_fixed = set(fixed_term.get_vals())
    vals_to_attach = set(term_to_attach.get_vals())
    intersection = vals_fixed.intersection(vals_to_attach)
    if len(intersection) == 0:
        print(f"Formulas {fixed_term} and {term_to_attach} can be merged because they do not share any value")
        if isinstance(term_to_attach, F.And):
            new_formula = F.Or(fixed_term, term_to_attach)
            print(f"New formula after merging: {new_formula}")
            return new_formula
        # elif isinstance(term_to_attach, F.Or):
        #     new_formula = F.And(fixed_term, term_to_attach)
        #     print(f"New formula after merging: {new_formula}")
        #     return new_formula
    else:
        return None
        # Merge fixed term (a AND b) and term to attach (a AND C) to (a AND (b OR C))
        print(f"Formulas {fixed_term} and {term_to_attach} can be merged because they share the following values: {intersection}")
        # for common_term in intersection:
        #     common_term = F.Leaf(common_term)
        #     if isinstance(fixed_term, F.And) and isinstance(term_to_attach, F.And):
        #         if fixed_term.left == common_term:
        #             if fixed_term.left == term_to_attach.left:
        #                 new_formula = F.And(common_term, F.Or(fixed_term.right, term_to_attach.right))
        #                 print(f"Formulas {fixed_term} and {term_to_attach} can be merged because they share the common term {common_term}. New formula after merging: {new_formula}")
        #                 return new_formula
        #             elif fixed_term.left == term_to_attach.right:
        #                 new_formula = F.And(common_term, F.Or(fixed_term.right, term_to_attach.left))
        #                 print(f"Formulas {fixed_term} and {term_to_attach} can be merged because they share the common term {common_term}. New formula after merging: {new_formula}")
        #                 return new_formula
        #         elif fixed_term.right == common_term:
        #             if fixed_term.right == term_to_attach.left:
        #                 new_formula = F.And(common_term, F.Or(fixed_term.left, term_to_attach.right))
        #                 print(f"Formulas {fixed_term} and {term_to_attach} can be merged because they share the common term {common_term}. New formula after merging: {new_formula}")
        #                 return new_formula
        #             elif fixed_term.right == term_to_attach.right:
        #                 new_formula = F.And(common_term, F.Or(fixed_term.left, term_to_attach.left))
        #                 print(f"Formulas {fixed_term} and {term_to_attach} can be merged because they share the common term {common_term}. New formula after merging: {new_formula}")
        #                 return new_formula
        #         for common_term in intersection:



        # if isinstance(fixed_term, F.And) and isinstance(term_to_attach, F.And):
        #     # The second condition ensures that we do not repeat terms
        #     if fixed_term.left == term_to_attach.left and len(fixed_term.left) == len(intersection):
        #         new_formula = F.And(fixed_term.left, F.Or(fixed_term.right, term_to_attach.right))
        #         print(f"Formulas {fixed_term} and {term_to_attach} can be merged because they share the common term {fixed_term.left}. New formula after merging: {new_formula}")
        #         return new_formula
        #     elif fixed_term.left == term_to_attach.right and len(fixed_term.left) == len(intersection):
        #         new_formula = F.And(fixed_term.left, F.Or(fixed_term.right, term_to_attach.left))
        #         print(f"Formulas {fixed_term} and {term_to_attach} can be merged because they share the common term {fixed_term.left}. New formula after merging: {new_formula}")
        #         return new_formula
        #     if fixed_term.right == term_to_attach.left and len(fixed_term.right) == len(intersection):
        #         new_formula = F.And(fixed_term.right, F.Or(fixed_term.left, term_to_attach.right))
        #         print(f"Formulas {fixed_term} and {term_to_attach} can be merged because they share the common term {fixed_term.right}. New formula after merging: {new_formula}")
        #         return new_formula
        #     elif fixed_term.right == term_to_attach.right and len(fixed_term.right) == len(intersection):
        #         new_formula = F.And(fixed_term.right, F.Or(fixed_term.left, term_to_attach.left))
        #         print(f"Formulas {fixed_term} and {term_to_attach} can be merged because they share the common term {fixed_term.right}. New formula after merging: {new_formula}")
        #         return new_formula
        # elif isinstance(fixed_term, F.Or) and isinstance(term_to_attach, F.Or):
        #     if fixed_term.left == term_to_attach.left and len(fixed_term.left) == len(intersection):
        #         new_formula = F.Or(fixed_term.left, F.Or(fixed_term.right, term_to_attach.right))
        #         print(f"Formulas {fixed_term} and {term_to_attach} can be merged because they share the common term {fixed_term.left}. New formula after merging: {new_formula}")
        #         return new_formula
        #     elif fixed_term.left == term_to_attach.right and len(fixed_term.left) == len(intersection):
        #         new_formula = F.Or(fixed_term.left, F.Or(fixed_term.right, term_to_attach.left))
        #         print(f"Formulas {fixed_term} and {term_to_attach} can be merged because they share the common term {fixed_term.left}. New formula after merging: {new_formula}")
        #         return new_formula
        #     if fixed_term.right == term_to_attach.left and len(fixed_term.right) == len(intersection):
        #         new_formula = F.Or(fixed_term.right, F.Or(fixed_term.left, term_to_attach.right))
        #         print(f"Formulas {fixed_term} and {term_to_attach} can be merged because they share the common term {fixed_term.right}. New formula after merging: {new_formula}")
        #         return new_formula
        #     elif fixed_term.right == term_to_attach.right and len(fixed_term.right) == len(intersection):
        #         new_formula = F.Or(fixed_term.right, F.Or(fixed_term.left, term_to_attach.left))
        #         print(f"Formulas {fixed_term} and {term_to_attach} can be merged because they share the common term {fixed_term.right}. New formula after merging: {new_formula}")
        #         return new_formula
        if isinstance(fixed_term, F.And) and isinstance(term_to_attach, F.And):
            # The second condition ensures that we do not repeat terms

            if fixed_term.left == term_to_attach.left:
                common_term = fixed_term.left
                different_terms = (fixed_term.right, term_to_attach.right)
            elif fixed_term.left == term_to_attach.right:
                common_term = fixed_term.left
                different_terms = (fixed_term.right, term_to_attach.left)
            elif fixed_term.right == term_to_attach.left:
                common_term = fixed_term.right
                different_terms = (fixed_term.left, term_to_attach.right)
            elif fixed_term.right == term_to_attach.right:
                common_term = fixed_term.right
                different_terms = (fixed_term.left, term_to_attach.left)        
            if len(common_term) == len(intersection):
                new_formula = F.And(common_term, F.Or(different_terms[0], different_terms[1]))
                print(f"Formulas {fixed_term} and {term_to_attach} can be merged because they share the common term {common_term}. New formula after merging: {new_formula}")
                return new_formula

            # if fixed_term.left == term_to_attach.left and len(fixed_term.left) == len(intersection):
            #     new_formula = F.And(fixed_term.left, F.Or(fixed_term.right, term_to_attach.right))
            #     print(f"Formulas {fixed_term} and {term_to_attach} can be merged because they share the common term {fixed_term.left}. New formula after merging: {new_formula}")
            #     return new_formula
            # elif fixed_term.left == term_to_attach.right and len(fixed_term.left) == len(intersection):
            #     new_formula = F.And(fixed_term.left, F.Or(fixed_term.right, term_to_attach.left))
            #     print(f"Formulas {fixed_term} and {term_to_attach} can be merged because they share the common term {fixed_term.left}. New formula after merging: {new_formula}")
            #     return new_formula
            # if fixed_term.right == term_to_attach.left and len(fixed_term.right) == len(intersection):
            #     new_formula = F.And(fixed_term.right, F.Or(fixed_term.left, term_to_attach.right))
            #     print(f"Formulas {fixed_term} and {term_to_attach} can be merged because they share the common term {fixed_term.right}. New formula after merging: {new_formula}")
            #     return new_formula
            # elif fixed_term.right == term_to_attach.right and len(fixed_term.right) == len(intersection):
            #     new_formula = F.And(fixed_term.right, F.Or(fixed_term.left, term_to_attach.left))
            #     print(f"Formulas {fixed_term} and {term_to_attach} can be merged because they share the common term {fixed_term.right}. New formula after merging: {new_formula}")
            #     return new_formula
        elif isinstance(fixed_term, F.Or) and isinstance(term_to_attach, F.Or):
            if fixed_term.left == term_to_attach.left and len(fixed_term.left) == len(intersection):
                new_formula = F.Or(fixed_term.left, F.Or(fixed_term.right, term_to_attach.right))
                print(f"Formulas {fixed_term} and {term_to_attach} can be merged because they share the common term {fixed_term.left}. New formula after merging: {new_formula}")
                return new_formula
            elif fixed_term.left == term_to_attach.right and len(fixed_term.left) == len(intersection):
                new_formula = F.Or(fixed_term.left, F.Or(fixed_term.right, term_to_attach.left))
                print(f"Formulas {fixed_term} and {term_to_attach} can be merged because they share the common term {fixed_term.left}. New formula after merging: {new_formula}")
                return new_formula
            if fixed_term.right == term_to_attach.left and len(fixed_term.right) == len(intersection):
                new_formula = F.Or(fixed_term.right, F.Or(fixed_term.left, term_to_attach.right))
                print(f"Formulas {fixed_term} and {term_to_attach} can be merged because they share the common term {fixed_term.right}. New formula after merging: {new_formula}")
                return new_formula
            elif fixed_term.right == term_to_attach.right and len(fixed_term.right) == len(intersection):
                new_formula = F.Or(fixed_term.right, F.Or(fixed_term.left, term_to_attach.left))
                print(f"Formulas {fixed_term} and {term_to_attach} can be merged because they share the common term {fixed_term.right}. New formula after merging: {new_formula}")
                return new_formula

# Merge compounds
f10 = F.And(a, b)
f11 = F.And(a, c)
f12 = F.And(d, e)
print(f10)
print(f11)
print(f12)
assert merge_beam(fixed_term=f11, term_to_attach=f12) == F.Or(f11, f12)
f13 = F.Or(a,b)
f14 = F.Or(c,d)
print(f13)
print(f14)
assert merge_beam(fixed_term=f13, term_to_attach=f14) is None
f15 = merge_beam(fixed_term=f10, term_to_attach=f11)
assert f15 == F.And(a, F.Or(b, c))
f16 = F.And(a,e)
assert merge_beam(fixed_term=f15, term_to_attach=f16) == F.And(a, F.Or(F.Or(b, c), e))
f17 = F.Or(f10, f12)
f18 = F.Or(f10, c)
assert merge_beam(fixed_term=f17, term_to_attach=f18) == F.Or(f10, F.Or(f12, c))
f19 = F.Or(a, b)
f20 = F.Or(a, c)
f21 = F.Or(d, e)
f22 = F.And(f19, f21)
f23 = F.And(f20, f21)
assert merge_beam(fixed_term=f22, term_to_attach=f23) == None
