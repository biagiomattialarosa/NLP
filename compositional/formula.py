"""
This module is an adaptation of the formula module used
in the excellent repository https://github.com/jayelm/compexp .
It contains functions to represent a formula as a tree, to compute the hash
value of a formula and to order formulas.
"""

from functools import total_ordering
from typing import List


class F:
    """
    Abstract class to represent a formula.
    """
    def get_atoms(self):
        pass
    
    def tree_path(self):
        """
        Function to return the path of the tree.
        """
        raise NotImplementedError(
            "This method should be implemented in the subclasses."
        )
    pass

def compute_hash_value(f: F):
    if isinstance(f, Leaf):
        return ("LEAF", f.val)

    if isinstance(f, Not):
        return ("NOT", compute_hash_value(f.val))

    if isinstance(f, Or):
        return ("OR", tuple(sorted(_flatten_same_op(f, Or))))

    if isinstance(f, And):
        return ("AND", tuple(sorted(_flatten_same_op(f, And))))

    raise TypeError(f"Unsupported formula type: {type(f)}")


def _flatten_same_op(f: F, op_type):
    """
    Flatten associative chains of the same operator.

    Example:
    (1 OR (2 OR 3)) and ((1 OR 2) OR 3)
    become the same canonical tuple.
    """
    if isinstance(f, op_type):
        return (
            _flatten_same_op(f.left, op_type)
            + _flatten_same_op(f.right, op_type)
        )

    return (compute_hash_value(f),)

# def compute_hash_value(formula: F) -> float:
#     """
#     Function to compute the hash value of a given formula.
#     This function is used to avoid computation of equivalent formulas,
#     thus, two equivalent formulas will have the same hash value.
#     """
#     if isinstance(formula, Leaf):
#         return round(formula.val / 2000, 4)
#     elif isinstance(formula, Not):
#         return round(1 - compute_hash_value(formula.val), 4)
#     elif isinstance(formula, Or):
#         if len(set(formula.get_ops())) == 1:
#             values = sorted(formula.get_vals())
#             hash_sum = 0
#             prod = 1
#             for v in values:
#                 leaf_value = compute_hash_value(Leaf(v))
#                 hash_sum += leaf_value
#                 prod *= leaf_value
#             return round(hash_sum - prod, 4)   
#         else:
#             return round(
#                 compute_hash_value(formula.left)
#                 + compute_hash_value(formula.right)
#                 - (
#                     (
#                         compute_hash_value(formula.right)
#                         * compute_hash_value(formula.left)
#                     )
#                 ),
#                 4,
#             )
#     elif isinstance(formula, And):
#         if len(set(formula.get_ops())) == 1:
#             values = sorted(formula.get_vals())
#             hash_product = 1
#             for v in values:
#                 leaf_value = compute_hash_value(Leaf(v))
#                 hash_product *= leaf_value                
#             return round(hash_product, 4)
#         else:
#             return round(
#                 compute_hash_value(
#                     formula.left)
#                 * compute_hash_value(
#                     formula.right), 4
#             )
#     return None


class Leaf(F):
    """
    Class to represent an atomic concept.
    """
    def __init__(self, val):
        self.val = val
        # We added this term to order the formulas by IoU
        self.iou = None

    def __str__(self):
        return str(self.val)

    def to_str(self, namer):
        return namer(self.val)

    def __len__(self):
        return 1

    # we redefine the hash function
    def __hash__(self):
        return hash(compute_hash_value(self))

    def __repr__(self):
        return f"Leaf({str(self)})"

    def get_vals(self):
        return [self.val]
    
    def get_leaves(self):
        return [self]
    
    def get_atoms(self):
        atoms = [self]
        return atoms
    
    def tree_path(self):
        return [self]

    def is_leaf(self):
        return True

    def get_ops(self):
        return []
    
    def flip_atoms(self):
        return self

    # we redefine the equality function
    def __eq__(self, other):
        if not isinstance(other, Leaf):
            return False
        return self.val == other.val

    def __lt__(self, other):
        if isinstance(other, Leaf):
            return self.val < other.val
        else:
            return True

class Node(F):
    """
    Abstract class to represent a node in a formula.
    """
    def is_leaf(self):
        return False


class UnaryNode(Node):
    """ Class to represent a unary node. """
    arity = 1
    op = None

    def __init__(self, val):
        self.val = val
        # We added this term to order the formulas by IoU
        self.iou = 0

    def __str__(self):
        return f"({self.op} {self.val})"

    def to_str(self, namer):
        op_name = self.val.to_str(namer)
        return f"({self.op} {op_name})"

    def __len__(self):
        return len(self.val)

    def __repr__(self):
        return f"{self.op}({self.val})"

    def get_vals(self):
        return self.val.get_vals()
    
    def get_leaves(self):
        return self.val.get_leaves()
    
    def get_atoms(self):
        atoms = []
        if self.op == 'NOT':
            atoms.extend(self)
        return atoms
    
    def tree_path(self):
        return [self]


class BinaryNode(Node):
    """ Class to represent a binary node. """
    arity = 2
    op = None

    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right
        # We added this term to order the formulas by IoU
        self.iou = 0

    def __str__(self):
        return f"({self.left} {self.op} {self.right})"

    def to_str(self, namer, sort=False):
        left_name = self.left.to_str(namer, sort=sort)
        right_name = self.right.to_str(namer, sort=sort)
        if not sort or (left_name < right_name):
            return f"({left_name} {self.op} {right_name})"
        else:
            return f"({right_name} {self.op} {left_name})"

    def __len__(self):
        return len(self.left) + len(self.right)

    def __repr__(self):
        return f"{self.op}({self.left}, {self.right})"

    def get_vals(self):
        vals = []
        vals.extend(self.right.get_vals())
        vals.extend(self.left.get_vals())
        return vals
    
    def get_leaves(self):
        leaves = []
        leaves.extend(self.right.get_leaves())
        leaves.extend(self.left.get_leaves())
        return leaves
    
    def get_atoms(self):
        atoms = []
        if self.op == 'OR':
            atoms.extend(self.left.get_atoms())
            atoms.extend(self.right.get_atoms())
        elif self.op == 'AND':
            if isinstance(self.right, Not):
                atoms_left = self.left.get_atoms()
                for atom in atoms_left:
                    atoms.append(And(atom, self.right))
            else:
                atoms = [self]
        return atoms
    
    def flip_atoms(self):
        return BinaryNode(self.right, self.left)

    def get_ops(self):
        """ Function to return the operators in a formula. """
        vals = []
        vals.append(self.op)
        if isinstance(self.left, BinaryNode) or isinstance(
            self.left, UnaryNode
        ):
            vals.extend(self.left.get_ops())
        if isinstance(self.right, BinaryNode) or isinstance(
            self.right, UnaryNode
        ):
            vals.extend(self.right.get_ops())
        return vals

    def __eq__(self, other):
        raise NotImplementedError(
            "This method should be implemented in the subclasses."
        )
    
    def tree_path(self):
        return [self] + self.left.tree_path() + self.right.tree_path()
    


class Not(UnaryNode):
    """
    Class to represent the NOT operator.
    """
    op = "NOT"

    def __init__(self, val):
        super().__init__(val)
        self.op = "NOT"

    def __hash__(self):
        return hash(compute_hash_value(self))

    def __eq__(self, other):
        if not isinstance(other, Not):
            return False
        if isinstance(other, Not):
            return self.val == other.val

    def get_ops(self):
        """ Function to return the operators in a formula. """
        vals = []
        vals.append(self.op)
        assert isinstance(self.val, Leaf)
        return vals
    
    def get_leaves(self):
        return [self]
    
    def __lt__(self, other):
        if isinstance(other, Not):
            return self.val < other.val
        else:
            False


class Or(BinaryNode):
    """Class to represent the OR operator."""
    op = "OR"

    def __init__(self, left, right):
        super().__init__(left, right)
        self.op = "OR"

    def __hash__(self):
        return hash(compute_hash_value(self))
    
    def __eq__(self, other):
        if not isinstance(other, Or):
            return False
        
        # Case mono operator
        self_ops = self.get_ops()
        other_ops = other.get_ops()
        if len(set(self_ops)) == 1 and len(set(other_ops)) == 1:
            # In this case we can sort the values and check if they are equal
            self_vals = sorted(self.get_vals())
            other_vals = sorted(other.get_vals())
            if self_vals == other_vals:
                return True
            else:
                return False
        
        # Non trivial casaes
        if self.left == other.left and self.right == other.right:
            return True
        elif self.left == other.right and self.right == other.left:
            return True
        else:
            return False
    
    def flip_atoms(self):
        return Or(self.right, self.left)

    def __lt__(self, other):
        if isinstance(other, Or):
            return self.left < other.left or (self.left == other.left and self.right < other.right)
        elif isinstance(other, Leaf):
            return False
        else:
            return True

class And(BinaryNode):
    """Class to represent the AND operator."""
    op = "AND"

    def __init__(self, left, right):
        assert not isinstance(left, Not), "NOT terms must appear on the right side of AND"
        super().__init__(left, right)
        self.op = "AND"


    def __hash__(self):
        return hash(compute_hash_value(self))
    
    def flip_atoms(self):
        return And(self.right, self.left)
    
    def __lt__(self, other):
        if isinstance(other, And):
            return self.left < other.left or (self.left == other.left and self.right < other.right)
        elif isinstance(other, Leaf):
            return False
        elif isinstance(other, Or):
            return False
        else:
            return True
        
    def __eq__(self, other):
        if not isinstance(other, And):
            return False
        
        # Case mono operator
        self_ops = self.get_ops()
        other_ops = other.get_ops()
        if len(set(self_ops)) == 1 and len(set(other_ops)) == 1:
            # In this case we can sort the values and check if they are equal
            self_vals = sorted(self.get_vals())
            other_vals = sorted(other.get_vals())
            if self_vals == other_vals:
                return True
            else:
                return False

        # Non trivial casaes
        self_is_not = isinstance(self.right, Not)
        other_is_not = isinstance(other.right, Not)
        if self_is_not and other_is_not:
            # Both are not
            if self.left == other.left and self.right.val == other.right.val:
                return True
            else:
                return False
        if not self_is_not and not other_is_not:
            # Both are and
            if self.left == other.left and self.right == other.right:
                return True
            elif self.left == other.right and self.right == other.left:
                return True
            else:
                return False
        else:
            # Mixed case, one is not and the other is and
            return False
    



@total_ordering
class OrderedFormula:
    """
    Class to order formulas by their iou.
    """
    def __init__(self, formula) -> None:
        self.formula = formula
        self.iou = formula.iou

    def __eq__(self, __o: object) -> bool:
        return self.iou == __o.iou

    def __lt__(self, __o: object) -> bool:
        return self.iou < __o.iou
    
    def __repr__(self) -> str:
        return f"OrderedFormula(formula={self.formula}, iou={self.iou})"


def get_formula_str_atoms(f: F, namer_vec: List[str]):
    atoms = f.get_atoms()
    str_atoms = []
    for atom in atoms:
        str_atoms.append(get_formula_str(atom, namer_vec))
    return str_atoms


def get_formula_str(f: F, namer_vec: List[str]):
    """
    Function to get the string representation of a formula.

    Args:
        f: Formula to get the string representation of.
        namer_vec: List of names for the variables in the formula.

    Returns:
        String representation of the formula.
    """
    if isinstance(f, And):
        masks_l = get_formula_str(f.left, namer_vec)
        masks_r = get_formula_str(f.right, namer_vec)
        return f"({masks_l} AND {masks_r})"
    elif isinstance(f, Or):
        masks_l = get_formula_str(f.left, namer_vec)
        masks_r = get_formula_str(f.right, namer_vec)
        return f"({masks_l} OR {masks_r})"
    elif isinstance(f, Not):
        return f"NOT {get_formula_str( f.val, namer_vec)}"
    elif isinstance(f, Leaf):
        return namer_vec[f.val]
    elif isinstance(f, int):
        return namer_vec[f]
