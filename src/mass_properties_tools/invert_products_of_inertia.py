import numpy as np
from copy import deepcopy


def invert_products_of_inertia(i: np.ndarray) -> np.ndarray:
    """
    Args:
        i (np.ndarray): inertia tensor.
    
    Returns:
        np.ndarray: inertia tensor with products of inertia inverted.
    """
    i_inverted = deepcopy(i)
    i_inverted[0, 1] = -i[0, 1]
    i_inverted[0, 2] = -i[0, 2]
    i_inverted[1, 2] = -i[1, 2]
    i_inverted[1, 0] = -i[1, 0]
    i_inverted[2, 0] = -i[2, 0]
    i_inverted[2, 1] = -i[2, 1]
    # or i_inv = -i, then only invert the diagonals...
    return i_inverted


def main() -> None:
    pass


if __name__ == "__main__":
    main()
    