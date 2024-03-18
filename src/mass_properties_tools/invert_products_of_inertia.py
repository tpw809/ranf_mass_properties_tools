import numpy as np
from copy import deepcopy


def invert_products_of_inertia(i: np.ndarray) -> np.ndarray:
    """Invert the sign of the products of inertia for an inertia tensor.
    
    Args:
        i (np.ndarray): Inertia tensor.
    
    Returns:
        np.ndarray: Inertia tensor with products of inertia sign inverted.
    """
    i_inverted = -deepcopy(i)
    i_inverted[0, 0] = i[0, 0]
    i_inverted[1, 1] = i[1, 1]
    i_inverted[2, 2] = i[2, 2]
    return i_inverted


def main() -> None:
    i = np.ones((3, 3))
    print(i)
    
    i_inverted = invert_products_of_inertia(i)
    print(i_inverted)
    
    i_inverted[0,0] = 2.0
    print(i)
    print(i_inverted)


if __name__ == "__main__":
    main()
    