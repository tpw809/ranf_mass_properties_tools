"""Defines function to check if inertia tensor products of inertia are symmetric.

Valid inertia tensors must be symmetric.
"""
import numpy as np


def check_symmetric_poi(i: np.ndarray, tol: float=1.0e-9) -> bool:
    """Check if products of inertia (poi) for an inertia tensor are symmetric.
    i[0,1] == i[1,0]
    i[0,2] == i[2,0]
    i[1,2] == i[2,1]
    
    Args:
        i (np.ndarray): Inertia tensor.
        tol (float): Tolerance for equality.
    
    Returns:
        bool: Are the products of inertia symmetric with the tolerance allowed?
    """
    symmetry_checks = [
        np.abs(i[0,1] - i[1,0]) > tol,
        np.abs(i[2,1] - i[1,2]) > tol,
        np.abs(i[0,2] - i[2,0]) > tol,
    ]
    if any(symmetry_checks):
        return False
    else: 
        return True


def main() -> None:
    i1 = np.identity(3)
    print(check_symmetric_poi(i1))
    
    i2 = np.zeros((3,3))
    print(check_symmetric_poi(i2))


if __name__ == "__main__":
    main()
    