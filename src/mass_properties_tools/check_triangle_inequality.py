"""Defines check_triangle_inequality function."""
import numpy as np
from mass_properties_tools.diagonalize_inertia_tensor import diagonalize3


def check_triangle_inequality(i: np.ndarray, tol: float=1.0e-9) -> bool:
    """Check for realistic inertias by evaluating the triangle inequality.
    
    Not passing this implies existance of negative mass or negative density.
    
    The check is only applicable to a diagonalized inertia tensor, so this diagonalizes before checking.
    
    Args:
        i (np.ndarray): Inertia tensor.
        tol (float): Error tolerance.
    
    Returns:
        bool: Does the inertia tensor pass triangle inequality check?
    """
    valid = True
    
    i_diag, eig_vec = diagonalize3(i, tol=tol)
    
    i_xx = i_diag[0, 0]
    i_yy = i_diag[1, 1]
    i_zz = i_diag[2, 2]
    
    if i_xx + i_yy < i_zz:
        valid = False
    if i_xx + i_zz < i_yy: 
        valid = False
    if i_yy + i_zz < i_xx: 
        valid = False
        
    if valid:    
        print("passed triangle inequality check")
    if not valid:
        print("failed triangle inequality")
    
    return valid


def main() -> None:
    i1 = np.identity(3)
    print(check_triangle_inequality(i1))
    
    i2 = np.zeros((3,3))
    print(check_triangle_inequality(i2))


if __name__ == "__main__":
    main()
    