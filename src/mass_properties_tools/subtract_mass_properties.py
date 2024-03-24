"""Defines subtract_mass_properties function."""
import numpy as np


def subtract_mass_properties(
        m1: float, 
        m2: float, 
        cm1: np.ndarray, 
        cm2: np.ndarray, 
        i1: np.ndarray, 
        i2: np.ndarray,
    ):
    """Find new mass, center of mass, inertia for combined rigid bodies
    
    Must be defined in the same reference frame.
    
    Inertias defined about body center of mass for m1 - m2
    
    Args:
        m1 (float): mass of body 1
        m2 (float): mass of body 2
        cm1 (np.ndarray): center of mass of body 1
        cm2 (np.ndarray): center of mass of body 2
        i1 (np.ndarray): inertia tensor of body 1
        i2 (np.ndarray): inertia tensor of body 2
    
    Returns:
        
    """
    # combined mass:
    m3 = m1 - m2
    
    # combined center of mass location:
    cm3 = np.array([0.0, 0.0, 0.0])
    cm3[0] = (cm1[0] * m1 - cm2[0] * m2) / m3
    cm3[1] = (cm1[1] * m1 - cm2[1] * m2) / m3
    cm3[2] = (cm1[2] * m1 - cm2[2] * m2) / m3
    
    # combined inertias:
    dcm1 = cm1 - cm3
    dcm2 = cm2 - cm3
    i3 = np.zeros((3, 3))

    # parallel axis theorem:
    i3 += i1 + m1 * (np.dot(dcm1, dcm1) * np.identity(3) - np.outer(dcm1, dcm1))
    i3 -= i2 + m2 * (np.dot(dcm2, dcm2) * np.identity(3) - np.outer(dcm2, dcm2))
    return m3, cm3, i3


def main() -> None:
    # body 1 mass properties:
    m1 = 1.0
    cm1 = np.array([0.0, 0.0, -1.0])
    i1 = np.identity(3)
    
    # body 2 mass properties:
    m2 = 1.0
    cm2 = np.array([0.0, 0.0, 1.0])
    i2 = np.identity(3)
    
    # subtract mass properties:
    m4, cm4, i4 = subtract_mass_properties(
        m1=m3,
        m2=m2, 
        cm1=cm3,
        cm2=cm2,
        i1=i3,
        i2=i2)

    print(f"m4 = {m4}")
    print(f"cm4 = {cm4}")
    print(f"i4 = \n{i4}")


if __name__ == "__main__":
    main()
    