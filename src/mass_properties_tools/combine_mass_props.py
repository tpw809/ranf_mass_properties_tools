import numpy as np


def combine_mass_props(
        m1: float, 
        m2: float, 
        cm1: np.ndarray, 
        cm2: np.ndarray, 
        i1: np.ndarray, 
        i2: np.ndarray,
    ):
    """
    find new mass, center of mass, inertia for combined rigid bodies
    must be defined in the same reference frame
    inertias defined about body center of mass
    
    Args:
        m1 (float): mass of body 1
        m2 (float): mass of body 2
        cm1 (np.ndarray): center of mass of body 1
        cm2 (np.ndarray): center of mass of body 2
        i1 (np.ndarray): inertia tensor of body 1
        i2 (np.ndarray): inertia tensor of body 2
    """
    
    # combined mass:
    m3 = m1 + m2
    
    # combined center of mass location:
    cm3 = np.array([0.0, 0.0, 0.0])
    cm3[0] = (cm1[0] * m1 + cm2[0] * m2) / m3
    cm3[1] = (cm1[1] * m1 + cm2[1] * m2) / m3
    cm3[2] = (cm1[2] * m1 + cm2[2] * m2) / m3
    
    # combined inertias:
    dcm1 = cm1 - cm3
    dcm2 = cm2 - cm3
    i3 = np.zeros((3, 3))

    # parallel axis theorem:
    i3 += i1 + m1 * (np.dot(dcm1, dcm1) * np.identity(3) - np.outer(dcm1, dcm1))
    i3 += i2 + m2 * (np.dot(dcm2, dcm2) * np.identity(3) - np.outer(dcm2, dcm2))
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
    
    # combined mass properties:
    m3, cm3, i3 = combine_mass_props(
        m1=m1,
        m2=m2, 
        cm1=cm1,
        cm2=cm2,
        i1=i1,
        i2=i2)
    
    print(f"m3 = {m3}")
    print(f"cm3 = {cm3}")
    print(f"i3 = \n{i3}")


if __name__ == "__main__":
    main()
    