import numpy as np


def combine_mass_properties(
        m1: float, 
        cm1: np.ndarray,
        i1: np.ndarray,
        m2: float,
        cm2: np.ndarray,
        i2: np.ndarray,
    ):
    """
    find new mass, center of mass, inertia for combined rigid bodies
    reference frame must be the same for both bodies
    product of inertia sign must be the same for both bodies
    """
    # check poi_positive:
    
    
    
    # combined mass:
    m3 = m1 + m2
    
    # combined center of mass location:
    cm3 = np.zeros(3)
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
    m1 = 1.0
    cm1 = np.array([0.0, 0.0, 0.0])
    i1 = np.identity(3)
    
    m2 = 1.0
    cm2 = np.array([0.0, 0.0, 0.0])
    i2 = np.identity(3)
    
    m3, cm3, i3 = combine_mass_properties(
        m1=m1,
        cm1=cm1,
        i1=i1,
        m2=m2,
        cm2=cm2,
        i2=i2,
    )
    
    print(m3)
    print(cm3)
    print(i3)
    

if __name__ == "__main__":
    main()
    