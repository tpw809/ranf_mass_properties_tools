"""Defines parallel axis theorem function.

Theory:
Single Axis Parallel Axis Theorem:
    I = transformed inertia to a new axis
    I_cm = inertia at center of mass
    m = mass
    d = distance from center of mass
    I = I_cm + m*d**2
    
Generalized Tensor Parallel Axis Theorem:
    J = I + m * [(R * R) * I3x3 - outer_product(R, R)]
    J = transformed inertia tensor about a new point
    m = mass
    I = inertia tensor about center of mass
    R = displacement vector from center of mass to new point
    I3x3 = 3x3 identity matrix
"""
import numpy as np


def parallel_axis_theorem(i: np.ndarray, r: np.ndarray, m: float) -> np.ndarray:
    """Parallel axis theorem to change reference point for an inertia tensor.
    
    Args:
        i (np.ndarray): Original inertia tensor at original point.
        r (np.ndarray): Displacement vector from original point to new point.
        m (float): Mass associated with the inertia.
    
    Returns:
        np.ndarray: Inertia tensor at new point.
    """
    i_trans = i + m * (np.dot(r, r) * np.identity(3) - np.outer(r, r))
    return i_trans


def main() -> None:
    from scipy import spatial
    from sympy import symbols

    I_cm = 2.0
    m = 5.0
    d = 3.0
    I = I_cm + m*d**2
    # print(I)


    def kronecker_delta(i, j):
        if i == j:
            return 1.0
        else:
            return 0.0


    I = np.array([
        [2.8, 0.0, 0.0],
        [0.0, 2.5, 0.0],
        [0.0, 0.0, 2.3],
    ])
    # print(I)

    T = spatial.transform.Rotation.random().as_matrix()
    print(f"T = \n{T}\n")

    It = np.dot(T, np.dot(I, np.linalg.inv(T)))
    print(f"It = \n{It}\n")

    R = np.array([1.0, 2.0, 3.0])
    J = It + m * (np.dot(R, R) * np.identity(3) - np.outer(R, R))
    print(f"J = \n{J}\n")

    J = parallel_axis_theorem(It, R, m)
    print(f"J = \n{J}\n")
    """
    R2 = -np.array([1.0, 2.0, 3.0])
    J2 = It + m * (np.dot(R2, R2) * np.identity(3) - np.outer(R2, R2))
    print(f"J2 = \n{J2}")

    J3 = np.zeros((3, 3))
    for i in range(0, 3):
        for j in range(0, 3):
            J3[i, j] = It[i, j] + m * \
                       (np.linalg.norm(R)**2 * kronecker_delta(i, j) - R[i]*R[j])

    print(f"J3 = \n{J3}")

    J4 = np.zeros((3, 3))
    for i in range(0, 3):
        for j in range(0, 3):
            J4[i, j] = It[i, j] + m * \
                       (np.linalg.norm(R2)**2 * kronecker_delta(i, j) - R2[i]*R2[j])

    print(f"J4 = \n{J4}")
    """

    # Try sympy:

    # inertia symbols:
    i_00, i_01, i_02, i_11, i_12, i_22 = symbols('i_00 i_01 i_02 i_11 i_12 i_22')

    # mass symbol:
    ms = symbols('m')

    # displacement symbols:
    rx, ry, rz = symbols('rx ry rz')

    # inertia tensor:
    i = np.array([
        [i_00, i_01, i_02],
        [i_01, i_11, i_12],
        [i_02, i_12, i_22],
    ])

    # displacement vector:
    r = np.array((rx, ry, rz))

    # transform using generalized parallel axis theorem:
    Js = i + ms * (np.dot(r, r) * np.identity(3) - np.outer(r, r))
    print(f"\nJs = \n{Js}\n")

    # used defined function:
    Js = parallel_axis_theorem(i, r, ms)
    print(f"\nJs = \n{Js}\n")

    # print each moment and product indvidually:
    for i in range(0, 3):
        for j in range(0, 3):
            print(f"j_{i}{j} = {Js[i, j]}")

    # implement sympy results equations:
    j00 = It[0, 0] + m * (R[1]**2 + R[2]**2)
    j11 = It[1, 1] + m * (R[0]**2 + R[2]**2)
    j22 = It[2, 2] + m * (R[0]**2 + R[1]**2)
    j01 = It[0, 1] - m * R[0] * R[1]
    j02 = It[0, 2] - m * R[0] * R[2]
    j12 = It[1, 2] - m * R[1] * R[2]

    print(f"\nj00 = {j00}")
    print(f"j11 = {j11}")
    print(f"j22 = {j22}")
    print(f"j01 = {j01}")
    print(f"j02 = {j02}")
    print(f"j12 = {j12}")

    # If I use positive products of inertia:
    Itp = It
    Itp[0, 1] = -It[0, 1]
    Itp[0, 2] = -It[0, 2]
    Itp[1, 2] = -It[1, 2]
    Itp[1, 0] = -It[1, 0]
    Itp[2, 0] = -It[2, 0]
    Itp[2, 1] = -It[2, 1]
    print(It)

    R = np.array([1.0, 2.0, 3.0])
    Jp = Itp + m * (np.dot(R, R) * np.identity(3) - np.outer(R, R))
    print(f"Jp = \n{Jp}\n")


if __name__ == "__main__":
    main()
    