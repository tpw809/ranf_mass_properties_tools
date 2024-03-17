import numpy as np


def cuboid_inertia(m: float, lx: float, ly: float, lz: float) -> np.ndarray:
    """
    cuboid is a solid rectangle
    solid implies uniform density
    
    Args:
        m (float): mass
        lx (float): length in the x direction
        ly (float): length in the y direction
        lz (float): length in the z direction
    
    Returns:
        np.ndarray: inertia tensor at center of mass
    """
    ixx = (m / 12.0) * (ly**2 + lz**2)
    iyy = (m / 12.0) * (lx**2 + lz**2)
    izz = (m / 12.0) * (ly**2 + lx**2)
    return np.array([
        [ixx, 0.0, 0.0], 
        [0.0, iyy, 0.0], 
        [0.0, 0.0, izz],
    ])


def sphere_inertia(m: float, r: float) -> np.ndarray:
    """
    solid sphere
    solid implies uniform density
    
    Args:
        m (float): mass
        r (float): radius
    
    Returns:
        np.ndarray: inertia tensor at center of mass
    """
    i = m * 2.0 / 5.0 * r**2
    return i * np.identity(3)


def hollow_sphere_inertia(m: float, r: float) -> np.ndarray:
    """
    hollow sphere has all mass concentrated
    to outer surface
    
    Args:
        m (float): mass
        r (float): radius
    
    Returns:
        np.ndarray: inertia tensor at center of mass
    """
    i = m * 2.0 / 3.0 * r**2
    return i * np.identity(3)


def ellipsoid_inertia(m: float, ax: float, by: float, cz: float) -> np.ndarray:
    """
    solid ellipsoid
    solid implies uniform density
    
    Args:
        m (float): mass
        ax (float): semi-axis along x
        by (float): semi-axis along y
        cz (float): semi-axis along z
    
    Returns:
        np.ndarray: inertia tensor at center of mass
    """
    ixx = m / 5.0 * (by**2 + cz**2)
    iyy = m / 5.0 * (ax**2 + cz**2)
    izz = m / 5.0 * (by**2 + ax**2)
    return np.array([
        [ixx, 0.0, 0.0], 
        [0.0, iyy, 0.0], 
        [0.0, 0.0, izz],
    ])


def cylinder_inertia(m: float, r: float, lz: float) -> np.ndarray:
    """
    solid cylinder
    solid implies uniform density
    z axis is axis of cylinder rotation
    
    Args:
        m (float): mass
        r (float): radius
        lz (float): length in the z direction
    
    Returns:
        np.ndarray: inertia tensor at center of mass
    """
    ixx = m / 12.0 * (3.0*r**2 + lz**2)
    izz = m / 2.0 * r**2
    return np.array([
        [ixx, 0.0, 0.0], 
        [0.0, ixx, 0.0], 
        [0.0, 0.0, izz],
    ])


def hollow_cylinder_inertia(m: float, ri: float, ro: float, lz: float) -> np.ndarray:
    """
    z axis is axis of cylinder rotation
    
    Args:
        m (float): mass
        ri (float): inner radius
        ro (float): outer radius
        lz (float): length in the z direction
    
    Returns:
        np.ndarray: inertia tensor at center of mass
    """
    ixx = m / 12.0 * (3.0*(ri**2 + ro**2) + lz**2)
    izz = m / 2.0 * (ri**2 + ro**2)
    return np.array([
        [ixx, 0.0, 0.0], 
        [0.0, ixx, 0.0], 
        [0.0, 0.0, izz],
    ])


def main() -> None:
    # solid cube:
    Icube = cuboid_inertia(m=1.0, lx=1.0, ly=1.0, lz=1.0)
    print(Icube)

    # solid cuboid:
    Icuboid = cuboid_inertia(m=1.0, lx=1.0, ly=2.0, lz=3.0)
    print(Icuboid)

    # solid sphere:
    Isphere = sphere_inertia(m=1.0, r=1.0)
    print(Isphere)

    # solid ellipsoid:
    Isphere2 = ellipsoid_inertia(m=1.0, ax=1.0, by=1.0, cz=1.0)
    print(Isphere2)

    # hollow sphere:
    Iholsphere = hollow_sphere_inertia(m=1.0, r=1.0)
    print(Iholsphere)

    # solid ellipsoid:
    Iellipsoid = ellipsoid_inertia(m=1.0, ax=1.0, by=2.0, cz=3.0)
    print(Iellipsoid)

    # solid cylinder:
    Icyl = cylinder_inertia(m=1.0, r=1.0, lz=1.0)
    print(Icyl)

    # hollow cylinder:
    Iholcyl = hollow_cylinder_inertia(m=1.0, ri=0.5, ro=1.0, lz=1.0)
    print(Iholcyl)


if __name__ == "__main__":
    main()
    