"""Defines MassProperties class."""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from copy import deepcopy
from mass_properties_tools.check_symmetric_poi import check_symmetric_poi
from mass_properties_tools.check_triangle_inequality import check_triangle_inequality


class MassProperties:
    """
    MassProperties class contains mass properties information
    for a rigid body and provides methods for combining or 
    transforming.
    Inertia tensor is at the center of mass.
    """
    def __init__(
            self, 
            m: float=1.0,
            i: np.ndarray=np.identity(3),
            poi_positive: bool=False
        ):
        
        # mass:
        self.m = m    
        
        # inertia tensor (3x3 matrix)
        self._i = i
        
        # perform validity checks:
        # symmetric poi:
        if not check_symmetric_poi(i):
            raise ValueError("products of inertia are not symmetric")
        # triangle inequality:
        if not check_triangle_inequality(i):
            raise ValueError("inertia tensor is invalid (triangle inequality)")
        
        # tensor inverse:
        self._i_inv = np.zeros((3, 3))
        self._inverse_calculated = False
        # self.set_tensor_inverse()
        
        # products of inertia sign:
        self.poi_positive = poi_positive

    @property
    def i(self) -> np.ndarray:
        """
        inertia tensor
        """
        return self._i

    @property
    def i_xx(self) -> float:
        """
        moment of inertia at i[0,0]
        """
        return self.i[0][0]

    @property
    def i_yy(self) -> float:
        """
        moment of inertia at i[1,1]
        """
        return self.i[1][1]

    @property
    def i_zz(self) -> float:
        """
        moment of inertia at i[2,2]
        """
        return self.i[2][2]

    @property
    def i_xz(self) -> float:
        """
        product of inertia at i[0,2] = i[2,0]
        """
        return self.i[0][2]

    @property
    def i_zx(self) -> float:
        """
        product of inertia at i[2,0] = i[0,2]
        """
        return self.i[2][0]

    @property
    def i_xy(self) -> float:
        """
        product of inertia at i[0,1] = i[1,0]
        """
        return self.i[0][1]

    @property
    def i_yx(self) -> float:
        """
        product of inertia at i[1,0] = i[0,1]
        """
        return self.i[1][0]

    @property
    def i_yz(self) -> float:
        """
        product of inertia at i[1,2] = i[2,1]
        """
        return self.i[1][2]

    @property
    def i_zy(self) -> float:
        """
        product of inertia at i[2,1] = i[1,2]
        """
        return self.i[2][1]


    @property
    def i_inv(self) -> np.ndarray:
        """
        inverse of the inertia tensor (3x3 matrix)
        """
        if self._inverse_calculated:
            return self._i_inv
        else:
            self.set_tensor_inverse()
            return self._i_inv
    
    def set_tensor_inverse(self) -> None:
        """
        set the inverse of the inertia tensor (3x3 matrix)
        pre-calculate to avoid repeating the inverse calculation
        """
        self._i_inv = np.linalg.inv(self.i)
        self._inverse_calculated = True
        
    def copy(self) -> MassProperties:
        """Return a deep copy (not a shallow reference) of self."""
        return deepcopy(self)        
    
    def invert_products_of_inertia(self, inplace=False):
        """Invert the signs of the products of inertia."""
        if inplace:
            self.i[0,1] = -self.i[0,1]
            self.i[0,2] = -self.i[0,2]
            self.i[1,2] = -self.i[1,2]
            self.i[1,0] = -self.i[1,0]
            self.i[2,0] = -self.i[2,0]
            self.i[2,1] = -self.i[2,1]
            self.poi_positive = not self.poi_positive
            self._inverse_calculated = False
            return self
        else:
            mp_copy = self.copy()
            mp_copy.invert_products_of_inertia(inplace=True)
            return mp_copy
    
    def radius_of_gyration(self) -> np.ndarray:
        """Returns radii of gyration about each axis.
        k = sqrt(i / m)
        """
        kx = np.sqrt(self.i_xx / self.m)
        ky = np.sqrt(self.i_yy / self.m)
        kz = np.sqrt(self.i_zz / self.m)
        return np.array([kx, ky, kz])
    
    def to_dict(self):
        """Return dictionary with mass properties information."""
        return {
            'mass': self.m,
            'i_xx': self.i[0,0],
            'i_yy': self.i[1,1],
            'i_zz': self.i[2,2],
            'i_xy': self.i[0,1],
            'i_yz': self.i[1,2],
            'i_zx': self.i[2,0],
            'i_yx': self.i[1,0],
            'i_zy': self.i[2,1],
            'i_xz': self.i[0,2],
            'poi_positive': self.poi_positive,
        }
    
    def to_json(self):
        """Return a json object with mass properties information."""
        return json.dumps(self.to_dict())
    
    def write_to_json(self, filename: str or Path):
        """Save json data to a file."""
        with open(filename, "w") as f:
            f.write(self.to_json())
    
    @classmethod
    def from_json(cls, mp_json_file: str or Path): 
        """Create a MassProperties object from a json file."""
        mp_json = json.load(open(mp_json_file))
        # TODO: finish...
        return cls()
    
    @classmethod
    def from_params(
            cls, 
            mass: float, 
            i_xx: float, 
            i_yy: float, 
            i_zz: float, 
            i_xy: float=0.0, 
            i_yz: float=0.0, 
            i_zx: float=0.0, 
            poi_positive: bool=False,
        ):
        """
        
        """
        # TODO: finish...
        return cls()
    
    def __str__(self):
        return "\n".join([
            "\nMassProperties:",
            f"m = {self.m:1.6f}",
            f"i = \n{self.i}",
            f"poi_positive? = {self.poi_positive}",
        ])     


def main() -> None:
    
    # metric mass unit: [kg]
    # metric inertia unit: [kg-m^2]
    
    import sympy
    from scipy.spatial.transform import Rotation as R
    from mass_properties_tools.rotate_inertia_tensor import rotate_inertia_tensor
    from mass_properties_tools.diagonalize_inertia_tensor import diagonalize, diagonalize2, diagonalize3

    ixx, iyy, izz = sympy.symbols('ixx, iyy, izz')
    ixy, iyz, izx = sympy.symbols('ixy, iyz, izx')
    iyx, izy, ixz = sympy.symbols('iyx, izy, ixz')
    
    i = np.array([
        [ixx, ixy, ixz],
        [iyx, iyy, iyz],
        [izx, izy, izz],
    ])
    
    print(f"ixx = {i[0][0]}")
    print(f"iyy = {i[1][1]}")
    print(f"izz = {i[2][2]}")
    print(f"ixy = {i[0][1]}")
    print(f"iyz = {i[1][2]}")
    print(f"izx = {i[2][0]}")
    print(f"iyx = {i[1][0]}")
    print(f"izy = {i[2][1]}")
    print(f"ixz = {i[0][2]}")

    mp1 = MassProperties()
    print(mp1)
    
    mp2 = MassProperties(
        m=2.0,
        i=2.0*np.identity(3),
    )

    mp3 = mp2.copy()

    theta = np.pi / 8.0
    print(f"theta = {theta}")
    
    R_AtoB = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    print(f"R_AtoB = \n{R_AtoB}")

    I_A = np.array([
        [1.5, 0.0, 0.0],
        [0.0, 1.1, 0.0],
        [0.0, 0.0, 0.8],
    ])
    print(f"I_A = \n{I_A}")

    # print(np.dot(np.dot(R_AtoB, I_A), np.transpose(R_AtoB)))
    # print(np.dot(np.dot(np.transpose(R_AtoB), I_A), R_AtoB))

    i4 = rotate_inertia_tensor(I_A, R_AtoB)
    mp4 = MassProperties(
        m=4.0,
        i=i4,
    )
    print(mp4)
    
    mp5 = mp4.invert_products_of_inertia()
    print(mp5)

    # mp5 = diagonalize2(mp1.i)
    # print(mp5)

    # mp6 = mp1.diagonalize(inplace=False)
    # print(f"\nmp6 = \n{mp6}\n")

    # eig_vals, eig_vecs = diagonalize(mp1.i)
    # print(eig_vals, eig_vecs)


if __name__ == "__main__":
    main()
    