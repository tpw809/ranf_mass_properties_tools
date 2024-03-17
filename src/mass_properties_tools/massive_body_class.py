from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
from mass_properties_tools.diagonalize_inertia_tensor import diagonalize3
from mass_properties_tools.frame_class import Frame
from mass_properties_tools.mass_properties_class import MassProperties


class MassiveBody:
    """
    MassiveBody class contains mass properties and center of mass frame.
    """
    def __init__(
            self,
            mp: MassProperties,
            cm: Frame,
        ):
        # MassProperties:
        self.mp = mp
        
        # Center of Mass Frame:
        self.cm = cm
        
    @property
    def i_global(self):
        """
        inertia tensor rotated to the global frame
        I' = R@I@R^T
        """
        return self.cm.rotation_global.inv().as_matrix() @ self.mp.i @ self.cm.rotation_global.as_matrix()
    
    @property
    def i_inv_global(self):
        """
        inertia tensor inverse rotated to the global frame
        """
        return self.cm.rotation_global.inv().as_matrix() @ self.mp.i_inv @ self.cm.rotation_global.as_matrix()

    def change_cm_reference_frame(self, reference_frame: Frame) -> MassiveBody:
        """
        Change the center of mass frame reference frame.
        
        Args:
            reference_frame (Frame): new reference frame
        """
        pass
    
    def diagonalize(self, tol: float=1.0e-9) -> MassiveBody:
        """Diagonalize the inertia tensor and rotate center of mass frame to principal axes.
        
        Args:
            tol (float): error tolerance.
        """
        i_diag3, rm_diag3 = diagonalize3(self.mp.i, tol)
    
        # TODO: inplace or new object?
        # if inplace:
        #     self.mp.i = i_diag3
        #     self.cm.rotation = R.from_matrix(rm_diag3)
        # else:
        
        mp_diag = MassProperties(
            m=self.mp.m,
            i=i_diag3,
            poi_positive=self.mp.poi_positive,
        )
        
        rotation_diag = self.cm.rotation * R.from_matrix(rm_diag3)
        
        cm_diag = Frame(
            name=self.cm.name+'_diag',
            position=self.cm.position,
            rotation=rotation_diag,
            reference_frame=self.cm.reference_frame,
        )
        
        mb_diag = MassiveBody(
            cm=cm_diag,
            mp=mp_diag,
        )
        return mb_diag
        
    def copy(self) -> MassiveBody:
        """
        return a deep copy (not a shallow reference) of self
        """
        return deepcopy(self)

    def __str__(self):
        return "\n".join([
            "\nMassiveBody:",
            f"cm = {self.cm}",
            f"mp = {self.mp}",
        ])


def main() -> None:
    
    # metric mass unit: [kg]
    # metric inertia unit: [kg-m^2]
    
    frm0 = Frame(
        name='frm0',
        position=np.array([0, 0, 0]),
        rotation=R.from_quat([0,0,0,1]),
        reference_frame=None,
    )
    
    rot1 = R.from_euler(
        seq='xyz',
        angles=[
            1.0 * np.pi / 10.0,
            1.0 * np.pi / 10.0,
            1.0 * np.pi / 10.0,
        ],
        degrees=False,
    )
    
    frm1 = Frame(
        name='frm1',
        position=np.array([1, 1, 1]),
        rotation=rot1,
        reference_frame=None,
    )
    
    cm1 = Frame(
        name='cm1',
        reference_frame=frm1,
    )
    
    mp1 = MassProperties(
        m=0.0,
        i=np.zeros((3,3)),
    )
    
    mb1 = MassiveBody(
        mp=mp1,
        cm=cm1,
    )
    
    print(mb1)
    
    mb1_diag = mb1.diagonalize()
    print(mb1_diag)
    

if __name__ == "__main__":
    main()
    