import numpy as np
from scipy.spatial.transform import Rotation as R
from mass_properties_tools import primitives
from mass_properties_tools import rotate_inertia_tensor
from mass_properties_tools import diagonalize3
from mass_properties_tools import Frame
from mass_properties_tools import MassProperties
from mass_properties_tools import MassiveBody




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

