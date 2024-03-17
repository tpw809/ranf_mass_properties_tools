import numpy as np
from scipy.spatial.transform import Rotation as R
from mass_properties_tools import primitives
from mass_properties_tools import Frame
from mass_properties_tools import MassProperties
from mass_properties_tools import MassiveBody


i_cyl = primitives.cylinder_inertia(m=3.0, r=1.0, lz=1.0)
print(f"\ni_cyl = \n{i_cyl}")

rot_i = R.from_euler(
    seq='XYZ',
    angles=[
        1.0 * np.pi / 10.0,
        1.0 * np.pi / 10.0,
        1.0 * np.pi / 10.0,
    ],
    degrees=False,
)
print(f"\nrot_i = \n{rot_i.as_matrix()}")

i_cyl_rot = rot_i.as_matrix() @ i_cyl @ rot_i.inv().as_matrix()
print(f"\ni_cyl_rot = \n{i_cyl_rot}")


frm0 = Frame(
    name='frm0',
    position=np.array([0,0,0]),
    rotation=R.from_quat([0,0,0,1]),
    reference_frame=None,
)
print(frm0)

rot1 = R.from_euler(
    seq='XYZ',
    angles=[
        0.0 * np.pi / 10.0,
        0.0 * np.pi / 10.0,
        0.0 * np.pi / 10.0,
    ],
    degrees=False,
)
print(f"\nrot1 = \n{rot1.as_matrix()}")

frm1 = Frame(
    name='frm1',
    position=np.array([1,1,1]),
    rotation=rot1,
    reference_frame=None,
)
print(frm1)

rot2 = R.from_euler(
    seq='XYZ',
    angles=[
        0.0 * np.pi / 10.0,
        0.0 * np.pi / 10.0,
        0.0 * np.pi / 10.0,
    ],
    degrees=False,
)
print(f"\nrot1 = \n{rot2.as_matrix()}")

frm2 = Frame(
    name='frm2',
    position=np.array([1,1,1]),
    rotation=rot2,
    reference_frame=frm1,
)
print(frm2)

mp_cyl = MassProperties(
    m=3.0,
    i=i_cyl,
)
# print(mp_cyl)

mb_cyl = MassiveBody(
    mp=mp_cyl,
    cm=frm1,
)
print(mb_cyl)

frm3 = frm1.transform_to_frame(frm2, new_name='frm3')
print(frm3)

mb_cyl.cm = mb_cyl.cm.transform_to_frame(frm2)
print(mb_cyl)
