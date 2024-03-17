import numpy as np
from mass_properties_tools.frame_class import Frame
from mass_properties_tools.mass_properties_class import MassProperties
from mass_properties_tools.massive_body_class import MassiveBody


def add_massive_bodies(
        mb1: MassiveBody, 
        mb2: MassiveBody, 
        reference_frame: Frame,
    ) -> MassiveBody:
    """Add MassiveBodys.
    Find new mass, center of mass, inertia for combined rigid bodies.
    Inertias defined about body center of mass.
    
    Args:
        mb1 (MassiveBody): MassiveBody 1 to add
        mb2 (MassiveBody): MassiveBody 2 to add
        reference_frame (Frame): reference frame for the resulting MassiveBody.
    
    Returns:
        mb3 (MassiveBody): Summed mass properties.
    """
    # check poi_positive (always work with negative poi):
    if mb1.mp.poi_positive:
        print("warning: invert poi sign for mb1...")
        # TODO: invert sign of poi
    if mb2.mp.poi_positive:
        print("warning: invert poi sign for mb2...")
        # TODO: invert sign of poi
    
    # combined mass:
    m1 = mb1.mp.m
    m2 = mb2.mp.m
    m3 = m1 + m2
    
    # combined center of mass location:
    cm1 = mb1.cm.position_global
    cm2 = mb2.cm.position_global
    cm3 = np.array([0.0, 0.0, 0.0])
    
    cm3[0] = (cm1[0] * m1 + cm2[0] * m2) / m3
    cm3[1] = (cm1[1] * m1 + cm2[1] * m2) / m3
    cm3[2] = (cm1[2] * m1 + cm2[2] * m2) / m3
    
    # combined inertias:
    dcm1 = cm1 - cm3
    dcm2 = cm2 - cm3
    
    i1 = mb1.mp.i
    i2 = mb2.mp.i
    i3 = np.zeros((3, 3))

    # parallel axis theorem:
    i3 += i1 + m1 * (np.dot(dcm1, dcm1) * np.identity(3) - np.outer(dcm1, dcm1))
    i3 += i2 + m2 * (np.dot(dcm2, dcm2) * np.identity(3) - np.outer(dcm2, dcm2))
    
    # create center of mass Frame:
    cm3_frame = Frame(
        name='cm3',
        position=cm3,
    )
    
    # create MassProperties:
    mp3 = MassProperties(
        m=m3,
        i=i3,
    )
    
    # create MassiveBody:
    mb3 = MassiveBody(
        cm=cm3_frame,
        mp=mp3,
    )
    
    # handle non-global reference frame:
    # TODO: transform frame to new reference frame:
    
    
    
    return mb3


def main() -> None:
    # body 1 mass properties:
    m1 = 1.0
    cm1 = np.array([0.0, 0.0, -1.0])
    i1 = np.identity(3)
    
    cm1_frame = Frame(
        position=cm1,
    )
    
    # body 2 mass properties:
    m2 = 1.0
    cm2 = np.array([0.0, 0.0, 1.0])
    i2 = np.identity(3)
    
    cm2_frame = Frame(
        position=cm2,
    )
    
    # combined mass properties:
    mb3 = add_massive_bodies(
        m1=m1,
        m2=m2, 
        cm1=cm1,
        cm2=cm2,
        i1=i1,
        i2=i2,
    )
    
    print(mb3)


if __name__ == "__main__":
    main()
    