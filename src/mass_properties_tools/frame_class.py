"""Frame class definition."""
from __future__ import annotations
from typing import TYPE_CHECKING
import json
import numpy as np
from pathlib import Path
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from transform_tools.unit_normal_direction_vectors import x_hat, y_hat, z_hat
from transform_tools.get_relative_position import get_relative_position
from transform_tools.get_relative_rotation import get_relative_rotation

if TYPE_CHECKING:
    from scipy.spatial.transform import Rotation


# global is the inertially grounded (fixed) reference frame 
# local is my reference frame


class Frame:
    """
    Coordinate system defined by a position and rotation within another reference_frame.
    Local is with respect to the reference frame.
    Global is with respect to the global reference frame.
    """
    def __init__(
            self,
            name: str,
            position: np.ndarray=np.array([0.0, 0.0, 0.0]),
            rotation: Rotation=R.from_quat([0.0, 0.0, 0.0, 1.0]),
            reference_frame: Frame=None,
        ): 
        
        self.name = name
        
        # "local" reference frame:
        # if None, then reference frame is ground / world / map
        self.reference_frame = reference_frame
        
        # position [x, y, z] in reference_frame:
        self.position = position

        # scipy.spatial.transform.Rotation object:
        # rotation in local reference frame:
        self.rotation = rotation

    def __str__(self):
        if self.reference_frame is not None:
            reference_frame_name = self.reference_frame.name
        else:
            reference_frame_name = None
            
        return "\n".join((
            "\nFrame:",
            f"name = {self.name}",
            f"position = {self.position}",
            f"position_global = {self.position_global}",
            f"rotation_mat = \n{self.rotation.as_matrix()}",
            f"rotation_mat_global = \n{self.rotation_global.as_matrix()}",
            f"reference_frame = {reference_frame_name}\n",
        ))

    @property
    def position_global(self) -> np.ndarray:
        """
        position [x,y,z] in global reference frame
        expressed in global reference frame
        """
        # no reference frame means ground is reference_frame:
        if self.reference_frame is None:
            return self.position
        else:
            return np.add(
                self.reference_frame.position_global, 
                self.reference_frame.rotation_global.apply(self.position)
            )

    @position_global.setter
    def position_global(self, arg: np.ndarray):
        """
        position [x,y,z] in global reference frame
        """
        # no reference frame means ground is reference_frame:
        if self.reference_frame is None:
            self.position = arg
        else:
            self.position = get_relative_position(
                to_frame=Frame(name='', position=arg), 
                from_frame=self.reference_frame, 
                expressed_in_frame=self.reference_frame,
            )
    
    @property
    def rotation_global(self) -> Rotation:
        """
        scipy Rotation of this frame expressed in global frame
        """
        # no reference frame means ground is reference_frame:
        if self.reference_frame is None:
            return self.rotation
        else:
            return self.reference_frame.rotation_global * self.rotation

    @rotation_global.setter
    def rotation_global(self, arg: Rotation):
        """
        scipy Rotation of this frame expressed in global frame
        """
        # no reference frame means ground is reference_frame:
        if self.reference_frame is None:
            self.rotation = arg
        else:
            self.rotation = self.reference_frame.rotation_global.inv() * arg

    @property
    def x_hat(self) -> np.ndarray:
        """
        x-direction unit vector for this (rotated) frame
        expressed in the reference_frame (local)
        """
        return self.rotation.apply(np.array([1.0, 0.0, 0.0]))

    @property
    def y_hat(self) -> np.ndarray:
        """
        y-direction unit vector for this (rotated) frame
        expressed in the reference_frame (local)
        """
        return self.rotation.apply(np.array([0.0, 1.0, 0.0]))

    @property
    def z_hat(self) -> np.ndarray:
        """
        y-direction unit vector for this (rotated) frame
        expressed in the reference_frame (local)
        """
        return self.rotation.apply(np.array([0.0, 0.0, 1.0]))

    @property
    def x_hat_global(self) -> np.ndarray:
        """
        global x direction of this frame
        in global reference frame
        """
        if self.reference_frame is None:
            return self.x_hat
        else:
            return self.rotation_global.apply(np.array([1.0, 0.0, 0.0]))
    
    @property
    def y_hat_global(self) -> np.ndarray:
        """
        global y direction of this frame
        in global reference frame
        """
        if self.reference_frame is None:
            return self.y_hat
        else:
            return self.rotation_global.apply(np.array([0.0, 1.0, 0.0]))
    
    @property
    def z_hat_global(self) -> np.ndarray:
        """
        global z direction of this frame
        in global reference frame
        """
        if self.reference_frame is None:
            return self.z_hat
        else:
            return self.rotation_global.apply(np.array([0.0, 0.0, 1.0]))
    
    # @property
    # def rot_mat(self):
    #     """
    #     rotation matrix
    #     """
    #     return self.rotation.as_matrix()
    
    # @property
    # def q(self):
    #     """
    #     rotation quaternion
    #     [v, w]
    #     [x, y, z, w]
    #     """
    #     return self.rotation.as_quat()

    # @property
    # def rot_mat_global(self):
    #     """
    #     global rotation matrix
    #     """
    #     return self.rotation_global.as_matrix()
    
    # @property
    # def q_global(self):
    #     """
    #     global rotation quaternion
    #     [v, w]
    #     [x, y, z, w]
    #     """
    #     return self.rotation_global.as_quat()


    def set_pos_rel_to(
            self, 
            position: np.ndarray, 
            relative_to_frame: Frame,
        ):
        """
        set position based on position relative to rel_to_frame
        (not switching local reference frame, so expressed in frame = self.reference_frame)
        """
        # TODO: check this:
        ref_pos = get_relative_position(
            to_frame=relative_to_frame, 
            from_frame=self.reference_frame, 
            expressed_in_frame=self.reference_frame,
        )
        
        self.position = (self.reference_frame.rotation_global.inv() * relative_to_frame.rotation_global).apply(position) + ref_pos

    def set_rot_rel_to(
            self, 
            rotation: Rotation, 
            relative_to_frame: Frame,
        ):
        """
        set rotation based on rotation relative to rel_to_frame
        """
        self.rotation = self.reference_frame.rotation_global.inv() * relative_to_frame.rotation_global * rotation

    def transform_to_frame(
            self, 
            reference_frame: Frame, 
            inplace=False,
            new_name: str=None,
        ) -> Frame:
        """
        Create a new frame at the same global position and orientation 
        with frm as self, in the new reference_frame.
        """
        if new_name is None:
            new_name=self.name
        
        if inplace:
            self.name = new_name
            
            self.rotation = get_relative_rotation(
                to_frame=self, 
                from_frame=reference_frame,
            )
            
            self.position = get_relative_position(
                to_frame=self,
                from_frame=reference_frame, 
                expressed_in_frame=reference_frame,
            )
                
            self.reference_frame = reference_frame
        else:
            new_rotation = get_relative_rotation(
                to_frame=self, 
                from_frame=reference_frame,
            )
            
            new_position = get_relative_position(
                to_frame=self,
                from_frame=reference_frame, 
                expressed_in_frame=reference_frame,
            )
            
            return Frame(
                name=new_name,
                position=new_position,
                rotation=new_rotation,
                reference_frame=reference_frame,
            )
    
    def apply_rotation(
            self, 
            rotation: Rotation, 
            inplace: bool=True,
        ):
        """
        apply a rotation to this frame
        rot = rotation being applied
        inplace = change this object or return a new object?
        """
        # pre or post multiply?
        # TODO: test pre or post multiply???
        # new_rot = rot * self.rotation
        new_rot = self.rotation * rotation
        print(f"\nApplying a Rotation to Frame {self.name}:")
        print(f"Original rot: \n{self.rotation.as_matrix()}")
        print(f"Applied rot: \n{rotation.as_matrix()}")
        print(f"New rot: \n{new_rot.as_matrix()}")
        if inplace:
            self.rotation = new_rot
        else:
            frame_copy = self.copy()
            frame_copy.rotation = new_rot
            return frame_copy

    # def apply_translation(
    #         self, 
    #         vec: np.ndarray, 
    #         reference_frame: Frame=None, 
    #         inplace: bool=True,
    #     ):
    #     """
    #     apply a translation (change in position) to this frame
    #     
    #     Args:
    #         vec = translation vector (position displacement)
    #         reference_frame = expressed in frame for vec
    #         inplace = change this object or return a new object?
    #     """
    #     if reference_frame is None:
    #         reference_frame = self.reference_frame
    #     pass
    #     # TODO:
    #     # done in global or local frame?
    #     new_position_global = self.position_global + reference_frame.rotation_global.apply(vec) 
    #     print(f"\nApplying a Translation:")
    #     print(f"Original pos: \n{self.pos}")
    #     print(f"Applied pos: \n{vec}")
    #     # print(f"New pos: \n{new_pos}")
    #     if inplace:
    #         self.position_global = new_position_global
    #     else:
    #         frame_copy = self.copy()
    #         frame_copy.position_global = new_position_global
    #         return frame_copy

    # def apply_transform(self, transform):
    #     """
    #     apply a transform (translation & rotation)
    #     """
    #     pass
    #     # TODO:
    
    def plot3d(self, **kwargs):
        """
        plot frame to 3d axes
        """
        # origin of the frame:
        p0 = self.position_global
        # direction unit vectors:
        px = np.add(p0, self.x_hat_global)
        py = np.add(p0, self.y_hat_global)
        pz = np.add(p0, self.z_hat_global)
        # plot:
        plt.plot(p0[0], p0[1], **kwargs)
        plt.plot(
            [p0[0], px[0]], 
            [p0[1], px[1]], 
            **kwargs,
        )
        plt.text(x=px[0], y=px[1], s='x')
        plt.plot(
            [p0[0], py[0]], 
            [p0[1], py[1]], 
            linestyle='--', 
            **kwargs,
        )
        plt.text(x=py[0], y=py[1], s='y')
        plt.plot(
            [p0[0], pz[0]], 
            [p0[1], pz[1]], 
            linestyle=':', 
            **kwargs,
        )
        plt.text(x=pz[0], y=pz[1], s='z')

    def plot_xy(self, **kwargs):
        """
        plot frame as projection on x-y plane 
        """
        # get current figure:
        # plt.gcf()
        # origin of the frame:
        p0 = self.position_global
        # direction unit vectors:
        px = np.add(p0, self.x_hat_global)
        py = np.add(p0, self.y_hat_global)
        pz = np.add(p0, self.z_hat_global)
        # plot:
        plt.plot(p0[0], p0[1], **kwargs)
        plt.plot([p0[0], px[0]], [p0[1], px[1]], **kwargs)
        plt.text(x=px[0], y=px[1], s='x')
        plt.plot(
            [p0[0], py[0]], 
            [p0[1], py[1]], 
            linestyle='--', 
            **kwargs,
        )
        plt.text(x=py[0], y=py[1], s='y')
        plt.plot(
            [p0[0], pz[0]], 
            [p0[1], pz[1]], 
            linestyle=':', 
            **kwargs,
        )
        plt.text(x=pz[0], y=pz[1], s='z')

    def plot_yz(self, **kwargs):
        """
        plot frame as projection on y-z plane 
        """
        # get current figure:
        # plt.gcf()
        # origin of the frame:
        p0 = self.position_global
        # direction unit vectors:
        px = np.add(p0, self.x_hat_global)
        py = np.add(p0, self.y_hat_global)
        pz = np.add(p0, self.z_hat_global)
        # plot:
        plt.plot(p0[1], p0[2], **kwargs)
        plt.plot([p0[1], px[1]], [p0[2], px[2]], **kwargs)
        plt.text(x=px[1], y=px[2], s='x')
        plt.plot(
            [p0[1], py[1]], [p0[2], py[2]], linestyle='--', **kwargs)
        plt.text(x=py[1], y=py[2], s='y')
        plt.plot(
            [p0[1], pz[1]], 
            [p0[2], pz[2]], 
            linestyle=':', 
            **kwargs,
        )
        plt.text(x=pz[1], y=pz[2], s='z')

    def plot_xz(self, **kwargs):
        """
        plot frame as projection on x-z plane 
        """
        # get current figure:
        # plt.gcf()
        # origin of the frame:
        p0 = self.position_global
        # direction unit vectors:
        px = np.add(p0, self.x_hat_global)
        py = np.add(p0, self.y_hat_global)
        pz = np.add(p0, self.z_hat_global)
        # plot:
        plt.plot(p0[0], p0[2], **kwargs)
        plt.plot([p0[0], px[0]], [p0[2], px[2]], **kwargs)
        plt.text(x=px[0], y=px[2], s='x')
        plt.plot(
            [p0[0], py[0]], [p0[2], py[2]], 
            linestyle='--', 
            **kwargs)
        plt.text(x=py[0], y=py[2], s='y')
        plt.plot(
            [p0[0], pz[0]], [p0[2], pz[2]], 
            linestyle=':', 
            **kwargs)
        plt.text(x=pz[0], y=pz[2], s='z')

    def copy(self) -> Frame:
        """
        make a deep copy of self
        """
        return deepcopy(self)
    
    # def to_dict(self):
    #     """
    #     return a dictionary defining the reference frame
    #     """
    #     if self.reference_frame is not None:
    #         reference_frame_name = self.reference_frame.name
    #     else:
    #         reference_frame_name = None
    #         
    #     quaternion = self.rotation.as_quat()
    #     
    #     frame_dict = {
    #         'name': self.name,
    #         'reference_frame': self.reference_frame_name,
    #         # expressed_in_frame???
    #         'pos_x': self.position[0],
    #         'pos_y': self.position[1],
    #         'pos_z': self.position[2],
    #         'quat_a': quaternion[3],
    #         'quat_b': quaternion[0],
    #         'quat_c': quaternion[1],
    #         'quat_d': quaternion[2],
    #         # rotation matrix?
    #         # direction unit vectors?
    #     }
    #     return frame_dict

    # def to_json(self):
    #     """
    #     convert dictionary to a json object
    #     """
    #     return json.dumps(self.to_dict())
    
    # def write_to_json(self, filename: str or Path):
    #     """
    #     save info to a .json file
    #     """
    #     with open(filename, 'w') as f:
    #         f.write(self.to_json())

    # @classmethod
    # def from_json(cls, frame_json_file: str or Path):
    #     """
    #     create frame object from json file
    #     """
    #     frame_json = json.load(open(frame_json_file))
    #     
    #     position = np.array([
    #         frame_json['pos_x'],
    #         frame_json['pos_y'],
    #         frame_json['pos_z']
    #     ])
    #     
    #     rotation = R.from_quat([
    #         frame_json['quat_b'],
    #         frame_json['quat_c'],
    #         frame_json['quat_d'],
    #         frame_json['quat_a']
    #     ])
    #     
    #     frame_obj = cls(
    #         name=frame_json['name'],
    #         position=position,
    #         rotation=rotation,
    #         reference_frame=None,
    #         expressed_in_frame=None,
    #     )
    #         
    #     return frame_obj
        

def main() -> None:
    from transform_tools.get_distance_between_frames import get_distance_between_frames

    ground_frame = Frame(name='ground_frame')
    frame1 = Frame(name='frame1')
    
    print(frame1)

    frame2 = Frame(
        name='frame2',
        position=np.array([1.0, 1.0, 1.0]),
        reference_frame=None,
    )
    
    print(frame2)

    # angle to rotate:
    alpha = np.pi / 8.0

    # scipy.spatial.transform.Rotation object:
    rot = R.from_euler(
        seq='z',
        angles=alpha,
        degrees=False,
    )

    frame3 = Frame(
        name='frame3',
        position=np.array([0.0, 0.0, 0.0]),
        rotation=rot,
        reference_frame=None,
    )
    
    print(frame3)

    print(f"\nframe3.rot_mat = \n{frame3.rotation.as_matrix()}\n")

    print("frame3 unit vectors:")
    print(frame3.x_hat)
    print(frame3.y_hat)
    print(frame3.z_hat)

    frame4 = Frame(
        name='frame4',
        position=np.array([1.0, 1.0, 1.0]),
        rotation=R.from_quat([0.0, 0.0, 0.0, 1.0]),
        reference_frame=None,
    )
    
    print(frame4)

    frame4.apply_rotation(rot)
    print(f"\nframe4.rot_mat = \n{frame4.rotation.as_matrix()}\n")

    print("frame4 unit vectors:")
    print(frame4.x_hat)
    print(frame4.y_hat)
    print(frame4.z_hat)

    print(f"frame4.pos = \n{frame4.position}")
    print(f"frame4.position_global = \n{frame4.position_global}")

    frame5 = Frame(
        name='frame5',
        position=np.array([1.0, 1.0, 1.0]),
        rotation=R.from_quat([0.0, 0.0, 0.0, 1.0]),
        reference_frame=frame4,
    )
    
    print(f"frame5.pos = \n{frame5.position}")
    print(f"frame5.position_global = \n{frame5.position_global}")

    frame6 = Frame(
        name='frame6',
        position=np.array([1.0, 2.0, 3.0]),
        rotation=R.from_quat([0.0, 0.0, 0.0, 1.0]),
        reference_frame=ground_frame,
    )
    
    rp6g = get_relative_position(
        to_frame=frame6, 
        from_frame=ground_frame, 
        expressed_in_frame=ground_frame,
    )
    
    print(f"rel_pos(frame6, ground_frame, ground_frame) = {rp6g}")

    dist_6_g = get_distance_between_frames(
        frame6, 
        ground_frame,
    )

    print(f"distance(frame6, ground_frame) = {dist_6_g}")
    
    dist_actual = np.sqrt(frame6.position[0]**2 + frame6.position[1]**2 + frame6.position[2]**2)
    print(f"dist_actual = {dist_actual}")

    # Test apply_rotation():
    print("\n\nTesting apply_rotation():\n")
    frame1 = Frame(
        name='frame1',
        position=np.array([1.0, 1.0, 1.0]),
        rotation=R.from_quat([0.0, 0.0, 0.0, 1.0]),
        reference_frame=ground_frame,
    )
    
    frame2 = Frame(
        name='frame2',
        position=np.array([1.0, 1.0, 1.0]),
        rotation=R.from_quat([0.0, 0.0, 0.0, 1.0]),
        reference_frame=frame1,
    )
    
    frame3 = Frame(
        name='frame3',
        position=np.array([1.0, 1.0, 1.0]),
        rotation=R.from_quat([0.0, 0.0, 0.0, 1.0]),
        reference_frame=frame2,
    )
    
    plt.figure()
    ground_frame.plot_xy(color='k')
    frame1.plot_xy(color='r')
    frame2.plot_xy(color='b')
    frame3.plot_xy(color='g')
    plt.title('initial')
    plt.grid()
    plt.axis('equal')

    # scipy.spatial.transform.Rotation object:
    rot1 = R.from_euler(
        seq='z',
        angles=np.pi/4.0,
        degrees=False,
    )
    
    frame1.apply_rotation(rot1)

    plt.figure()
    ground_frame.plot_xy(color='k')
    frame1.plot_xy(color='r')
    frame2.plot_xy(color='b')
    frame3.plot_xy(color='g')
    plt.title('rotated frame1')
    plt.grid()
    plt.axis('equal')

    # scipy.spatial.transform.Rotation object:
    rot2 = R.from_euler(
        seq='z',
        angles=-np.pi/4.0,
        degrees=False,
    )
    
    frame2.apply_rotation(rot2)

    plt.figure()
    ground_frame.plot_xy(color='k')
    frame1.plot_xy(color='r')
    frame2.plot_xy(color='b')
    frame3.plot_xy(color='g')
    plt.title('rotated frame1 & frame2')
    plt.grid()
    plt.axis('equal')


    print("\nTest setting global rotation:\n")

    frame1 = Frame(
        name='frame1',
        position=np.array([1.0, 1.0, 1.0]),
        rotation=R.from_quat([0.0, 0.0, 0.0, 1.0]),
        reference_frame=ground_frame,
    )
    
    frame2 = Frame(
        name='frame2',
        position=np.array([1.0, 1.0, 1.0]),
        rotation=R.from_quat([0.0, 0.0, 0.0, 1.0]),
        reference_frame=frame1,
    )
    
    plt.figure()
    ground_frame.plot_xy(color='k')
    frame1.plot_xy(color='r')
    frame2.plot_xy(color='b')
    plt.title('initial')
    plt.grid()
    plt.axis('equal')

    # scipy.spatial.transform.Rotation object:
    rot1 = R.from_euler(
        seq='z',
        angles=np.pi/4.0,
        degrees=False,
    )
    
    # frame1.apply_rotation(rot1)
    frame1.rotation_global = rot1

    plt.figure()
    ground_frame.plot_xy(color='k')
    frame1.plot_xy(color='r')
    frame2.plot_xy(color='b')
    plt.title('rotated frame1')
    plt.grid()
    plt.axis('equal')

    # scipy.spatial.transform.Rotation object:
    rot2 = R.from_euler(
        seq='z',
        angles=-0.0,
        degrees=False,
    )
    
    # frame2.apply_rotation(rot2)
    # frame2.rotation_global = rot2
    frame2.set_rot_rel_to(
        rot2, 
        ground_frame,
    )

    plt.figure()
    ground_frame.plot_xy(color='k')
    frame1.plot_xy(color='r')
    frame2.plot_xy(color='b')
    plt.title('rotated frame1 & frame2')
    plt.grid()
    plt.axis('equal')

    print(frame2)
    
    frame2 = frame2.transform_to_frame(ground_frame)
    
    print(frame2)
    
    frame2.set_pos_rel_to(
        position=np.array([0.0, 0.0, 0.0]), 
        relative_to_frame=frame1,
    )

    plt.figure()
    ground_frame.plot_xy(color='k')
    frame1.plot_xy(color='r')
    frame2.plot_xy(color='b')
    plt.title('rotated frame1 & frame2')
    plt.grid()
    plt.axis('equal')

    plt.show()


if __name__ == "__main__":
    main()
    