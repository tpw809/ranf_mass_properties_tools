"""Defines function to rotate an inertia tensor."""
import numpy as np


def rotate_inertia_tensor(i: np.ndarray, rot_mat: np.ndarray) -> np.ndarray:
    """Rotate an inertia tensor using a rotation matrix.
    I' = R . I . R^T
    
    Args:
        i (np.ndarray): Inertia tensor.
        rot_mat (np.ndarray): Rotation matrix.
    
    Returns:
        np.ndarray: Rotated inertia tensor.
    """
    return rot_mat @ i @ np.transpose(rot_mat)


def main() -> None:
    from scipy.spatial.transform import Rotation as R

    I_A = np.array([
        [1.5, 0.0, 0.0],
        [0.0, 1.1, 0.0],
        [0.0, 0.0, 0.8],
    ])
    print(f"Inertia tensor in frame A: \n{I_A}")

    theta = np.pi / 4.0
    R_AtoB = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])

    R_AtoB_scipy = R.from_euler(
        seq='z',
        angles=theta,
        degrees=False,
    )

    print("Rotation Matrix from A to B:")
    print(R_AtoB)
    print(R_AtoB_scipy.as_matrix())

    # manually:
    I_B = np.dot(np.dot(R_AtoB, I_A), np.transpose(R_AtoB))
    print(f"Inertia tensor in frame B: \n{I_B}")

    # use defined function:
    I_B2 = rotate_inertia_tensor(
        i=I_A, 
        rot_mat=R_AtoB,
    )
    print(f"Inertia tensor in frame B: \n{I_B2}")

    # rotate back:
    R_BtoA = R_AtoB_scipy.inv()
    I_A_back = R_BtoA.as_matrix() @ I_B @ np.transpose(R_BtoA.as_matrix())
    print(f"Inertia tensor rotated back to frame A: \n{I_A_back}")

    I_A_back2 = rotate_inertia_tensor(
        i=I_B2, 
        rot_mat=R_BtoA.as_matrix(),
    )
    print(f"Inertia tensor rotated back to frame A: \n{I_A_back2}")


if __name__ == "__main__":
    main()
    