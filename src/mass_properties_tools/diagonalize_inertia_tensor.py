import numpy as np
from scipy.sparse.linalg import eigs

# direction unit vectors:
ux = np.array([1.0, 0.0, 0.0])
uy = np.array([0.0, 1.0, 0.0])
uz = np.array([0.0, 0.0, 1.0])


def diagonalize(inertia: np.ndarray):
    """
    find principal moments of inertia
    and principal axes (rotation matrix)
    """
    # w, v = np.linalg.eig(inertia)
    w, v = np.linalg.eigh(inertia)

    # eigenvectors make the rotation matrix:
    # columns of the rotation matrix should be direction unit vectors...
    eigenvectors = v

    # eigenvalues are the new moments of inertia:
    eigenvalues = w
    return eigenvalues, eigenvectors


def diagonalize2(inertia: np.ndarray, tol: float=1.0e-6):
    """
    find principal moments of inertia
    and principal axes
    eigenvectors make the rotation matrix
    eigenvalues are the new moments of inertia
    tol = tolerance for what can be considered 0.0
    """
    # check to see if it is already diagonal:
    i_xy = inertia[0, 1]
    i_xz = inertia[0, 2]
    i_yz = inertia[1, 2]
    
    if np.abs(i_xy) < tol and np.abs(i_xz) < tol and np.abs(i_yz) < tol:
        return inertia, np.identity(3)
    
    i_xx = inertia[0, 0]
    i_yy = inertia[1, 1]
    i_zz = inertia[2, 2]
        
    i = inertia
    
    # initialize empty matrices:
    i_diag = np.zeros((3, 3))
    rm_diag = np.zeros((3, 3))
    
    if np.abs(i_xy) < 1.0e-6 and np.abs(i_xz) < 1.0e-6:
        eig0 = i_xx
        vec0 = np.array([1.0, 0.0, 0.0])
    else:
        eig0, vec0 = eigs(i, k=1, sigma=i[0][0])
    
    if np.abs(i_xy) < 1.0e-6 and np.abs(i_yz) < 1.0e-6:
        eig1 = i_yy
        vec1 = np.array([0.0, 1.0, 0.0])
    else:    
        eig1, vec1 = eigs(i, k=1, sigma=i[1][1])
    
    if np.abs(i_yz) < 1.0e-6 and np.abs(i_xz) < 1.0e-6:
        eig2 = i_zz
        vec2 = np.array([0.0, 0.0, 1.0])
    else:
        eig2, vec2 = eigs(i, k=1, sigma=i[2][2])

    i_diag[0][0] = np.squeeze(np.real(eig0))
    i_diag[1][1] = np.squeeze(np.real(eig1))
    i_diag[2][2] = np.squeeze(np.real(eig2))

    rm_diag[:][0] = np.squeeze(np.real(vec0))
    rm_diag[:][1] = np.squeeze(np.real(vec1))
    rm_diag[:][2] = np.squeeze(np.real(vec2))


    # check directional alignment:
    alpha0 = np.arccos(np.dot(ux, rm_diag[0, :]))
    # print(f"alpha0 = {alpha0}")
    if alpha0 > np.pi / 2.0:
        rm_diag[0, :] = -rm_diag[0, :]

    alpha1 = np.arccos(np.dot(uy, rm_diag[1, :]))
    # print(f"alpha1 = {alpha1}")
    if alpha1 > np.pi / 2.0:
        rm_diag[1, :] = -rm_diag[1, :]

    alpha2 = np.arccos(np.dot(uz, rm_diag[2, :]))
    # print(f"alpha1 = {alpha1}")
    if alpha2 > np.pi / 2.0:
        rm_diag[2, :] = -rm_diag[2, :]

    # cross product check:
    # checking for right handedness and orthogonality
    zhat = np.cross(rm_diag[0, :], rm_diag[1, :])
    # print(f"zhat = {zhat}")
    
    zhat_err = np.linalg.norm(rm_diag[2, :] - zhat)
    
    if np.abs(zhat_err) > tol:
        print(f"zhat_err = {zhat_err}")
        print("warning: resulting rotation matrix is not orthogonal")

    return i_diag, rm_diag


def diagonalize3(inertia: np.ndarray, tol: float=1.0e-9):
    """
    find principal moments of inertia
    and principal axes
    eigenvectors make the rotation matrix
    eigenvalues are the new moments of inertia
    tol = tolerance for what can be considered 0.0
    """
    # check to see if it is already diagonal:
    i_xy = inertia[0, 1]
    i_xz = inertia[0, 2]
    i_yz = inertia[1, 2]
    
    if np.abs(i_xy) < tol and np.abs(i_xz) < tol and np.abs(i_yz) < tol:
        return inertia, np.identity(3)


    # eig_vals, eig_vecs = np.linalg.eig(inertia)
    # print(f"\neig eigenvalues = {eig_vals}")
    # print(f"eig eigenvectors = \n{eig_vecs}")

    # eigh assumes a symmetric matrix (faster):
    eig_vals, eig_vecs = np.linalg.eigh(inertia)
    # print(f"\neigh eigenvalues = {eig_vals}")
    # print(f"eigh eigenvectors = \n{eig_vecs}")

    rm_diag = eig_vecs

    i_diag = np.zeros((3, 3))
    i_diag[0, 0] = eig_vals[0]
    i_diag[1, 1] = eig_vals[1]
    i_diag[2, 2] = eig_vals[2]

    # try some stuff:

    # this works to recover the original supplied inertia:
    # print("\nsee if I recover the input inertia:")
    # print(eig_vecs @ i_diag @ np.transpose(eig_vecs))

    # there are 6 possible combinations of the resulting eigenvectors
    # forming a rotation matrix...

    new_eig_vecs = np.zeros([3,3])
    new_i_diag = np.zeros([3,3])

    # try columns:
    # find eigenvector closest to x-direction:
    # print("\n1st eigenvector:")
    # print(eig_vecs[:, 0])
    alpha0x = np.arccos(np.dot(ux, eig_vecs[:, 0]))
    if alpha0x > np.pi / 2.0:
        alpha0x = np.pi - alpha0x
        # eig_vecs[:, 0] = -eig_vecs[:, 0]
        # alpha0x = np.arccos(np.dot(ux, eig_vecs[:, 0]))
        # print(f"alpha0x = {alpha0x}")
    alpha0y = np.arccos(np.dot(uy, eig_vecs[:, 0]))
    if alpha0y > np.pi / 2.0:
        alpha0y = np.pi - alpha0y
        # eig_vecs[:, 0] = -eig_vecs[:, 0]
        # alpha0y = np.arccos(np.dot(uy, eig_vecs[:, 0]))
        # print(f"alpha0y = {alpha0y}")
    alpha0z = np.arccos(np.dot(uz, eig_vecs[:, 0]))
    if alpha0z > np.pi / 2.0:
        alpha0z = np.pi - alpha0z
        # eig_vecs[:, 0] = -eig_vecs[:, 0]
        # alpha0z = np.arccos(np.dot(uz, eig_vecs[:, 0]))
        # print(f"alpha0z = {alpha0z}")

    # find smallest alpha0:
    alpha0_min = alpha0x
    alpha0_ind = 0
    if alpha0y < alpha0_min:
        alpha0_min = alpha0y
        alpha0_ind = 1
    if alpha0z < alpha0_min:
        alpha0_ind = 2
        alpha0_min = alpha0z

    if eig_vecs[alpha0_ind, 0] < 0.0:
        eig_vecs[:, 0] = -eig_vecs[:, 0]

    new_eig_vecs[:, alpha0_ind] = eig_vecs[:, 0]
    new_i_diag[alpha0_ind, alpha0_ind] = eig_vals[0]

    # print("\n2nd eigenvector:")
    # print(eig_vecs[:, 1])
    alpha1x = np.arccos(np.dot(ux, eig_vecs[:, 1]))
    if alpha1x > np.pi / 2.0:
        alpha1x = np.pi - alpha1x
        # eig_vecs[:, 1] = -eig_vecs[:, 1]
        # alpha1x = np.arccos(np.dot(ux, eig_vecs[:, 1]))
        # print(f"alpha1x = {alpha1x}")
    alpha1y = np.arccos(np.dot(uy, eig_vecs[:, 1]))
    if alpha1y > np.pi / 2.0:
        alpha1y = np.pi - alpha1y
        # eig_vecs[:, 1] = -eig_vecs[:, 1]
        # alpha1y = np.arccos(np.dot(uy, eig_vecs[:, 1]))
        # print(f"alpha1y = {alpha1y}")
    alpha1z = np.arccos(np.dot(uz, eig_vecs[:, 1]))
    if alpha1z > np.pi / 2.0:
        alpha1z = np.pi - alpha1z
        # eig_vecs[:, 1] = -eig_vecs[:, 1]
        # alpha1z = np.arccos(np.dot(uz, eig_vecs[:, 1]))
        # print(f"alpha1z = {alpha1z}")

    # find smallest alpha1:
    alpha1_min = alpha1x
    alpha1_ind = 0
    if alpha1y < alpha1_min:
        alpha1_min = alpha1y
        alpha1_ind = 1
    if alpha1z < alpha1_min:
        alpha1_ind = 2
        alpha1_min = alpha1z

    if eig_vecs[alpha1_ind, 1] < 0.0:
        eig_vecs[:, 1] = -eig_vecs[:, 1]

    new_eig_vecs[:, alpha1_ind] = eig_vecs[:, 1]
    new_i_diag[alpha1_ind, alpha1_ind] = eig_vals[1]

    # print("\n3rd eigenvector:")
    # print(eig_vecs[:, 2])
    alpha2x = np.arccos(np.dot(ux, eig_vecs[:, 2]))
    if alpha2x > np.pi / 2.0:
        alpha2x = np.pi - alpha2x
        # eig_vecs[:, 2] = -eig_vecs[:, 2]
        # alpha2x = np.arccos(np.dot(ux, eig_vecs[:, 2]))
        # print(f"alpha2x = {alpha2x}")
    alpha2y = np.arccos(np.dot(uy, eig_vecs[:, 2]))
    if alpha2y > np.pi / 2.0:
        alpha2y = np.pi - alpha2y
        # eig_vecs[:, 2] = -eig_vecs[:, 2]
        # alpha2y = np.arccos(np.dot(uy, eig_vecs[:, 2]))
        # print(f"alpha2y = {alpha2y}")
    alpha2z = np.arccos(np.dot(uz, eig_vecs[:, 2]))
    if alpha2z > np.pi / 2.0:
        alpha2z = np.pi - alpha2z
        # eig_vecs[:, 2] = -eig_vecs[:, 2]
        # alpha2z = np.arccos(np.dot(uz, eig_vecs[:, 2]))
        # print(f"alpha2z = {alpha2z}")

    # find smallest alpha0:
    alpha2_min = alpha2x
    alpha2_ind = 0
    if alpha2y < alpha2_min:
        alpha2_min = alpha2y
        alpha2_ind = 1
    if alpha2z < alpha2_min:
        alpha2_ind = 2
        alpha2_min = alpha2z

    if eig_vecs[alpha2_ind, 2] < 0.0:
        eig_vecs[:, 2] = -eig_vecs[:, 2]

    new_eig_vecs[:, alpha2_ind] = eig_vecs[:, 2]
    new_i_diag[alpha2_ind, alpha2_ind] = eig_vals[2]

    # ensure orthogonal and right handed:
    # cross product check:
    # checking for right handedness and orthogonality
    # zhat = np.cross(rm_diag[0, :], rm_diag[1, :])
    zhat = np.cross(new_eig_vecs[:, 0], new_eig_vecs[:, 1])
    # print(f"\nzhat = {zhat}")

    # zhat_err = np.linalg.norm(rm_diag[2, :] - zhat)
    zhat_err = np.linalg.norm(new_eig_vecs[:, 2] - zhat)
    
    if np.abs(zhat_err) > 1.0e-6:
        print(f"zhat_err = {zhat_err}")
        print("warning: resulting rotation matrix is not orthogonal right handed")

    # return i_diag, rm_diag
    return new_i_diag, new_eig_vecs


def main() -> None:
    from scipy.spatial.transform import Rotation as R
    from mass_properties_tools import triangle_inequality
    from mass_properties_tools import rotate_inertia_tensor

    # create a random inertia tensor:
    I = np.array([
        [1.2, 0.0, 0.0],
        [0.0, 1.1, 0.0],
        [0.0, 0.0, 0.8],
    ])
    print(f"I = \n{I}")

    print(triangle_inequality(I))

    # create a random rotation:
    # rot = R.random()

    # only works when rotation is relatively small:
    # i.e.: principal axes are nearly aligned with reference frame...
    rot = R.from_euler(
        seq='xyz',
        angles=[
            np.random.random() * np.pi / 6.0,
            np.random.random() * np.pi / 6.0,
            np.random.random() * np.pi / 6.0,
        ],
        degrees=False,
    )

    # rot = R.from_euler(
    #     seq='xyz',
    #     angles=[1.0 * np.pi / 18.0,
    #             1.0 * np.pi / 14.0,
    #             0.0 * np.pi / 4.0])

    # rot = R.from_quat([0.0,0.0,0.0,1.0])

    print(f"\nrot = \n{np.round(rot.as_matrix(), 6)}")

    rot_mat = rot.as_matrix()

    # rotate the inertia tensor:
    # I' = R.I.R^T
    I_rot = rot_mat @ I @ np.transpose(rot_mat)
    print(f"\nI_rot = \n{I_rot}")

    # try diagonalize 1:
    # [eigenvalues, eigenvectors] = diagonalize(I_rot)

    # print(f"eigenvalues = \n{eigenvalues}")
    # print(f"eigenvectors = \n{eigenvectors}")

    # I_rot_back = eigenvectors @ I @ np.transpose(eigenvectors)
    # I_rot_back = np.transpose(eigenvectors) @ I @ eigenvectors
    # print(f"I_rot_back = \n{I_rot_back}")
    # diag1 not guranteed to work...

    # try diagonalize 2:
    # i_diag2, rm_diag2 = diagonalize2(inertia=I_rot)
    # print(f"i_diag2 = \n{i_diag2}")
    # print(f"rm_diag2 = \n{rm_diag2}")
    # 
    # I_rot_back = rm_diag2 @ I @ np.transpose(rm_diag2)
    # print(f"I_rot_back = \n{I_rot_back}")

    # try diagonalize 3:
    i_diag3, rm_diag3 = diagonalize3(inertia=I_rot)
    print(f"\ni_diag3 = \n{i_diag3}")
    print(f"\nrm_diag3 = \n{np.round(rm_diag3, 6)}")

    # I_rot_back3 = np.transpose(rm_diag3) @ I_rot @ rm_diag3
    I_rot_back3 = rotate_inertia_tensor(
        i=I_rot, 
        rot_mat=np.transpose(rm_diag3),
    )
    print(f"\nI_rot_back3 = \n{np.round(I_rot_back3, 6)}")

    I_rot_back3 = rm_diag3 @ I @ np.transpose(rm_diag3)
    print(f"\nI_rot_back3 = \n{I_rot_back3}")


if __name__ == "__main__":
    main()
    