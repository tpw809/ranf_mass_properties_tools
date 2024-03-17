from scipy.linalg import eig
import numpy as np
from scipy.sparse.linalg import eigs

# direction unit vectors:
ux = np.array([1.0, 0.0, 0.0])
uy = np.array([0.0, 1.0, 0.0])
uz = np.array([0.0, 0.0, 1.0])


def rot_x(t):
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, np.cos(t), -np.sin(t)],
                     [0.0, np.sin(t), np.cos(t)]])


def rot_y(t):
    return np.array([[np.cos(t), 0.0, np.sin(t)],
                     [0.0, 1.0, 0.0],
                     [-np.sin(t), 0.0, np.cos(t)]])


def rot_z(t):
    return np.array([[np.cos(t), -np.sin(t), 0.0],
                     [np.sin(t), np.cos(t), 0.0],
                     [0.0, 0.0, 1.0]])


def eulerb313_to_rotmat(t1, t2, t3):
    """
    b = body-fixed
    313 = z-x-z
    """
    rm1 = rot_z(t1)
    rm2 = rot_x(t2)
    rm3 = rot_z(t3)
    return np.dot(rm1, np.dot(rm2, rm3))


def axis_angle_rot_mat(axis, angle):
    """
    take axis-angle representation and create a rotation matrix
    axis = axis of rotation
    angle = angle of rotation about the axis
    """
    # make axis a unit vector:
    e = axis / np.linalg.norm(axis)
    # pre-calculate cosine, sine:
    ca = np.cos(angle)
    sa = np.sin(angle)
    ux = np.array([1.0, 0.0, 0.0])
    uy = np.array([0.0, 1.0, 0.0])
    uz = np.array([0.0, 0.0, 1.0])
    uxr = ca * ux + sa * (np.cross(e, ux)) + (1.0 - ca) * np.dot(e, ux) * e
    uyr = ca * uy + sa * (np.cross(e, uy)) + (1.0 - ca) * np.dot(e, uy) * e
    uzr = ca * uz + sa * (np.cross(e, uz)) + (1.0 - ca) * np.dot(e, uz) * e
    return np.transpose(np.array([uxr, uyr, uzr]))


# inertia of ellipsoid:
m = 100.0
a = 1.0
b = 1.2
c = 0.8
Ia = 1.0 / 5.0 * m * (b**2 + c**2)
Ib = 1.0 / 5.0 * m * (a**2 + c**2)
Ic = 1.0 / 5.0 * m * (a**2 + b**2)
print(f"Ia = {Ia}")
print(f"Ib = {Ib}")
print(f"Ic = {Ic}")

I = np.array([[Ia, 0.0, 0.0],
              [0.0, Ib, 0.0],
              [0.0, 0.0, Ic]])
print(f"I = {I}")


R1 = axis_angle_rot_mat(axis=uz, angle=np.pi/2.0)
print(f"R1 = \n{R1}")

R1 = rot_z(np.pi/2.0)
print(f"R1 = \n{R1}")

Ry = axis_angle_rot_mat(axis=uy, angle=np.pi/2.0)
print(f"Ry = \n{Ry}")

Ry = rot_y(np.pi/2.0)
print(f"Ry = \n{Ry}")

# find principal moments of inertia:
eigvals, eigvecs = np.linalg.eig(I)
print(f"eigvals = {eigvals}")
print(f"eigvecs = {eigvecs}")

vals, vecs = eig(I)
print(f"vals = {vals}")
print(f"vecs = {vecs}")

"""
vals, vecs = eigs(I, k=1, sigma=I[0, 0])
print(f"vals = {vals}")
print(f"vecs = {vecs}")
"""

# how to rotate a vector:
vec1 = np.array([1.0, 0.0, 0.0])
vec2 = np.dot(R1, vec1)
print(f"vec2 = {vec2}")

# create rotated inertia tensor:
RM = axis_angle_rot_mat(axis=np.array([1.0, 1.0, 1.0]), angle=np.pi/20.0)
RMT = np.transpose(RM)
print(f"RM = \n{RM}")
# print(f"RMT * RM = \n{np.dot(RMT, RM)}")
# print(f"RM * RMT = \n{np.dot(RM, RMT)}")

A = np.dot(RMT, np.dot(I, RM))
print(f"A = \n{A}")

# Diagonalization:
# Q^T.A.Q = D
# Q = eigenvectors of A
# D = eigenvalues of A
vals, vecs = eig(A)
# print(f"vals = {vals}")
# print(f"vecs = {vecs}")

val0, vec0 = eigs(A, k=1, sigma=A[0, 0])
val1, vec1 = eigs(A, k=1, sigma=A[1, 1])
val2, vec2 = eigs(A, k=1, sigma=A[2, 2])
# print(f"val0 = {val0}")
# print(f"vec0 = {vec0}")

Q = np.zeros((3, 3))
Q[:, 0] = -np.squeeze(np.real(vec0))
Q[:, 1] = -np.squeeze(np.real(vec1))
Q[:, 2] = -np.squeeze(np.real(vec2))
print(f"Q = \n{Q}")

print(np.round(Q - RMT, 8))
