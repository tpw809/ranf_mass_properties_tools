import sympy
import numpy as np
from scipy.spatial.transform import Rotation as R
from diagonalize_inertia_tensor import diagonalize, diagonalize2, diagonalize3


mp1 = MassProps()
mp1.cm = np.array([0.0, 0.0, -1.0])
print(mp1)

mp2 = MassProps()
mp2.cm = np.array([0.0, 0.0, 1.0])
mp3 = combine_mass_props(mp1, mp2)
print(mp3)

mp4 = mp3.copy()
print(diagonalize(mp4.i)[0])
print(diagonalize(mp4.i)[1])

print(f"combined ixx = {mp3.ixx}")
print(f"combined iyy = {mp3.iyy}")
print(f"combined izz = {mp3.izz}")
print(f"combined ixy = {mp3.ixy} = iyx = {mp3.iyx}")
print(f"combined ixz = {mp3.ixz} = izx = {mp3.izx}")
print(f"combined iyz = {mp3.iyz} = izy = {mp3.izy}")


# metric mass unit: [kg]
# metric inertia unit: [kg-m^2]

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
mp1.cm = np.array([0.0, 0.0, -1.0])
print(mp1)

mp2 = MassProperties()
mp2.cm = np.array([0.0, 0.0, 1.0])
mp3 = combine_mass_props(mp1, mp2)
print(mp3)

print(f"combined i_xx = {mp3.i_xx}")
print(f"combined i_yy = {mp3.i_yy}")
print(f"combined i_zz = {mp3.i_zz}")
print(f"combined i_xy = {mp3.i_xy} = i_yx = {mp3.i_yx}")
print(f"combined i_xz = {mp3.i_xz} = i_zx = {mp3.i_zx}")
print(f"combined i_yz = {mp3.i_yz} = i_zy = {mp3.i_zy}")

mp4 = mp3.copy()
print(diagonalize3(mp4.i)[0])
print(diagonalize3(mp4.i)[1])

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

mp1.i = rotate_inertia_tensor(I_A, R_AtoB)
# print(mp1)

# mp5 = diagonalize2(mp1.i)
# print(mp5)

mp6 = mp1.diagonalize(inplace=False)
print(f"\nmp6 = \n{mp6}\n")

eig_vals, eig_vecs = diagonalize(mp1.i)
print(eig_vals, eig_vecs)