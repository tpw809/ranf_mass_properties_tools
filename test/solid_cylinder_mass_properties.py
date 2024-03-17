import numpy as np

# cylinder revolved about z axis

# [m], radius:
r = 0.02

# [m], height:
h = 0.02

# [kg/m^3], density:
rho = 0.5

# [m^3], volume:
v = h * np.pi * r**2
print(f"volume = {v}")

# [kg], mass:
m = v * rho
print(f"mass = {m}")

# [kg-m^2], principal inertia:
i_xx = (1.0 / 12.0) * m * (3.0 * r**2 + h**2)
i_yy = (1.0 / 12.0) * m * (3.0 * r**2 + h**2)
i_zz = 0.5 * m * r**2

print(f"i_xx = {i_xx}")
print(f"i_yy = {i_yy}")
print(f"i_zz = {i_zz}")
