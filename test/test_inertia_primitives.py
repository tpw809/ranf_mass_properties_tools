import numpy as np
import mass_properties_tools as mp_tools

# uniform density solid cylinder:

# [m], cylinder radius:
r = 1.0

# [m], cylinder length:
l = 2.0

# [kg/m^3], cylinder material density:
rho = 1.0

# [m^3], cylinder volume:
vol = np.pi * r**2 * l
print(f"vol = {vol} [m^3]")

# [kg], cylinder mass:
m = rho * vol
print(f"m = {m} [kg]")

# [kg-m^2], cylinder inertia tensor at center of mass:
# get inertia tensor from mass_properties_tools:
# z axis is along the centerline of revolution
i = mp_tools.primitives.cylinder_inertia(
    m=m, r=r, lz=l)
print(f"i = \n{i}\n [kg-m^2]")
