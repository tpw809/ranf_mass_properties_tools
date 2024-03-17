from distutils.core import setup
# from setuptools import setup, find_packages

setup(
    name='mass_properties_tools',
    version='0.1.0',  # major.minor[.patch[.sub]]
    description='tools to help accomplish common tasks involving rigid body mass properties',
    author='Timothy Woodard',
    author_email='timothy.woodard.809@outlook.com',
    url='',
    download_url='',
    license='',
    packages=['mass_properties_tools'],
    # install_requires=['pandas', 'numpy', 'scipy', 'matplotlib']
    package_dir={'': 'src'},
    # py_modules=['mass_properties_tools'],
    )
