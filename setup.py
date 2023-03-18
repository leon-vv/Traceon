from setuptools import setup

setup(
    name='traceon',
    version='0.1',
    description='Solver and tracer for electrostatic axisymmetrical problems',
    url='https://github.com/leon-vv/Traceon',
    author='Léon van Velzen',
    license='AGPLv3',
    packages=['traceon'],
    scripts= [],
    install_requires = ['matplotlib', 'numpy', 'numba', 'pygmsh>=7.1.13', 'scipy', 'findiff']
)




