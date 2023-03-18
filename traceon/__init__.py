"""Welcome!

Traceon is a general software package for numerical
charged particle optics. The heart of the package is an implementation of the Boundary Element
Method (BEM) to efficiently compute electrostatic fields. Currently radial symmetry and general
three dimensional geometries are supported. In both symmetries very accuracute and efficient radial
series interpolation can be used to make electron tracing very fast. The resulting electron trajectories
can be used to determine the aberrations of optical components under study.

If you have any issues using the package, please open an issue on the [Traceon Github page](https://github.com/leon-vv/Traceon).

The software is currently distributed under the `AGPLv3` license. 

# Usage

In general, one starts with the `traceon.geometry` module to create a mesh. For the BEM only the boundary of 
electrodes needs to be meshed. So in 2D (radial symmetry) the mesh consists of line elements while in 3D the
mesh consists of triangles.  Next, one specifies a suitable excitation (voltages) using the `traceon.excitation` module.
The excited geometry can then be passed to the `traceon.solver.solve_bem` function, which computes the resulting field. 
The field can be passed to the `traceon.tracing.Tracer` class to compute the trajectory of electrons moving through the field.

# Validations

To make sure the software is correct, various problems from the literature with known solutions are analyzed using the Traceon software and the
results compared. In this manner it has been shown that the software produces very accurate results very quickly. The validations can be found in the
[/validations](https://github.com/leon-vv/Traceon/tree/main/validation) directory in the Github project. After installing Traceon, the validations can be 
executed as follows:

```bash
    git clone https://github.com/leon-vv/Traceon
    cd traceon
    python3 ./validation/edwards2007.py --help
```

# Units

The units used throughout the code base are as follows:

| **Name**       	| **Unit** 	|
|----------------	|----------	|
| Length         	| mm       	|
| Time           	| ns       	|
| Velocity       	| mm/ns    	|
| Potential      	| V        	|
| Electric field 	| V/mm     	|
| Charge         	| see below 	|
|                	|          	|

To keep the charge values from becoming very small, the charge values are always saved as \( \\frac{ \\sigma}{ \\epsilon_0 \\pi} \). Since this term appears
in the formulas for the potential (and thus also in the formulas for the electric field) the actual \( \\sigma \) values themselves are never computed.

"""

__pdoc__ = {}
__pdoc__['traceon.util'] = False
__pdoc__['traceon.backend'] = False
__pdoc__['traceon.tracing.Tracer.__call__'] = True



