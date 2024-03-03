"""Welcome!

Traceon is a general software package used for numerical electron optics. Its main feature is the implementation of the Boundary Element Method (BEM) to quickly calculate the surface charge distribution.
The program supports both radial symmetry and general three-dimensional geometries. 
Electron tracing can be done very quickly using accurate radial series interpolation in both geometries.
The electron trajectories obtained can help determine the aberrations of the optical components under study.

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

SI units are used throughout the codebase. Except for charge, which is stored as \( \\frac{ \\sigma}{ \\epsilon_0} \).
"""

import warnings

__pdoc__ = {}
__pdoc__['util'] = False
__pdoc__['backend'] = False
__pdoc__['data'] = False
__pdoc__['fast_multipole_method'] = False
__pdoc__['traceon.tracing.Tracer.__call__'] = True

warnings.filterwarnings('ignore', '.*The value of the smallest subnormal for.* type is zero.')


