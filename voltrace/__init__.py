"""Welcome!

Voltrace is a general software package used for numerical electron optics. Its main feature is the implementation of the Boundary Element Method (BEM) to quickly calculate the surface charge distribution.
The program supports both radial symmetry and general three-dimensional geometries. 
Electron tracing can be done very quickly using accurate radial series interpolation in both geometries.
The electron trajectories obtained can help determine the aberrations of the optical components under study.

If you have any issues using the package, please open an issue on the [Voltrace Github page](https://github.com/leon-vv/Voltrace).

The software is currently distributed under the `MPL 2.0` license. 

# Usage

In general, one starts with the `voltrace.geometry` module to create a mesh. For the BEM only the boundary of 
electrodes needs to be meshed. So in 2D (radial symmetry) the mesh consists of line elements while in 3D the
mesh consists of triangles.  Next, one specifies a suitable excitation (voltages) using the `voltrace.excitation` module.
The excited geometry can then be passed to the `voltrace.solver.solve_direct` function, which computes the resulting field. 
The field can be passed to the `voltrace.tracing.Tracer` class to compute the trajectory of electrons moving through the field.

# Validations

To make sure the software is correct, various problems from the literature with known solutions are analyzed using the Voltrace software and the
results compared. In this manner it has been shown that the software produces very accurate results very quickly. The validations can be found in the
[/validations](https://github.com/leon-vv/Voltrace/tree/main/validation) directory in the Github project. After installing Voltrace, the validations can be 
executed as follows:

```bash
    git clone https://github.com/leon-vv/Voltrace
    cd voltrace
    python3 ./validation/edwards2007.py --help
```

# Units

SI units are used throughout the codebase. Except for charge, which is stored as \\( \\frac{ \\sigma}{ \\epsilon_0} \\).
"""

import warnings as _warnings # Prevents _warnings to be reexported

__pdoc__ = {}
__pdoc__['util'] = False
__pdoc__['backend'] = False
__pdoc__['data'] = False
__pdoc__['voltrace.tracing.Tracer.__call__'] = True

_warnings.filterwarnings('ignore', '.*The value of the smallest subnormal for.* type is zero.')

from . import typing
from . import logging
from . import mesher
from . import geometry
from . import excitation
from . import field
from . import solver
from . import tracing
from . import focus
from . import plotting

from .logging import set_log_level, LogLevel
from .mesher import GeometricObject, Mesh
from .geometry import Path, PathCollection, Surface, SurfaceCollection
from .excitation import Excitation, ExcitationType, Symmetry
from .field import Field, FieldAxial, FieldBEM, FieldRadialAxial, FieldRadialBEM, FieldSuperposition
from .solver import solve_direct, solve_direct_superposition
from .tracing import Tracer, velocity_vec, velocity_vec_spherical, velocity_vec_xz_plane
from .focus import focus_position
from .plotting import new_figure, get_current_figure, plot_charge_density, plot_equipotential_lines, plot_mesh, plot_trajectories, show







