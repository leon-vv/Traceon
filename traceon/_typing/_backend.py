from __future__ import annotations
import sys
from typing import Literal
from collections.abc import Sequence

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

import numpy as np

from ._array_like import _NumpyFloat
from ._geometric_types import _VectorShape2D, _VectorShape3D, PointLike2D, PointLike3D

_NumQuad2D: TypeAlias = Literal[16]  # Number of quadrature points on a line
_NumQuad3D: TypeAlias = Literal[12]  # Number of quadrature points on triangles

_QuadShape2D: TypeAlias = tuple[_NumQuad2D]  # Shape of 2D quadrature points: `(N_QUAD_2D,)`
_BatchQuadShape2D: TypeAlias = tuple[int, *_QuadShape2D]  # Shape of batch of 2D quadrature points: `(N, N_QUAD_2D)`

_QuadShape3D: TypeAlias = tuple[_NumQuad3D]  # Shape of 3D quadrature points: `(N_TRIANGLE_QUAD,)`
_BatchQuadShape3D: TypeAlias = tuple[int, *_QuadShape3D]  # Shape of batch of 3D quadrature points: `(N, N_TRIANGLE_QUAD)`

Jacobian2D: TypeAlias = np.ndarray[_QuadShape2D, _NumpyFloat]
"""A two-dimensional Jacobian
 as a `(N_QUAD_2D,)` NumPy array of `float`."""

Jacobians2D: TypeAlias = np.ndarray[_BatchQuadShape2D, _NumpyFloat]
"""A batch of two-dimensional Jacobians
 as a `(N, N_QUAD_2D)` NumPy array of `float`."""

Jacobian3D: TypeAlias = np.ndarray[_QuadShape3D, _NumpyFloat]
"""A three-dimensional Jacobian 
as a `(N_TRIANGLE_QUAD,)` NumPy array of `float`."""

Jacobians3D: TypeAlias = np.ndarray[_BatchQuadShape3D, _NumpyFloat]
"""A batch of three-dimensional Jacobians
as a `(N, N_TRIANGLE_QUAD)` NumPy array of `float`."""


_QuadVectorShape2D: TypeAlias = tuple[*_QuadShape2D, *_VectorShape2D] 
_QuadVectorShape3D: TypeAlias = tuple[*_QuadShape3D, *_VectorShape3D]

_BatchQuadVectorShape2D: TypeAlias = tuple[*_BatchQuadShape2D, *_VectorShape2D]
_BatchQuadVectorShape3D: TypeAlias = tuple[*_BatchQuadShape3D, *_VectorShape3D]

LineQuadPoints: TypeAlias = np.ndarray[_QuadVectorShape2D, _NumpyFloat]
"""Quadrature points of a line
as `(N_QUAD_2D, 2)` NumPy array of `float`"""

LinesQuadPoints: TypeAlias = np.ndarray[_BatchQuadVectorShape2D, _NumpyFloat]
"""Quadrature points of a batch of lines
as `(N, N_QUAD_2D, 2)` NumPy array of `float`"""

TriangleQuadPoints: TypeAlias = np.ndarray[_QuadVectorShape3D, _NumpyFloat]
"""Quadrature points of a triangle
as `(N_TRIANLGE_QUAD, 3)` NumPy array of `float`"""

TrianglesQuadPoints: TypeAlias = np.ndarray[_BatchQuadVectorShape3D, _NumpyFloat]
"""Quadrature points of a batch of triangles
as `(N_TRIANLGE_QUAD, 3)` NumPy array of `float`"""



