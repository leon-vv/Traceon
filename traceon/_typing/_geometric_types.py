from __future__ import annotations
import sys
from typing import Literal
from collections.abc import Callable, Sequence

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

import numpy as np
from ._array_like import _NumpyFloat

_Dim2D: TypeAlias = Literal[2]
_Dim3D: TypeAlias = Literal[3]

#: Vectors
_VectorShape2D: TypeAlias = tuple[_Dim2D]  # `(2,)`
_BatchVectorShape2D: TypeAlias = tuple[int, *_VectorShape2D]  # `(N, 2)`

_VectorShape3D: TypeAlias = tuple[_Dim3D]  # `(3,)`
_BatchVectorShape3D: TypeAlias = tuple[int, *_VectorShape3D]

Vector2D: TypeAlias = np.ndarray[_VectorShape2D, _NumpyFloat]
"""Two-dimensional vector 
as `(2,)` NumPy array of `float`."""

VectorLike2D: TypeAlias = Vector2D | Sequence[float]
"""Two-dimensional vector 
as `(2,)` NumPy array or sequence of `float`."""

Vectors2D: TypeAlias = np.ndarray[_BatchVectorShape2D, _NumpyFloat]
"""Batch of two-dimensional vectors 
as `(N, 2)` NumPy array of `float`."""

VectorsLike2D: TypeAlias = Vectors2D | Sequence[VectorLike2D]
"""Batch of two-dimensional vectors 
as `(N, 2)` NumPy array of `float` or sequence of `VectorLike2D` objects."""

Vector3D: TypeAlias = np.ndarray[_VectorShape3D, _NumpyFloat]
"""Three-dimensional vector 
as `(3,)` NumPy array of `float`."""

VectorLike3D: TypeAlias = Vector3D | Sequence[float]
"""Three-dimensional vector 
as `(3,)` NumPy array or sequence of `float`."""

Vectors3D: TypeAlias = np.ndarray[_BatchVectorShape3D, _NumpyFloat]
"""Batch of three-dimensional vectors 
as `(N, 3)` NumPy array of `float`."""

VectorsLike3D: TypeAlias = Vectors3D | Sequence[VectorLike3D]
"""Batch of three-dimensional vectors 
as `(N, 3)` NumPy array of `float` or sequence of `VectorLike3D` objects."""

#: Points
Point2D: TypeAlias = np.ndarray[_VectorShape2D, _NumpyFloat]
"""Two-dimensional point 
as `(2,)` NumPy array of `float`."""

PointLike2D: TypeAlias = Point2D | Sequence[float]
"""Two-dimensional point 
as `(2,)` NumPy array or sequence of `float`."""

Points2D: TypeAlias = np.ndarray[_BatchVectorShape2D, _NumpyFloat]
"""Batch of two-dimensional points 
as `(N, 2)` NumPy array of `float`."""

PointsLike2D: TypeAlias = Points2D | Sequence[PointLike2D]
"""Batch of two-dimensional points 
as `(N, 2)` NumPy array of `float` or sequence of two-dimensional `Pointlike2D` objects."""

Point3D: TypeAlias = np.ndarray[_VectorShape3D, _NumpyFloat]
"""A three-dimensional point 
as `(3,)` NumPy array of `float`."""

PointLike3D: TypeAlias = Point3D | Sequence[float]
"""A three-dimensional point 
as `(3,)` NumPy array or sequence of `float`."""

Points3D: TypeAlias = np.ndarray[_BatchVectorShape3D, _NumpyFloat]
"""Batch of three-dimensional points 
as `(N, 3)` NumPy array of `float`."""

PointsLike3D: TypeAlias = Points3D | Sequence[PointLike3D]
"""Batch of three-dimensional points 
as `(N, 3)` NumPy array of `float` or sequence of `PointLike3D` objects."""

PointTransformFunction: TypeAlias = Callable[[PointLike3D], Point3D]
"""Function mapping a three-dimensional point as `PointLike3D` 
to another three-dimensional point as `Point3D`."""

#: Bounds
_BoundShape: TypeAlias = tuple[Literal[2]]  # `(2,)`
_BatchBoundShape2D: TypeAlias = tuple[Literal[2], Literal[2]]  # `(2, 2)`
_BatchBoundShape3D: TypeAlias = tuple[Literal[3], Literal[2]]  # `(3, 2)`

_Bound: TypeAlias = np.ndarray[_BoundShape, _NumpyFloat]

_BoundLike: TypeAlias = _Bound | Sequence[float]

Bounds2D: TypeAlias = np.ndarray[_BatchBoundShape2D, _NumpyFloat]
"""Two-dimensional bounds `[(x_min, x_max), (y_min, y_max)]` 
as `(2, 2)` NumPy array of `float`."""

BoundsLike2D: TypeAlias = Bounds2D | Sequence[_BoundLike]
"""Two-dimensional bounds `[(x_min, x_max), (y_min, y_max)]` 
as `(2, 2)` NumPy array of `float` or a sequence of  objects."""

Bounds3D: TypeAlias = np.ndarray[_BatchBoundShape3D, _NumpyFloat]
"""Three-dimensional bounds `[(x_min, x_max), (y_min, y_max), (z_min, z_max)]` 
as `(3, 2)` NumPy array of `float`."""

BoundsLike3D: TypeAlias = Bounds3D | Sequence[_BoundLike]
"""Three-dimensional bounds `[(x_min, x_max), (y_min, y_max), (z_min, z_max)]` 
as `(3, 2)` NumPy array of `float` or a sequence of `_BoundLike` objects."""
