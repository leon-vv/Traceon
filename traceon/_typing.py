from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any, 
    Self,
    TypeAlias, 
)

from collections.abc import (
    Callable,
    Generator,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence
)

import numpy as np

from numpy.typing import NDArray


import numpy as np
from numpy.typing import NDArray


Numeric: TypeAlias = float | int
NumpyNumeric: TypeAlias = np.floating | np.integer

Point3D: TypeAlias = NDArray[NumpyNumeric]  # Single point as a (3,) array
Point3DLike: TypeAlias = Point3D | Sequence[Numeric]  # Single point as (3,) array or sequence

Points: TypeAlias = NDArray[NumpyNumeric]  # Multiple points as (N, 3) array
PointsLike: TypeAlias = Points | Sequence[Point3DLike]  # (N, 3) array or sequence of Point3DLike

# Vector definitions
Vector3D: TypeAlias = NDArray[NumpyNumeric]  # Single vector as a (3,) array
Vector3DLike: TypeAlias = Vector3D | Sequence[Numeric]  # Single vector as (3,) array or sequence

# Vectors definitions
Vectors: TypeAlias = NDArray[NumpyNumeric]  # Multiple vectors as (N, 3) array
VectorsLike: TypeAlias = Vectors | Sequence[Vector3DLike]  # (N, 3) array or sequence of Vector3DLike

PointTransformFunction: TypeAlias = Callable[[Point3DLike], Point3D]

Line: TypeAlias = NDArray[np.integer] # Single line as (2,) (or (4,) for higher order) array of integer indices
LineLike: TypeAlias = Line | Sequence[int] # Single line as (2,) (or (4,)) array or sequence

Lines: TypeAlias = NDArray[np.integer] # Multiple lines as (N, 2) (or (N,4)) array of integer indices
LinesLike: TypeAlias = Lines | Sequence[LineLike] # Multiple lines as (N, 2) (or (N,4)) array or sequence of LineLike

Triangle : TypeAlias = NDArray[np.integer] # Single triangle as (3,) (or (6,) for higher order) array of integer indices
TriangleLike : TypeAlias = Triangle | Sequence[int] # Single triangle as (3,) (or (6,) array or sequence

Triangles: TypeAlias = NDArray[np.integer] # Multiple triangles as (N, 3) (or (N, 6)) array of integer
TrianglesLike: TypeAlias = Triangles | Sequence[TriangleLike] # Multiple triangles as (N, 3) (or (N, 6)) array or sequence of TriangleLike

Quad: TypeAlias = NDArray[np.integer]
QuadLike: TypeAlias = Quad | Sequence[int]

Quads: TypeAlias = NDArray[np.integer]
QuadsLike: TypeAlias = Quads | Sequence[QuadLike]

if TYPE_CHECKING:
    from .geometry import (
        Surface
    )