from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any, 
    Self,
    TypeAlias, 
    cast
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


Point3D: TypeAlias = NDArray[np.floating]  # Single point as a (3,) array
Point3DLike: TypeAlias = Point3D | Sequence[float]  # Single point as (3,) array or sequence

Points: TypeAlias = NDArray[np.floating]  # Multiple points as (N, 3) array
PointsLike: TypeAlias = Points | Sequence[Point3DLike]  # (N, 3) array or sequence of Point3DLike

Vector3D: TypeAlias = NDArray[np.floating]  # Single vector as a (3,) array
Vector3DLike: TypeAlias = Vector3D | Sequence[float]  # Single vector as (3,) array or sequence

Vectors: TypeAlias = NDArray[np.floating]  # Multiple vectors as (N, 3) array
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

Indices = NDArray[np.integer]
IndicesLike = Indices | Sequence[int]


Jacobian = NDArray[np.floating]
Jacobians = NDArray[np.floating]
if TYPE_CHECKING:
    from .geometry import Path, PathCollection, Surface, SurfaceCollection
    from .mesher import Mesh
    from .excitation import Excitation
    from .field import EffectivePointCharges, Field, FieldBEM, FieldRadialBEM