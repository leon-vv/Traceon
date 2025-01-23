from __future__ import annotations
import sys 

from typing import TYPE_CHECKING, Any, cast
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from ._array_like import(
    ArrayFloat, ArrayFloat1D, ArrayFloat2D, ArrayFloat3D, ArrayFloat4D, ArrayFloat5D,
    ArrayLikeFloat, ArrayLikeFloat1D, ArrayLikeFloat2D, ArrayLikeFloat3D, ArrayLikeFloat4D, ArrayLikeFloat5D,
    ArrayInt, ArrayInt1D, ArrayInt2D, ArrayInt3D, ArrayInt4D,
    ArrayLikeInt, ArrayLikeInt1D, ArrayLikeInt2D, ArrayLikeInt3D, ArrayLikeInt4D,
)

from ._backend import(
    Jacobian2D, Jacobian3D, Jacobians2D, Jacobians3D,
    LineQuadPoints, LinesQuadPoints, TriangleQuadPoints, TrianglesQuadPoints
)

from ._geometric_types import (
    Vector2D, VectorLike2D, Vectors2D, VectorsLike2D,
    Vector3D, VectorLike3D, Vectors3D, VectorsLike3D,
    Point2D, PointLike2D, Points2D, PointsLike2D,
    Point3D, PointLike3D, Points3D, PointsLike3D,
    PointTransformFunction,
    Bounds2D, BoundsLike2D, Bounds3D, BoundsLike3D
)

from ._mesh_types import (
    Line, LineLike, Lines, LinesLike,
    Triangle, TriangleLike, Triangles, TrianglesLike,
    Quad, QuadLike, Quads, QuadsLike,
    LineVertices, LinesVertices, Triangles, TrianglesVertices,
    ActiveLines, ActiveTriangles
)

