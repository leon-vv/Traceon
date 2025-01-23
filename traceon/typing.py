from __future__ import annotations
from typing import TYPE_CHECKING, Any, Self, cast
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence

from ._typing import (
    # Array-Like types
    ArrayFloat, ArrayFloat1D, ArrayFloat2D, ArrayFloat3D, ArrayFloat4D, ArrayFloat5D,
    ArrayLikeFloat, ArrayLikeFloat1D, ArrayLikeFloat2D, ArrayLikeFloat3D, ArrayLikeFloat4D, ArrayLikeFloat5D,
    ArrayInt, ArrayInt1D, ArrayInt2D, ArrayInt3D, ArrayInt4D,
    ArrayLikeInt, ArrayLikeInt1D, ArrayLikeInt2D, ArrayLikeInt3D, ArrayLikeInt4D,

    # Backend types
    Jacobian2D, Jacobian3D, Jacobians2D, Jacobians3D,
    LineQuadPoints, LinesQuadPoints, TriangleQuadPoints, TrianglesQuadPoints,

    # Geometric types
    Vector2D, VectorLike2D, Vectors2D, VectorsLike2D,
    Vector3D, VectorLike3D, Vectors3D, VectorsLike3D,
    Point2D, PointLike2D, Points2D, PointsLike2D,
    Point3D, PointLike3D, Points3D, PointsLike3D,
    PointTransformFunction,
    Bounds2D, BoundsLike2D, Bounds3D, BoundsLike3D,

    # Mesh Types
    Line, LineLike, Lines, LinesLike,
    Triangle, TriangleLike, Triangles, TrianglesLike,
    Quad, QuadLike, Quads, QuadsLike,
    LineVertices, LinesVertices, Triangles, TrianglesVertices,
    ActiveLines, ActiveTriangles

)

if TYPE_CHECKING:
    from .geometry import Path, PathCollection, Surface, SurfaceCollection
    from .mesher import Mesh
    from .excitation import Excitation
    from .field import EffectivePointCharges, Field, FieldBEM, FieldRadialBEM
    try: 
        from traceon_pro.field import Field3DBEM # type: ignore
    except ImportError:
        Field3DBEM = Any


