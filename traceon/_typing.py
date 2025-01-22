from __future__ import annotations
import sys 

from typing import (
    TYPE_CHECKING,
    Any, 
    Generic,
    Literal,
    Self,
    TypeAlias, 
    TypeVar,
    Tuple,
    cast,
    overload
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

if sys.version_info >= (3, 12):
    from typing import overload  # Python 3.12+ overload
else:
    from typing_extensions import overload

if TYPE_CHECKING:
    from .geometry import Path, PathCollection, Surface, SurfaceCollection
    from .mesher import Mesh
    from .excitation import Excitation
    from .field import EffectivePointCharges, Field, FieldBEM, FieldRadialBEM
    from traceon_pro.field import Field3DBEM # type: ignore

from typing import TypeAlias, Callable, Literal, Sequence, Union
import numpy as np

_NumpyFloat: TypeAlias = np.dtype[np.floating]
_NumpyInt: TypeAlias = np.dtype[np.integer]
_NumpySignedInt: TypeAlias = np.dtype[np.signedinteger]

ShapeLike: TypeAlias = tuple[int, ...]

_Shape1D: TypeAlias = tuple[int]
_Shape2D: TypeAlias = tuple[int, int]
_Shape3D: TypeAlias = tuple[int, int, int]
_Shape4D: TypeAlias = tuple[int, int, int, int]
_Shape5D: TypeAlias = tuple[int, int, int, int, int]

ArrayFloat: TypeAlias = np.ndarray[ShapeLike, _NumpyFloat]
ArrayFloat1D: TypeAlias = np.ndarray[_Shape1D, _NumpyFloat]
ArrayFloat2D: TypeAlias = np.ndarray[_Shape2D, _NumpyFloat]
ArrayFloat3D: TypeAlias = np.ndarray[_Shape3D, _NumpyFloat]
ArrayFloat4D: TypeAlias = np.ndarray[_Shape4D, _NumpyFloat]
ArrayFloat5D: TypeAlias = np.ndarray[_Shape5D, _NumpyFloat]

ArrayLikeFloat: TypeAlias = ArrayFloat | Sequence[float]
ArrayLikeFloat1D: TypeAlias = ArrayFloat1D | Sequence[float]
ArrayLikeFloat2D: TypeAlias = ArrayFloat2D | Sequence[ArrayLikeFloat1D]
ArrayLikeFloat3D: TypeAlias = ArrayFloat3D | Sequence[ArrayLikeFloat2D]
ArrayLikeFloat4D: TypeAlias = ArrayFloat4D | Sequence[ArrayLikeFloat3D]
ArrayLikeFloat5D: TypeAlias = ArrayFloat5D | Sequence[ArrayLikeFloat4D]

ArrayInt: TypeAlias = np.ndarray[ShapeLike, _NumpyInt]
ArrayInt1D: TypeAlias = np.ndarray[_Shape1D, _NumpyInt]
ArrayInt2D: TypeAlias = np.ndarray[_Shape2D, _NumpyInt]
ArrayInt3D: TypeAlias = np.ndarray[_Shape3D, _NumpyInt]
ArrayInt4D: TypeAlias = np.ndarray[_Shape4D, _NumpyInt]

ArrayLikeInt: TypeAlias = ArrayInt | Sequence[int]
ArrayLikeInt1D: TypeAlias = ArrayInt1D | Sequence[int]
ArrayLikeInt2D: TypeAlias = ArrayInt2D | Sequence[ArrayLikeInt1D]
ArrayLikeInt3D: TypeAlias = ArrayInt3D | Sequence[ArrayLikeInt2D]
ArrayLikeInt4D: TypeAlias = ArrayInt4D | Sequence[ArrayLikeInt3D]

_Dim2D: TypeAlias = Literal[2]
_Dim3D: TypeAlias = Literal[3]

_VectorShape2D: TypeAlias = tuple[_Dim2D]
_BatchVectorShape2D: TypeAlias = tuple[int, *_VectorShape2D]

_VectorShape3D: TypeAlias = tuple[_Dim3D]
_BatchVectorShape3D: TypeAlias = tuple[int, *_VectorShape3D]

Vector2D: TypeAlias = np.ndarray[_VectorShape2D, _NumpyFloat]
VectorLike2D: TypeAlias = Vector2D | Sequence[float]

Vectors2D: TypeAlias = np.ndarray[_BatchVectorShape2D, _NumpyFloat]
VectorsLike2D: TypeAlias = Vectors2D | Sequence[VectorLike2D]

Vector3D: TypeAlias = np.ndarray[_VectorShape3D, _NumpyFloat]
VectorLike3D: TypeAlias = Vector3D | Sequence[float]

Vectors3D: TypeAlias = np.ndarray[_BatchVectorShape3D, _NumpyFloat]
VectorsLike3D: TypeAlias = Vectors3D | Sequence[VectorLike3D]

Point2D: TypeAlias = np.ndarray[_VectorShape2D, _NumpyFloat]
PointLike2D: TypeAlias = Point2D | Sequence[float]

Points2D: TypeAlias = np.ndarray[_BatchVectorShape2D, _NumpyFloat]
PointsLike2D: TypeAlias = Points2D | Sequence[PointLike2D]

Point3D: TypeAlias = np.ndarray[_VectorShape3D, _NumpyFloat]
PointLike3D: TypeAlias = Point3D | Sequence[float]

Points3D: TypeAlias = np.ndarray[_BatchVectorShape3D, _NumpyFloat]
PointsLike3D: TypeAlias = Points3D | Sequence[PointLike3D]

PointTransformFunction: TypeAlias = Callable[[PointLike3D], Point3D]

_NumQuad2D: TypeAlias = Literal[16]
_NumQuad3D: TypeAlias = Literal[12]

_QuadShape2D: TypeAlias = tuple[_NumQuad2D]
_BatchQuadShape2D: TypeAlias = tuple[int, *_QuadShape2D]
_QuadShape3D: TypeAlias = tuple[_NumQuad3D]
_BatchQuadShape3D: TypeAlias = tuple[int, *_QuadShape3D]

_Jacobian2D: TypeAlias = np.ndarray[_QuadShape2D, _NumpyFloat]
_JacobianLike2D: TypeAlias = _Jacobian2D | Sequence[float]

Jacobians2D: TypeAlias = np.ndarray[_BatchQuadShape2D, _NumpyFloat]
JacobiansLike2D: TypeAlias = Jacobians2D | Sequence[_JacobianLike2D]

_Jacobian3D: TypeAlias = np.ndarray[_QuadShape3D, _NumpyFloat]
_JacobianLike3D: TypeAlias = _Jacobian3D | Sequence[float]

Jacobians3D: TypeAlias = np.ndarray[_BatchQuadShape3D, _NumpyFloat]
JacobiansLike3D: TypeAlias = Jacobians3D | Sequence[_JacobianLike3D]

_BatchQuadVectorShape2D: TypeAlias = tuple[*_BatchQuadShape2D, *_VectorShape2D]
_BatchQuadVectorShape3D: TypeAlias = tuple[*_BatchQuadShape3D, *_VectorShape3D]

QuadPoints2D: TypeAlias = np.ndarray[_BatchQuadVectorShape2D, _NumpyFloat]
QuadPoints3D: TypeAlias = np.ndarray[_BatchQuadVectorShape3D, _NumpyFloat]

QuadVectors2D: TypeAlias = np.ndarray[_BatchQuadVectorShape2D, _NumpyFloat]
QuadVectors3D: TypeAlias = np.ndarray[_BatchQuadVectorShape3D, _NumpyFloat]

_LineShape: TypeAlias = tuple[Literal[2, 4]]
_BatchLineShape: TypeAlias = tuple[int, *_LineShape]

_TriangleShape: TypeAlias = tuple[Literal[3, 6]]
_BatchTriangleShape: TypeAlias = tuple[int, *_TriangleShape]

Line: TypeAlias = np.ndarray[_LineShape, _NumpyInt]
LineLike: TypeAlias = Line | Sequence[int]

Lines: TypeAlias = np.ndarray[_BatchLineShape, _NumpyInt]
LinesLike: TypeAlias = Lines | Sequence[LineLike]

Triangle: TypeAlias = np.ndarray[_TriangleShape, _NumpyInt]
TriangleLike: TypeAlias = Triangle | Sequence[int]

Triangles: TypeAlias = np.ndarray[_BatchTriangleShape, _NumpyInt]
TrianglesLike: TypeAlias = Triangles | Sequence[TriangleLike]

_QuadShape: TypeAlias = tuple[Literal[5]]
_BatchQuadShape: TypeAlias = tuple[int, *_QuadShape]

Quad: TypeAlias = np.ndarray[_QuadShape, _NumpyInt]
QuadLike: TypeAlias = Quad | Sequence[int]

Quads: TypeAlias = np.ndarray[_BatchQuadShape, _NumpyInt]
QuadsLike: TypeAlias = Quads | Sequence[QuadLike]

_LineVerticesShape: TypeAlias = tuple[*_LineShape, *_VectorShape3D]
_BatchLineVerticesShape: TypeAlias = tuple[int, *_LineVerticesShape]

_TriangleVerticesShape: TypeAlias = tuple[*_TriangleShape, *_VectorShape3D]
_BatchTriangleVerticesShape: TypeAlias = tuple[int, *_TriangleVerticesShape]

LineVertices: TypeAlias = np.ndarray[_LineVerticesShape, _NumpyFloat]
LinesVertices: TypeAlias = np.ndarray[_BatchLineVerticesShape, _NumpyFloat]

TriangleVertices: TypeAlias = np.ndarray[_TriangleVerticesShape, _NumpyFloat]
TrianglesVertices: TypeAlias = np.ndarray[_BatchTriangleVerticesShape, _NumpyFloat]

_BoundShape: TypeAlias = tuple[Literal[2]]

_BatchBoundShape2D: TypeAlias = tuple[Literal[2], Literal[2]]
_BatchBoundShape3D: TypeAlias = tuple[Literal[3], Literal[2]]

_Bound: TypeAlias = np.ndarray[_BoundShape, _NumpyFloat]
_BoundLike: TypeAlias = _Bound | Sequence[float]

Bounds2D: TypeAlias = np.ndarray[_BatchBoundShape2D, _NumpyFloat]
BoundsLike2D: TypeAlias = Bounds2D | Sequence[_BoundLike]

Bounds3D: TypeAlias = np.ndarray[_BatchBoundShape3D, _NumpyFloat]
BoundsLike3D: TypeAlias = Bounds3D | Sequence[_BoundLike]

ActiveLines: TypeAlias = tuple[LinesVertices, Mapping[str, ArrayInt1D]]
ActiveTriangles: TypeAlias = tuple[TrianglesVertices, Mapping[str, ArrayInt1D]]
