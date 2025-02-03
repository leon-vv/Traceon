from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple, Dict, Union, Sequence, Callable, Mapping, Iterator, Generator, cast, TypeVar, Generic
from typing_extensions import TypeAlias, Self

import numpy as np

from numpy.typing import NDArray


# T = TypeVar("T", bound=tuple[int, ...])  # Covariant shape

# class ArrayFloat(Generic[T]):
#     def __class_getitem__(cls, shape: T) -> type[np.ndarray[T, np.dtype[np.floating]]]:
#         return np.ndarray[shape, np.dtype[np.floating]]



if TYPE_CHECKING:
    #: 0. Shape and dtype definitions
    _NumpyFloat: TypeAlias = np.dtype[np.floating]
    _NumpyInt: TypeAlias = np.dtype[np.integer]

    _ShapeLike: TypeAlias = Tuple[int, ...]

    _Shape1D: TypeAlias = Tuple[int, ...]
    _Shape2D: TypeAlias = Tuple[int, ...]
    _Shape3D: TypeAlias = Tuple[int, ...]


    #: 1. Array and array-like type-aliases

    #: 1.1 Floating-point

    #: 1.1.1 Array
    ArrayFloat: TypeAlias = np.ndarray[_ShapeLike, _NumpyFloat]
    ArrayFloat1D: TypeAlias = np.ndarray[_Shape1D, _NumpyFloat]
    ArrayFloat2D: TypeAlias = np.ndarray[_Shape2D, _NumpyFloat]
    ArrayFloat3D: TypeAlias = np.ndarray[_Shape3D, _NumpyFloat]

    #: 1.1.2 Array-like
    ArrayLikeFloat: TypeAlias = ArrayFloat | Sequence[float]
    """A NumPy array of `float` with arbitrary shape 
    or a sequence of `float`."""

    ArrayLikeFloat1D: TypeAlias = ArrayFloat1D | Sequence[float]
    """A one-dimensional NumPy array of `float` 
    or a sequence of `float`."""

    ArrayLikeFloat2D: TypeAlias = ArrayFloat2D | Sequence[ArrayLikeFloat1D]
    """A two-dimensional NumPy array of `float`
    or a sequence of one-dimensional `ArrayLikeFloat1D`."""

    ArrayLikeFloat3D: TypeAlias = ArrayFloat3D | Sequence[ArrayLikeFloat2D]
    """A three-dimensional NumPy array of `float` 
    or a sequence of `ArrayLikeFloat2D`."""


    #: 1.2 Integer

    #: 1.2.1 Array
    ArrayInt: TypeAlias = np.ndarray[_ShapeLike, _NumpyInt]
    ArrayInt1D: TypeAlias = np.ndarray[_Shape1D, _NumpyInt]
    ArrayInt2D: TypeAlias = np.ndarray[_Shape2D, _NumpyInt]
    ArrayInt3D: TypeAlias = np.ndarray[_Shape3D, _NumpyInt]

    #: 1.2.2 Array-like
    ArrayLikeInt: TypeAlias = ArrayInt | Sequence[int]
    """A NumPy array of `int` with arbitrary shape 
    or a sequence of `int`."""

    ArrayLikeInt1D: TypeAlias = ArrayInt1D | Sequence[int]
    """A one-dimensional NumPy array of `int` 
    or a sequence of `int`."""

    ArrayLikeInt2D: TypeAlias = ArrayInt2D | Sequence[ArrayLikeInt1D]
    """A two-dimensional NumPy array of `int` 
    or a sequence of `ArrayLikeInt1D`."""

    ArrayLikeInt3D: TypeAlias = ArrayInt3D | Sequence[ArrayLikeInt2D]
    """A three-dimensional NumPy array of `int`
    or a sequence of `ArrayLikeInt2D`."""


    #: 2. Geometric type-aliases

    #: 2.1 Vectors 

    #: 2.1.1 Two-dimensional
    Vector2D: TypeAlias = ArrayFloat1D
    """Two-dimensional vector 
    as `(2,)` NumPy array of `float`."""

    VectorLike2D: TypeAlias = ArrayLikeFloat1D
    """Two-dimensional vector 
    as `(2,)` NumPy array or sequence of `float`."""

    Vectors2D: TypeAlias = ArrayFloat2D
    """Batch of two-dimensional vectors 
    as `(N, 2)` NumPy array of `float`."""

    VectorsLike2D: TypeAlias = ArrayLikeFloat2D
    """Batch of two-dimensional vectors 
    as `(N, 2)` NumPy array of `float` or sequence of `VectorLike2D` objects."""


    #: 2.1.2 Three-dimensional
    Vector3D: TypeAlias = ArrayFloat1D
    """Three-dimensional vector 
    as `(3,)` NumPy array of `float`."""

    VectorLike3D: TypeAlias = ArrayLikeFloat1D
    """Three-dimensional vector 
    as `(3,)` NumPy array or sequence of `float`."""

    Vectors3D: TypeAlias = ArrayFloat2D
    """Batch of three-dimensional vectors 
    as `(N, 3)` NumPy array of `float`."""

    VectorsLike3D: TypeAlias = ArrayLikeFloat2D
    """Batch of three-dimensional vectors 
    as `(N, 3)` NumPy array of `float` or sequence of `VectorLike3D` objects."""


    #: 2.2 Points

    #: 2.2.1 Two-dimensional
    Point2D: TypeAlias = ArrayFloat1D
    """Two-dimensional point 
    as `(2,)` NumPy array of `float`."""

    PointLike2D: TypeAlias = ArrayLikeFloat1D
    """Two-dimensional point 
    as `(2,)` NumPy array or sequence of `float`."""

    Points2D: TypeAlias = ArrayFloat2D
    """Batch of two-dimensional points 
    as `(N, 2)` NumPy array of `float`."""

    PointsLike2D: TypeAlias = ArrayLikeFloat2D
    """Batch of two-dimensional points 
    as `(N, 2)` NumPy array of `float` or sequence of two-dimensional `Pointlike2D` objects."""


    #: 2.2.2 Three-dimensional
    Point3D: TypeAlias = ArrayFloat1D
    """A three-dimensional point 
    as `(3,)` NumPy array of `float`."""

    PointLike3D: TypeAlias = ArrayLikeFloat1D
    """A three-dimensional point 
    as `(3,)` NumPy array or sequence of `float`."""

    Points3D: TypeAlias = ArrayFloat2D
    """Batch of three-dimensional points 
    as `(N, 3)` NumPy array of `float`."""

    PointsLike3D: TypeAlias = ArrayLikeFloat2D
    """Batch of three-dimensional points 
    as `(N, 3)` NumPy array of `float` or sequence of `PointLike3D` objects."""


    #: 2.3 Bounds

    #: 2.3.1 Two-dimensional
    Bounds2D: TypeAlias = ArrayFloat2D
    """Two-dimensional bounds `[(x_min, x_max), (y_min, y_max)]` 
    as `(2, 2)` NumPy array of `float`."""

    BoundsLike2D: TypeAlias = ArrayLikeFloat2D
    """Two-dimensional bounds `[(x_min, x_max), (y_min, y_max)]` 
    as `(2, 2)` NumPy array of `float` or a sequence of  objects."""


    #: 2.3.2 Three-dimensional
    Bounds3D: TypeAlias = ArrayFloat2D
    """Three-dimensional bounds `[(x_min, x_max), (y_min, y_max), (z_min, z_max)]` 
    as `(3, 2)` NumPy array of `float`."""

    BoundsLike3D: TypeAlias = ArrayLikeFloat2D
    """Three-dimensional bounds `[(x_min, x_max), (y_min, y_max), (z_min, z_max)]` 
    as `(3, 2)` NumPy array of `float` or a sequence of `_BoundLike` objects."""


    #: 3. Mesh elements type-aliases

    #: 3.1 Index-based elements

    #: 3.1.1 Lines
    Line: TypeAlias = ArrayInt1D
    """Index representation of a line 
    as `(2,)` or `(4,)` NumPy array of `int` 
    for regular and higher-order lines respectively."""

    LineLike: TypeAlias = ArrayLikeInt1D
    """Index representation of a line 
    as `(2,)` or `(4,)` NumPy array or sequence of `int`
    for regular and higher-order lines respectively."""

    Lines: TypeAlias = ArrayInt2D
    """Index representation of a batch of lines 
    as `(N, 2)` or `(N, 4)` NumPy array of `int` 
    for regular and higher-order lines respectively."""

    LinesLike: TypeAlias = ArrayLikeInt2D
    """Index representation of a batch of lines 
    as `(N, 2)` or `(N, 4)` NumPy array or sequence of `LineLike` objects
    for regular and higher-order lines respectively."""


    #: 3.1.2 Triangles
    Triangle: TypeAlias = ArrayInt1D
    """Index representation of a triangle as `(3,)` or `(6,)` NumPy array of `int` 
    for regular and higher-order triangles respectively."""

    TriangleLike: TypeAlias = ArrayLikeInt1D
    """Index representation of a triangle 
    as `(3,)` or `(6,)` NumPy array or sequence of `int`
    for regular and higher-order triangles respectively."""

    Triangles: TypeAlias = ArrayInt2D
    """Index representation of a batch of triangles 
    as `(N, 3)` or `(N, 6)` NumPy array of `int` 
    for regular and higher-order triangles respectively."""

    TrianglesLike: TypeAlias = ArrayLikeInt2D
    """Index representation of a batch of triangles 
    as `(N, 3)` or `(N, 6)` NumPy array of `int` or sequence of `TriangleLike` objects
    for regular and higher-order triangles respectively."""


    #: 3.1.3 Quads
    Quad: TypeAlias = ArrayInt1D
    """Index representation of quadrilaterals `(depth, i0, i1, j1, j2)` 
    as `(5,)` NumPy array of `int`."""

    QuadLike: TypeAlias = ArrayLikeInt1D
    """
    Index representation of quadrilaterals `(depth, i0, i1, j1, j2)` 
    as `(5,)` NumPy array or sequence of `int`.
    """

    Quads: TypeAlias = ArrayInt2D
    """
    Index representation of a batch of quadrilaterals `(depth, i0, i1, j1, j2)` 
    as `(N, 5)` NumPy array of `int`.
    """
    QuadsLike: TypeAlias = ArrayLikeInt2D
    """
    Index representation of a batch of quadrilaterals `(depth, i0, i1, j1, j2)`
    as `(N, 5)` NumPy array of `int` or sequence of `QuadLike` objects.
    """


    #: 3.2 Vertex-based elements

    #: 3.2.1 Lines
    LineVertices: TypeAlias = ArrayFloat2D
    """Vertex representation of a line 
    as `(2, 3)` or `(4, 3)` NumPy array of `float` 
    for regular and higher-order lines respectively."""

    LinesVertices: TypeAlias = ArrayFloat3D
    """Vertex representation of a batch of lines 
    as `(N, 2, 3)` or `(N, 4, 3)` NumPy array of `float` 
    for regular and higher-order lines respectively."""

    #: 3.2.2 Triangles
    TriangleVertices: TypeAlias = ArrayFloat2D
    """Vertex representation of a triangle 
    as `(3, 3)` or `(6, 3)` NumPy array of `float` 
    for regular and higher-order triangles respectively."""

    TrianglesVertices: TypeAlias = ArrayFloat3D
    """Vertex representation of a batch of triangles 
    as `(N, 3, 3)` or `(N, 6, 3)` NumPy array of `float` 
    for regular and higher-order triangles respectively."""

    #: 3.3.3 Active elements
    ActiveLines: TypeAlias = Tuple[LinesVertices, Mapping[str, ArrayInt1D]]
    """Active lines in a mesh 
    as a tuple of `LinesVertices`, an `(N, 4, 3)` NumPy array of `float` representing the line, 
    and a dictionary mapping the names of the physical groups to an `(N,)` NumPy array of `int` 
    representing the line indices."""

    ActiveTriangles: TypeAlias = Tuple[TrianglesVertices, Mapping[str, ArrayInt1D]]
    """Active triangles in a mesh 
    as a tuple of `TrianglesVertices`, an `(N, 3, 3)` NumPy array of `float` points representing the triangles, 
    and a dictionary mapping the names of the physical groups to an `(N,)` NumPy array of `int` 
    representing the triangle indices."""

    from .geometry import Path, PathCollection, Surface, SurfaceCollection
    from .mesher import Mesh
    from .excitation import Excitation
    from .field import EffectivePointCharges, Field, FieldBEM, FieldRadialBEM
    try:
        from traceon_pro.field import Field3DBEM  # type: ignore
    except ImportError:
        Field3DBEM = None  # Fallback for unavailable import
