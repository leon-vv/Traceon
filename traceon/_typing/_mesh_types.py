from __future__ import annotations

from typing import Literal, TypeAlias
from collections.abc import Sequence, Mapping
import numpy as np

from ._array_like import _NumpyFloat, _NumpyInt, ArrayInt1D
from ._geometric_types import _VectorShape2D, _VectorShape3D

# Index-Based
_LineShape: TypeAlias = tuple[Literal[2, 4]]  # `(2,)` for regular and `(4,)` for higher-order lines
_BatchLineShape: TypeAlias = tuple[int, *_LineShape]  # `(N, 2)` for regular and `(N, 4)` for higher-order lines

_TriangleShape: TypeAlias = tuple[Literal[3, 6]]  # `(3,)` for regular and `(6,)` for higher-order triangles
_BatchTriangleShape: TypeAlias = tuple[int, *_TriangleShape]  # `(N, 3)` for regular and `(N, 6)` for higher-order triangles

_QuadShape: TypeAlias = tuple[Literal[5]]  # `(5,)`: `(depth, i0, i1, j0, j1)`
_BatchQuadShape: TypeAlias = tuple[int, *_QuadShape]  # `(N, 5)`

Line: TypeAlias = np.ndarray[_LineShape, _NumpyInt]
"""Index representation of a line 
as `(2,)` or `(4,)` NumPy array of `int` 
for regular and higher-order lines respectively."""

LineLike: TypeAlias = Line | Sequence[int]
"""Index representation of a line 
as `(2,)` or `(4,)` NumPy array or sequence of `int`
for regular and higher-order lines respectively."""

Lines: TypeAlias = np.ndarray[_BatchLineShape, _NumpyInt]
"""Index representation of a batch of lines 
as `(N, 2)` or `(N, 4)` NumPy array of `int` 
for regular and higher-order lines respectively."""

LinesLike: TypeAlias = Lines | Sequence[LineLike]
"""Index representation of a batch of lines 
as `(N, 2)` or `(N, 4)` NumPy array or sequence of `LineLike` objects 
for regular and higher-order lines respectively."""

Triangle: TypeAlias = np.ndarray[_TriangleShape, _NumpyInt]
"""Index representation of a triangle as `(3,)` or `(6,)` NumPy array of `int` 
for regular and higher-order triangles respectively."""

TriangleLike: TypeAlias = Triangle | Sequence[int]
"""Index representation of a triangle 
as `(3,)` or `(6,)` NumPy array or sequence of `int`
for regular and higher-order triangles respectively."""

Triangles: TypeAlias = np.ndarray[_BatchTriangleShape, _NumpyInt]
"""Index representation of a batch of triangles 
as `(N, 3)` or `(N, 6)` NumPy array of `int` 
for regular and higher-order triangles respectively."""

TrianglesLike: TypeAlias = Triangles | Sequence[TriangleLike]
"""Index representation of a batch of triangles 
as `(N, 3)` or `(N, 6)` NumPy array of `int` or sequence of `TriangleLike` objects
for regular and higher-order triangles respectively."""

Quad: TypeAlias = np.ndarray[_QuadShape, _NumpyInt]
"""Index representation of quadrilaterals `(depth, i0, i1, j1, j2)` 
as `(5,)` NumPy array of `int`."""

QuadLike: TypeAlias = Quad | Sequence[int]
"""
Index representation of quadrilaterals `(depth, i0, i1, j1, j2)` 
as `(5,)` NumPy array or sequence of `int`.
"""

Quads: TypeAlias = np.ndarray[_BatchQuadShape, _NumpyInt]
"""
Index representation of a batch of quadrilaterals `(depth, i0, i1, j1, j2)` 
as `(N, 5)` NumPy array of `int`.
"""
QuadsLike: TypeAlias = Quads | Sequence[QuadLike]
"""
Index representation of a batch of quadrilaterals `(depth, i0, i1, j1, j2)`
as `(N, 5)` NumPy array of `int` or sequence of `QuadLike` objects.
"""

# Vertex-based
_LineVerticesShape: TypeAlias = tuple[*_LineShape, *_VectorShape3D]  # `(2, 3)` or `(4, 3)`
_BatchLineVerticesShape: TypeAlias = tuple[int, *_LineVerticesShape]  # `(N, 2, 3)` or `(N, 4, 3)`

_TriangleVerticesShape: TypeAlias = tuple[*_TriangleShape, *_VectorShape3D]  # `(3, 3)` or `(6, 3)`
_BatchTriangleVerticesShape: TypeAlias = tuple[int, *_TriangleVerticesShape]  # `(N, 3, 3)` or `(N, 6, 3)`

LineVertices: TypeAlias = np.ndarray[_LineVerticesShape, _NumpyFloat]
"""Vertex representation of a line 
as `(2, 3)` or `(4, 3)` NumPy array of `float` 
for regular and higher-order lines respectively."""

LinesVertices: TypeAlias = np.ndarray[_BatchLineVerticesShape, _NumpyFloat]
"""Vertex representation of a batch of lines 
as `(N, 2, 3)` or `(N, 4, 3)` NumPy array of `float` 
for regular and higher-order lines respectively."""

TriangleVertices: TypeAlias = np.ndarray[_TriangleVerticesShape, _NumpyFloat]
"""Vertex representation of a triangle 
as `(3, 3)` or `(6, 3)` NumPy array of `float` 
for regular and higher-order triangles respectively."""

TrianglesVertices: TypeAlias = np.ndarray[_BatchTriangleVerticesShape, _NumpyFloat]
"""Vertex representation of a batch of triangles 
as `(N, 3, 3)` or `(N, 6, 3)` NumPy array of `float` 
for regular and higher-order triangles respectively."""

ActiveLines: TypeAlias = tuple[LinesVertices, Mapping[str, ArrayInt1D]]
"""Active lines in a mesh 
as a tuple of `LinesVertices`, an `(N, 4, 3)` NumPy array of `float` representing the line, 
and a dictionary mapping the names of the physical groups to an `(N,)` NumPy array of `int` 
representing the line indices."""

ActiveTriangles: TypeAlias = tuple[TrianglesVertices, Mapping[str, ArrayInt1D]]
"""Active triangles in a mesh 
as a tuple of `TrianglesVertices`, an `(N, 3, 3)` NumPy array of `float` points representing the triangles, 
and a dictionary mapping the names of the physical groups to an `(N,)` NumPy array of `int` 
representing the triangle indices."""