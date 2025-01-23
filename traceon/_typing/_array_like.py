from __future__ import annotations
import sys
from collections.abc import Sequence

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

import numpy as np

_NumpyFloat: TypeAlias = np.dtype[np.floating]
_NumpyInt: TypeAlias = np.dtype[np.integer]

_ShapeLike: TypeAlias = tuple[int, ...]

_Shape1D: TypeAlias = tuple[int]
_Shape2D: TypeAlias = tuple[int, int]
_Shape3D: TypeAlias = tuple[int, int, int]
_Shape4D: TypeAlias = tuple[int, int, int, int]
_Shape5D: TypeAlias = tuple[int, int, int, int, int]


ArrayFloat: TypeAlias = np.ndarray[_ShapeLike, _NumpyFloat]
"""A NumPy array of `float` with arbitrary shape."""

ArrayFloat1D: TypeAlias = np.ndarray[_Shape1D, _NumpyFloat]
"""A one-dimensional NumPy array of `float`."""

ArrayFloat2D: TypeAlias = np.ndarray[_Shape2D, _NumpyFloat]
"""A two-dimensional NumPy array of `float`."""

ArrayFloat3D: TypeAlias = np.ndarray[_Shape3D, _NumpyFloat]
"""A three-dimensional NumPy array of `float`."""

ArrayFloat4D: TypeAlias = np.ndarray[_Shape4D, _NumpyFloat]
"""A four-dimensional NumPy array of `float`."""

ArrayFloat5D: TypeAlias = np.ndarray[_Shape5D, _NumpyFloat]
"""A five-dimensional NumPy array of `float`."""


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

ArrayLikeFloat4D: TypeAlias = ArrayFloat4D | Sequence[ArrayLikeFloat3D]
"""A four-dimensional NumPy array of `float` 
or a sequence of `ArrayLikeFloat3D`."""

ArrayLikeFloat5D: TypeAlias = ArrayFloat5D | Sequence[ArrayLikeFloat4D]
"""A five-dimensional NumPy array of `float` 
or a sequence of `ArrayLikeFloat4D`."""

ArrayInt: TypeAlias = np.ndarray[_ShapeLike, _NumpyInt]
"""A NumPy array of `int` with arbitrary shape."""

ArrayInt1D: TypeAlias = np.ndarray[_Shape1D, _NumpyInt]
"""A one-dimensional NumPy array of `int`."""

ArrayInt2D: TypeAlias = np.ndarray[_Shape2D, _NumpyInt]
"""A two-dimensional NumPy array of `int`."""

ArrayInt3D: TypeAlias = np.ndarray[_Shape3D, _NumpyInt]
"""A three-dimensional NumPy array of `int`."""

ArrayInt4D: TypeAlias = np.ndarray[_Shape4D, _NumpyInt]
"""A four-dimensional NumPy array of `int`."""

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

ArrayLikeInt4D: TypeAlias = ArrayInt4D | Sequence[ArrayLikeInt3D]
"""A four-dimensional NumPy array of `int`
or a sequence of `ArrayLikeInt3D`."""
