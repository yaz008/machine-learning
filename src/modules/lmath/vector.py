from collections.abc import Callable, Iterable
from math import sqrt

from modules.lmath.matrix import is_dtype, Matrix, render_numeric


def vector[NumericT: (
    bool,
    int,
    float,
    complex,
)](
    iterable: Iterable[NumericT],
    render_numeric: Callable[[NumericT, NumericT], str] = render_numeric,
) -> Matrix[NumericT]:
    values: tuple[NumericT, ...] = tuple(iterable)
    return Matrix(
        values,
        shape=(len(values), 1),
        nocopy=True,
        transposed=False,
        render_numeric=render_numeric,
    )


def is_vector[NumericT: (bool, int, float, complex)](matrix: Matrix[NumericT]) -> bool:
    return any(dim == 1 for dim in matrix.shape)


def is_row_vector[NumericT: (
    bool,
    int,
    float,
    complex,
)](vector: Matrix[NumericT]) -> bool:
    return vector.shape.m == 1 and vector.shape.n >= 1


def is_col_vector[NumericT: (
    bool,
    int,
    float,
    complex,
)](vector: Matrix[NumericT]) -> bool:
    return vector.shape.n == 1 and vector.shape.m >= 1


def as_row_vector[NumericT: (
    bool,
    int,
    float,
    complex,
)](vector: Matrix[NumericT]) -> Matrix[NumericT]:
    if not is_vector(vector):
        raise ValueError(f'Cannot interpret {vector.shape} as row vector')
    return vector if is_row_vector(vector) else vector.T


def as_col_vector[NumericT: (
    bool,
    int,
    float,
    complex,
)](vector: Matrix[NumericT]) -> Matrix[NumericT]:
    if not is_vector(vector):
        raise ValueError(f'Cannot interpret {vector.shape} as colon vector')
    return vector if is_col_vector(vector) else vector.T


def norm[NumericT: (
    bool,
    int,
    float,
    complex,
)](vector: Matrix[NumericT]) -> float:
    if not is_vector(vector):
        raise ValueError(f'norm({vector.shape}) is not defined')
    row_vector: Matrix[NumericT] = as_row_vector(vector)
    return sqrt(float(sum(abs(row_vector[0, i]) ** 2 for i in range(len(vector)))))


def dot[NumericT: (
    bool,
    int,
    float,
    complex,
)](v1: Matrix[NumericT], v2: Matrix[NumericT]) -> NumericT:
    if not is_vector(v1) or not is_vector(v2) or len(v1) != len(v2):
        raise ValueError(f'dot({v1.shape}, {v2.shape}) is not defined')
    if is_dtype(v1, complex) and is_dtype(v2, complex):
        (as_row_vector(v1.apply(lambda c: c.conjugate())) @ as_col_vector(v2))[0, 0]
    return (as_row_vector(v1) @ as_col_vector(v2))[0, 0]
