from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from functools import reduce
from itertools import chain, product, repeat
from operator import add
from operator import and_ as bitwise_and
from operator import matmul, mul
from operator import or_ as bitwise_or
from operator import sub, xor
from typing import cast, Literal, NamedTuple, overload, TypeGuard
import builtins


def render_numeric[NumericT: (
    bool,
    int,
    float,
    complex,
)](x: NumericT, longest: NumericT) -> str:
    match type(x):
        case builtins.bool:
            return '1' if x else '0'
        case builtins.int:
            return f'{x:>{len(str(longest))}}'
        case builtins.float:
            whole, precision = map(len, str(longest).split(sep='.', maxsplit=1))
            return f'{x:>{(whole + precision + 1)}.{precision}f}'
        case builtins.complex:
            length: int = len(str(longest).strip('()'))
            return f'{x:>{length}.{max(
                map(
                    lambda f: len(str(f).split(sep='.', maxsplit=1)[1]),
                    (x.real, x.imag),
                )
            )}f}'
        case _:
            raise TypeError(f'Matrix[{type(x).__name__}] is not allowed')


class MatrixShape(NamedTuple):
    m: int
    n: int

    @property
    def size(self) -> int:
        return self.m * self.n

    @property
    def T(self) -> MatrixShape:
        return MatrixShape(m=self.n, n=self.m)

    def __repr__(self) -> str:
        return f'{self.m} by {self.n}'


@dataclass(slots=True)
class Matrix[NumericT: (bool, int, float, complex)]:
    __values: tuple[NumericT, ...] = field(init=False)
    __shape: MatrixShape
    __render: Callable[[NumericT, NumericT], str]
    __is_transposed: bool

    @overload
    def __init__(
        self,
        values: tuple[NumericT, ...],
        /,
        shape: tuple[int, int],
        *,
        nocopy: Literal[True],
        transposed: bool = False,
        render_numeric: Callable[[NumericT, NumericT], str] = render_numeric,
    ) -> None:
        raise NotImplementedError

    @overload
    def __init__(
        self,
        values: Iterable[NumericT],
        /,
        shape: tuple[int, int],
        *,
        render_numeric: Callable[[NumericT, NumericT], str] = render_numeric,
    ) -> None:
        raise NotImplementedError

    @overload
    def __init__(
        self,
        values: Sequence[Sequence[NumericT]],
        /,
        *,
        render_numeric: Callable[[NumericT, NumericT], str] = render_numeric,
    ) -> None:
        raise NotImplementedError

    def __init__(
        self,
        values: (
            Iterable[NumericT] | tuple[NumericT, ...] | Sequence[Sequence[NumericT]]
        ),
        /,
        shape: tuple[int, int] | None = None,
        *,
        nocopy: bool = False,
        transposed: bool = False,
        render_numeric: Callable[[NumericT, NumericT], str] = render_numeric,
    ) -> None:
        self.__is_transposed = transposed
        self.__render = render_numeric
        if shape is None:
            self.__shape = MatrixShape(
                m=len(cast(Sequence[Sequence[NumericT]], values)),
                n=(
                    len(cast(Sequence[Sequence[NumericT]], values)[0])
                    if len(cast(Sequence[Sequence[NumericT]], values)) > 0
                    else 0
                ),
            )
            if not all(
                cast(Sequence[Sequence[NumericT]], values)[i] == self.shape.n
                for i in range(self.shape.m)
            ):
                ValueError('Matrix rows have different lengths')
            self.__values = tuple(chain(*cast(Sequence[Sequence[NumericT]], values)))
        else:
            self.__shape = MatrixShape(*shape)
            self.__values = (
                cast(tuple[NumericT, ...], values)
                if nocopy
                else tuple(cast(Iterable[NumericT], values))
            )
        if self.shape.size == 0:
            raise ValueError(f'Cannot create {self.shape} matrix')
        if len(self.__values) != self.shape.size:
            raise ValueError(
                f'Cannot interpret {len(self.__values)} values as {self.shape} matrix'
            )

    def __repr__(self) -> str:
        longest: NumericT = max(
            self.__values, key=lambda x: max(len(str(abs(x))), len(str(x)))
        )
        return f'[{'\n'.join(
            f'{' ' * (i != 0)}[{' '.join(
                self.__render(self[i, j], longest) for j in range(self.shape.n)
            )}]'
            for i in range(self.shape.m)
        )}]'

    def __len__(self) -> int:
        return len(self.__values)

    def __getitem__(self, __index: tuple[int, int], /) -> NumericT:
        return self.__values[
            (
                __index[self.__is_transposed]
                * (self.shape.m if self.__is_transposed else self.shape.n)
                + __index[1 - self.__is_transposed]
            )
        ]

    def __add__(self, __other: Matrix[NumericT], /) -> Matrix[NumericT]:
        return self.combine(__other, func=(xor if self.dtype is bool else add))

    def __iadd__(self, __other: Matrix[NumericT], /) -> Matrix[NumericT]:
        return self.__add__(__other)

    def __sub__(self, __other: Matrix[NumericT], /) -> Matrix[NumericT]:
        return self.combine(__other, func=(xor if self.dtype is bool else sub))

    def __isub__(self, __other: Matrix[NumericT], /) -> Matrix[NumericT]:
        return self.__sub__(__other)

    def __mul__(self, __other: Matrix[NumericT], /) -> Matrix[NumericT]:
        return self.combine(__other, func=(bitwise_and if self.dtype is bool else mul))

    def __imul__(self, __other: Matrix[NumericT], /) -> Matrix[NumericT]:
        return self.__mul__(__other)

    def __matmul__(self, __other: Matrix[NumericT], /) -> Matrix[NumericT]:
        if self.shape.n != __other.shape.m:
            raise ValueError(
                f'Cannot multiply {self.shape} and {__other.shape} matrices'
            )
        star: Callable[[NumericT, NumericT], NumericT] = (
            bitwise_and if self.dtype is bool else mul
        )
        plus: Callable[[NumericT, NumericT], NumericT] = (
            bitwise_or if self.dtype is bool else add
        )
        return Matrix(
            tuple(
                reduce(
                    plus,
                    (star(self[i, k], __other[k, j]) for k in range(self.shape.n)),
                )
                for i, j in product(range(self.shape.m), range(__other.shape.n))
            ),
            shape=(self.shape.m, __other.shape.n),
            nocopy=True,
            transposed=False,
            render_numeric=self.__render,
        )

    def __imatmul__(self, __other: Matrix[NumericT], /) -> Matrix[NumericT]:
        return self.__matmul__(__other)

    def __pow__(self, __power: int) -> Matrix[NumericT]:
        return reduce(matmul, repeat(self, __power), eye(self.shape.n, self.dtype))

    def __ipow__(self, __power: int, /) -> Matrix[NumericT]:
        return self.__pow__(__power)

    def __rmul__(self, __number: NumericT) -> Matrix[NumericT]:
        return self.apply(
            cast(
                Callable[[NumericT], NumericT],
                lambda x: (bitwise_and if self.dtype is bool else mul)(x, __number),
            )
        )

    def __eq__(self, __other: object, /) -> bool:
        return (
            isinstance(__other, Matrix)
            and self.shape == __other.shape
            and self.__values == __other.__values
        )

    def __ne__(self, __other: object, /) -> bool:
        return not self.__eq__(__other)

    def combine(
        self,
        __other: Matrix[NumericT],
        /,
        func: Callable[[NumericT, NumericT], NumericT],
    ) -> Matrix[NumericT]:
        if not self.shape == __other.shape:
            raise ValueError(
                f'Cannot apply {func.__name__} on {(
                    self.shape
                )} and {__other.shape} matrices'
            )
        return Matrix(
            tuple(
                func(self[i, j], __other[i, j])
                for i, j in product(range(self.shape.m), range(self.shape.n))
            ),
            shape=self.shape,
            nocopy=True,
            transposed=False,
            render_numeric=self.__render,
        )

    def apply(self, func: Callable[[NumericT], NumericT]) -> Matrix[NumericT]:
        return Matrix(
            tuple(
                func(self[i, j])
                for i, j in product(range(self.shape.m), range(self.shape.n))
            ),
            shape=self.shape,
            nocopy=True,
            transposed=False,
            render_numeric=self.__render,
        )

    @property
    def dtype(self) -> type[NumericT]:
        return type(self.__values[0])

    @property
    def shape(self) -> MatrixShape:
        return self.__shape.T if self.__is_transposed else self.__shape

    @property
    def T(self) -> Matrix[NumericT]:
        return Matrix(
            self.__values,
            shape=self.__shape,
            nocopy=True,
            transposed=(not self.__is_transposed),
            render_numeric=self.__render,
        )


@overload
def is_dtype[NumericT: (
    bool,
    int,
    float,
    complex,
)](matrix: Matrix[NumericT], dtype: type[bool]) -> TypeGuard[Matrix[bool]]:
    raise NotImplementedError


@overload
def is_dtype[NumericT: (
    bool,
    int,
    float,
    complex,
)](matrix: Matrix[NumericT], dtype: type[int]) -> TypeGuard[Matrix[int]]:
    raise NotImplementedError


@overload
def is_dtype[NumericT: (
    bool,
    int,
    float,
    complex,
)](matrix: Matrix[NumericT], dtype: type[float]) -> TypeGuard[Matrix[float]]:
    raise NotImplementedError


@overload
def is_dtype[NumericT: (
    bool,
    int,
    float,
    complex,
)](matrix: Matrix[NumericT], dtype: type[complex]) -> TypeGuard[Matrix[complex]]:
    raise NotImplementedError


def is_dtype[NumericT: (
    bool,
    int,
    float,
    complex,
)](matrix: Matrix[NumericT], dtype: type[bool | int | float | complex]) -> bool:
    return matrix.dtype is dtype


def const[NumericT: (
    bool,
    int,
    float,
    complex,
)](value: NumericT, shape: tuple[int, int]) -> Matrix[NumericT]:
    return Matrix((value for _ in range(shape[0] * shape[1])), shape=shape)


def zeros[NumericT: (
    bool,
    int,
    float,
    complex,
)](shape: tuple[int, int], dtype: type[NumericT]) -> Matrix[NumericT]:
    return const(value=dtype(0), shape=shape)


def ones[NumericT: (
    bool,
    int,
    float,
    complex,
)](shape: tuple[int, int], dtype: type[NumericT]) -> Matrix[NumericT]:
    return const(value=dtype(1), shape=shape)


def eye[NumericT: (
    bool,
    int,
    float,
    complex,
)](n: int, dtype: type[NumericT]) -> Matrix[NumericT]:
    return Matrix(
        (dtype(i == j) for i, j in product(range(n), range(n))),
        shape=(n, n),
    )
