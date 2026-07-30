from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from ._lib import addr, lib


class ExplicitBitVect:
    _mutation_epoch = 0

    def __init__(self, size: int, bits: Iterable[int] | None = None):
        if size <= 0:
            raise ValueError("bit vector size must be positive")
        self._size = int(size)
        self._words = np.zeros((self._size + 63) // 64, dtype=np.uint64)
        if bits is not None:
            for bit in bits:
                self.SetBit(bit)

    @classmethod
    def _from_words(cls, size: int, words: np.ndarray) -> "ExplicitBitVect":
        size = int(size)
        if size <= 0:
            raise ValueError("bit vector size must be positive")
        values = np.asarray(words)
        if values.dtype != np.dtype(np.uint64):
            raise TypeError("word buffer dtype must be uint64")
        if values.ndim != 1 or len(values) != (size + 63) // 64:
            raise ValueError("word buffer size does not match bit vector size")
        if not values.flags.c_contiguous:
            values = np.ascontiguousarray(values)
        remainder = size & 63
        if remainder and int(values[-1]) >> remainder:
            values = values.copy()
            values[-1] &= (np.uint64(1) << np.uint64(remainder)) - np.uint64(1)
        result = cls.__new__(cls)
        result._size = size
        result._words = values
        owner = values
        while isinstance(owner.base, np.ndarray):
            owner = owner.base
        result._word_owner = owner
        result._word_offset = values.ctypes.data - owner.ctypes.data
        return result

    @classmethod
    def _empty(cls, size: int) -> "ExplicitBitVect":
        result = cls.__new__(cls)
        result._size = int(size)
        result._words = np.empty((result._size + 63) // 64, dtype=np.uint64)
        return result

    def GetNumBits(self) -> int:
        return self._size

    def GetNumOnBits(self) -> int:
        return sum(int(word).bit_count() for word in self._words)

    def GetBit(self, bit: int) -> bool:
        bit = self._check(bit)
        return bool(int(self._words[bit >> 6]) & (1 << (bit & 63)))

    def SetBit(self, bit: int) -> bool:
        bit = self._check(bit)
        old = self.GetBit(bit)
        self._words[bit >> 6] |= np.uint64(1) << np.uint64(bit & 63)
        if not old:
            ExplicitBitVect._mutation_epoch += 1
        return old

    def UnSetBit(self, bit: int) -> bool:
        bit = self._check(bit)
        old = self.GetBit(bit)
        self._words[bit >> 6] &= ~(np.uint64(1) << np.uint64(bit & 63))
        if old:
            ExplicitBitVect._mutation_epoch += 1
        return old

    def GetOnBits(self) -> tuple[int, ...]:
        return tuple(i for i in range(self._size) if self.GetBit(i))

    def ToBitString(self) -> str:
        return "".join("1" if self.GetBit(i) else "0" for i in range(self._size))

    def ToList(self) -> list[int]:
        return [int(self.GetBit(i)) for i in range(self._size)]

    def to_rdkit(self):
        from rdkit.DataStructs import ExplicitBitVect as RDExplicitBitVect

        result = RDExplicitBitVect(self._size)
        for bit in self.GetOnBits():
            result.SetBit(bit)
        return result

    def _check(self, bit: int) -> int:
        bit = int(bit)
        if bit < 0:
            bit += self._size
        if bit < 0 or bit >= self._size:
            raise IndexError(bit)
        return bit

    def _binary(self, other, operation) -> "ExplicitBitVect":
        other = _coerce(other)
        _same_size(self, other)
        return self._from_words(self._size, operation(self._words, other._words))

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, bit: int) -> int:
        return int(self.GetBit(bit))

    def __setitem__(self, bit: int, value) -> None:
        self.SetBit(bit) if value else self.UnSetBit(bit)

    def __and__(self, other) -> "ExplicitBitVect":
        return self._binary(other, np.bitwise_and)

    def __or__(self, other) -> "ExplicitBitVect":
        return self._binary(other, np.bitwise_or)

    def __xor__(self, other) -> "ExplicitBitVect":
        return self._binary(other, np.bitwise_xor)

    def __eq__(self, other) -> bool:
        try:
            other = _coerce(other)
        except (TypeError, ValueError):
            return False
        return self._size == other._size and np.array_equal(self._words, other._words)


def _coerce(fp) -> ExplicitBitVect:
    if isinstance(fp, ExplicitBitVect):
        return fp
    if hasattr(fp, "GetNumBits") and hasattr(fp, "GetOnBits"):
        return ExplicitBitVect(fp.GetNumBits(), fp.GetOnBits())
    raise TypeError("expected an ExplicitBitVect-compatible object")


def _same_size(left: ExplicitBitVect, right: ExplicitBitVect) -> None:
    if left.GetNumBits() != right.GetNumBits():
        raise ValueError("BitVects must be same length")


_METRICS = {
    "tanimoto": 0,
    "dice": 1,
    "cosine": 2,
    "sokal": 3,
    "russel": 4,
    "kulczynski": 5,
    "mcconnaughey": 6,
}


def _similarity(left, right, metric: str, returnDistance: bool = False) -> float:
    left = _coerce(left)
    right = _coerce(right)
    _same_size(left, right)
    score = lib().mrd_similarity(
        addr(left._words),
        addr(right._words),
        len(left._words),
        len(left),
        _METRICS[metric],
    )
    return 1.0 - score if returnDistance else score


def TanimotoSimilarity(bv1, bv2, returnDistance: bool = False) -> float:
    return _similarity(bv1, bv2, "tanimoto", returnDistance)


def DiceSimilarity(bv1, bv2, returnDistance: bool = False) -> float:
    return _similarity(bv1, bv2, "dice", returnDistance)


def CosineSimilarity(bv1, bv2, returnDistance: bool = False) -> float:
    return _similarity(bv1, bv2, "cosine", returnDistance)


def SokalSimilarity(bv1, bv2, returnDistance: bool = False) -> float:
    return _similarity(bv1, bv2, "sokal", returnDistance)


def RusselSimilarity(bv1, bv2, returnDistance: bool = False) -> float:
    return _similarity(bv1, bv2, "russel", returnDistance)


def KulczynskiSimilarity(bv1, bv2, returnDistance: bool = False) -> float:
    return _similarity(bv1, bv2, "kulczynski", returnDistance)


def McConnaugheySimilarity(bv1, bv2, returnDistance: bool = False) -> float:
    return _similarity(bv1, bv2, "mcconnaughey", returnDistance)


def _bulk(query, others, metric: str, returnDistance: bool = False) -> list[float]:
    query = _coerce(query)
    signature = tuple(map(id, others)) if isinstance(others, list) else None
    cache = getattr(query, "_bulk_cache", None)
    if (
        signature is not None
        and cache is not None
        and cache[0] is others
        and cache[1] == signature
        and cache[2] == ExplicitBitVect._mutation_epoch
    ):
        targets = cache[3]
        n_vectors = len(signature)
        n_words = len(query._words)
        scores = np.empty(n_vectors, dtype=np.float64)
        lib().mrd_bulk_similarity(
            addr(query._words),
            addr(targets),
            addr(scores),
            n_vectors,
            n_words,
            len(query),
            _METRICS[metric],
        )
        if returnDistance:
            scores = 1.0 - scores
        return scores.tolist()

    vectors = [_coerce(item) for item in others]
    if not vectors:
        return []
    n_words = len(query._words)
    for vector in vectors:
        if vector._size != query._size:
            raise ValueError("BitVects must be same length")

    first = vectors[0]
    owner = getattr(first, "_word_owner", None)
    start = getattr(first, "_word_offset", -1)
    stride = n_words * np.dtype(np.uint64).itemsize
    contiguous = owner is not None
    if contiguous:
        for row, vector in enumerate(vectors):
            if (
                getattr(vector, "_word_owner", None) is not owner
                or getattr(vector, "_word_offset", -1) != start + row * stride
            ):
                contiguous = False
                break
    if contiguous:
        targets = first._words
    else:
        targets = np.stack([vector._words for vector in vectors])
    if signature is not None and all(
        isinstance(item, ExplicitBitVect) for item in others
    ):
        query._bulk_cache = (
            others,
            signature,
            ExplicitBitVect._mutation_epoch,
            targets,
        )
    scores = np.empty(len(vectors), dtype=np.float64)
    lib().mrd_bulk_similarity(
        addr(query._words),
        addr(targets),
        addr(scores),
        len(vectors),
        n_words,
        len(query),
        _METRICS[metric],
    )
    if returnDistance:
        scores = 1.0 - scores
    return scores.tolist()


def BulkTanimotoSimilarity(bv1, bvList, returnDistance: bool = False) -> list[float]:
    return _bulk(bv1, bvList, "tanimoto", returnDistance)


def BulkDiceSimilarity(bv1, bvList, returnDistance: bool = False) -> list[float]:
    return _bulk(bv1, bvList, "dice", returnDistance)


def FingerprintSimilarity(fp1, fp2, metric=TanimotoSimilarity) -> float:
    return metric(fp1, fp2)
