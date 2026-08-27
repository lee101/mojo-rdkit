from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJORDKIT_LIB", os.path.join(ROOT, "dist", "libmojo-rdkit.so"))

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mrd_morgan": ([I] * 27, I),
    "mrd_similarity": ([I, I, I, I, I], F),
    "mrd_bulk_similarity": ([I] * 9, None),
    "mrd_substruct_matches": ([I] * 19, I),
}

_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        if not os.path.exists(LIB):
            subprocess.run(
                ["bash", os.path.join(ROOT, "build", "build.sh")],
                check=True,
                cwd=ROOT,
            )
        _library = ctypes.CDLL(LIB)
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def addr(array: np.ndarray) -> int:
    if not isinstance(array, np.ndarray):
        raise TypeError("native buffers must be NumPy arrays")
    if array.size == 0 or not array.flags.c_contiguous:
        raise ValueError("native buffers must be non-empty and C-contiguous")
    address = int(array.ctypes.data)
    if address == 0:
        raise ValueError("native buffer has a null address")
    return address
