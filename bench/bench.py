"""Measured comparisons against RDKit on identical molecules and bit vectors."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np
from rdkit import Chem as rdChem
from rdkit import DataStructs as rdDataStructs
from rdkit import RDLogger
from rdkit.Chem import rdMolDescriptors

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

from mojordkit import Chem, DataStructs  # noqa: E402

RDLogger.DisableLog("rdApp.warning")


def timeit(function, repeats=5):
    best = math.inf
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def machine_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def row(name, mojo_time, rdkit_time):
    ratio = rdkit_time / mojo_time
    result = "faster" if ratio > 1 else "slower"
    print(
        f"| {name} | {mojo_time * 1e3:.3f} ms | {rdkit_time * 1e3:.3f} ms | "
        f"{ratio:.2f}x {result} |"
    )


def morgan_case():
    smiles = [
        "CCO",
        "CC(C)C",
        "c1ccccc1",
        "CC(=O)Oc1ccccc1C(=O)O",
        "N[C@@H](C)C(=O)O",
        "O=C(O)c1ccccc1O",
        "CCN(CC)CC",
        "COc1ccc(CCN)cc1",
    ]
    ours = [Chem.MolFromSmiles(item) for item in smiles] * 100
    theirs = [rdChem.MolFromSmiles(item) for item in smiles] * 100
    return (
        lambda: [Chem.GetMorganFingerprintAsBitVect(mol, 2) for mol in ours],
        lambda: [
            rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2)
            for mol in theirs
        ],
    )


def bulk_case():
    rng = np.random.default_rng(7)
    n_vectors = 5_000
    n_bits = 2048
    n_words = n_bits // 64
    words = rng.integers(
        0, np.iinfo(np.uint64).max, size=(n_vectors + 1, n_words), dtype=np.uint64
    )
    words &= rng.random(words.shape) < 0.12
    query = DataStructs.ExplicitBitVect._from_words(n_bits, words[0])
    vectors = [
        DataStructs.ExplicitBitVect._from_words(n_bits, value)
        for value in words[1:]
    ]
    upstream_query = query.to_rdkit()
    upstream_vectors = [vector.to_rdkit() for vector in vectors]
    return (
        lambda: DataStructs.BulkTanimotoSimilarity(query, vectors),
        lambda: rdDataStructs.BulkTanimotoSimilarity(
            upstream_query, upstream_vectors
        ),
    )


def substructure_case():
    smiles = "C" * 80
    ours = Chem.MolFromSmiles(smiles)
    ours_query = Chem.MolFromSmarts("CCCCCCCC")
    theirs = rdChem.MolFromSmiles(smiles)
    theirs_query = rdChem.MolFromSmarts("CCCCCCCC")
    return (
        lambda: ours.GetSubstructMatches(ours_query, maxMatches=1000),
        lambda: theirs.GetSubstructMatches(theirs_query, maxMatches=1000),
    )


def main():
    print(f"Machine: {machine_name()}; Python {platform.python_version()}")
    print()
    print("| operation | mojo-rdkit | RDKit | RDKit time / Mojo time |")
    print("|---|---:|---:|---:|")
    cases = [
        ("Morgan radius=2, 800 molecules", morgan_case),
        ("Bulk Tanimoto, 5,000 x 2,048-bit", bulk_case),
        ("Substructure C8 in C80, all matches", substructure_case),
    ]
    for name, build in cases:
        ours, theirs = build()
        ours()
        theirs()
        row(name, timeit(ours), timeit(theirs))


if __name__ == "__main__":
    main()
