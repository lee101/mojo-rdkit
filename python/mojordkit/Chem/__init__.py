from __future__ import annotations

import weakref
import threading
import operator

import numpy as np
from rdkit import Chem as _rdchem

from .._lib import addr, lib


def _native(mol):
    return getattr(mol, "_mol", mol)


def _needs_context(query) -> bool:
    return any("RecursiveStructure" in atom.DescribeQuery() for atom in query.GetAtoms())


def _prepare_substructure(target, query):
    n_query = query.GetNumAtoms()
    n_target = target.GetNumAtoms()
    query_bonds = list(query.GetBonds())
    target_bonds = list(target.GetBonds())
    n_query_bonds = len(query_bonds)
    n_target_bonds = len(target_bonds)

    atom_compat = np.empty(max(1, n_query * n_target), dtype=np.uint8)
    target_atoms = tuple(target.GetAtoms())
    for query_atom in query.GetAtoms():
        row = query_atom.GetIdx() * n_target
        for target_atom in target_atoms:
            atom_compat[row + target_atom.GetIdx()] = query_atom.Match(target_atom)

    query_u = np.empty(max(1, n_query_bonds), dtype=np.int32)
    query_v = np.empty(max(1, n_query_bonds), dtype=np.int32)
    parent_query = np.full(max(1, n_query), -1, dtype=np.int32)
    parent_edge = np.full(max(1, n_query), -1, dtype=np.int32)
    for bond in query_bonds:
        edge = bond.GetIdx()
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        query_u[edge] = u
        query_v[edge] = v
        child, parent = (u, v) if u > v else (v, u)
        if parent_query[child] < 0:
            parent_query[child] = parent
            parent_edge[child] = edge

    adjacency = np.zeros(max(1, n_target * n_target), dtype=np.int32)
    target_offsets = np.empty(n_target + 1, dtype=np.int32)
    target_neighbors = np.empty(max(1, 2 * n_target_bonds), dtype=np.int32)
    position = 0
    for atom in target_atoms:
        target_offsets[atom.GetIdx()] = position
        for neighbor in atom.GetNeighbors():
            target_neighbors[position] = neighbor.GetIdx()
            position += 1
    target_offsets[n_target] = position
    for bond in target_bonds:
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency[u * n_target + v] = bond.GetIdx() + 1
        adjacency[v * n_target + u] = bond.GetIdx() + 1

    bond_compat = np.empty(max(1, n_query_bonds * n_target_bonds), dtype=np.uint8)
    for query_bond in query_bonds:
        row = query_bond.GetIdx() * n_target_bonds
        for target_bond in target_bonds:
            bond_compat[row + target_bond.GetIdx()] = query_bond.Match(target_bond)
    return (
        n_query,
        n_target,
        n_query_bonds,
        n_target_bonds,
        atom_compat,
        query_u,
        query_v,
        parent_query,
        parent_edge,
        target_offsets,
        target_neighbors,
        adjacency,
        bond_compat,
    )


def _substruct_matches(
    target,
    query,
    uniquify=True,
    useChirality=False,
    useQueryQueryMatches=False,
    maxMatches=1000,
):
    target_native = _native(target)
    query_native = _native(query)
    query_owner = (
        query
        if hasattr(query, "_mol") and getattr(query, "_cache_safe", False)
        else None
    )
    target_owner = (
        target
        if hasattr(target, "_mol") and getattr(target, "_cache_safe", False)
        else None
    )
    needs_context = getattr(query_owner, "_needs_context_cache", None)
    if needs_context is None:
        needs_context = _needs_context(query_native)
        if query_owner is not None:
            query_owner._needs_context_cache = needs_context
    if useChirality or useQueryQueryMatches or needs_context:
        return target_native.GetSubstructMatches(
            query_native,
            uniquify=uniquify,
            useChirality=useChirality,
            useQueryQueryMatches=useQueryQueryMatches,
            maxMatches=maxMatches,
        )

    cache = getattr(target_owner, "_substruct_inputs", None)
    cache_entry = (
        cache.get(query_owner) if cache is not None and query_owner is not None else None
    )
    if cache_entry is None:
        prepared = _prepare_substructure(target_native, query_native)
        prepared_addresses = tuple(value.ctypes.data for value in prepared[4:])
        if cache is not None and query_owner is not None:
            cache[query_owner] = (prepared, prepared_addresses)
    else:
        prepared, prepared_addresses = cache_entry
    (
        n_query,
        n_target,
        n_query_bonds,
        n_target_bonds,
        atom_compat,
        query_u,
        query_v,
        parent_query,
        parent_edge,
        target_offsets,
        target_neighbors,
        adjacency,
        bond_compat,
    ) = prepared

    max_matches = operator.index(maxMatches)
    if max_matches < 0:
        raise OverflowError("maxMatches must be non-negative")
    if max_matches == 0:
        return target_native.GetSubstructMatches(
            query_native,
            uniquify=uniquify,
            useChirality=useChirality,
            useQueryQueryMatches=useQueryQueryMatches,
            maxMatches=0,
        )
    scratch_local = getattr(target_owner, "_substruct_scratch", None)
    scratch_cache = getattr(scratch_local, "buffers", None)
    if scratch_cache is None and scratch_local is not None:
        scratch_cache = weakref.WeakKeyDictionary()
        scratch_local.buffers = scratch_cache
    scratch_entry = (
        scratch_cache.get(query_owner)
        if scratch_cache is not None and query_owner is not None
        else None
    )
    if scratch_entry is None or scratch_entry[0] < max_matches:
        mapping = np.empty(max(1, n_query), dtype=np.int32)
        used = np.empty(max(1, n_target), dtype=np.uint8)
        next_candidate = np.empty(max(1, n_query), dtype=np.int32)
        matches = np.empty(max(1, max_matches * n_query), dtype=np.int32)
        scratch = (mapping, used, next_candidate, matches)
        scratch_addresses = tuple(value.ctypes.data for value in scratch)
        scratch_entry = (max_matches, scratch, scratch_addresses)
        if scratch_cache is not None and query_owner is not None:
            scratch_cache[query_owner] = scratch_entry
    _, scratch, scratch_addresses = scratch_entry
    mapping, used, next_candidate, matches = scratch
    count = lib().mrd_substruct_matches(
        *prepared_addresses,
        *scratch_addresses,
        n_query,
        n_target,
        n_query_bonds,
        n_target_bonds,
        max_matches,
        int(bool(uniquify)),
    )
    rows = matches[: count * n_query].reshape(count, n_query).tolist()
    return tuple(map(tuple, rows))


class Mol:
    def __init__(self, mol=None):
        if isinstance(mol, Mol):
            mol = mol._mol
        self._mol = _rdchem.Mol() if mol is None else _rdchem.Mol(mol)
        self._cache_safe = True
        self._substruct_inputs = weakref.WeakKeyDictionary()
        self._substruct_scratch = threading.local()

    def _disable_native_caches(self):
        self._cache_safe = False
        self._substruct_inputs.clear()
        self.__dict__.pop("_needs_context_cache", None)
        self.__dict__.pop("_morgan_inputs", None)
        self.__dict__.pop("_morgan_results", None)

    def GetAtomWithIdx(self, *args, **kwargs):
        self._disable_native_caches()
        return self._mol.GetAtomWithIdx(*args, **kwargs)

    def GetAtoms(self, *args, **kwargs):
        self._disable_native_caches()
        return self._mol.GetAtoms(*args, **kwargs)

    def GetBondWithIdx(self, *args, **kwargs):
        self._disable_native_caches()
        return self._mol.GetBondWithIdx(*args, **kwargs)

    def GetBonds(self, *args, **kwargs):
        self._disable_native_caches()
        return self._mol.GetBonds(*args, **kwargs)

    def GetSubstructMatches(
        self,
        query,
        uniquify=True,
        useChirality=False,
        useQueryQueryMatches=False,
        maxMatches=1000,
    ):
        return _substruct_matches(
            self,
            query,
            uniquify,
            useChirality,
            useQueryQueryMatches,
            maxMatches,
        )

    def GetSubstructMatch(
        self,
        query,
        useChirality=False,
        useQueryQueryMatches=False,
    ):
        matches = _substruct_matches(
            self,
            query,
            True,
            useChirality,
            useQueryQueryMatches,
            1,
        )
        return matches[0] if matches else ()

    def HasSubstructMatch(
        self,
        query,
        useChirality=False,
        useQueryQueryMatches=False,
    ):
        return bool(
            self.GetSubstructMatch(query, useChirality, useQueryQueryMatches)
        )

    def __getattr__(self, name):
        value = getattr(self._mol, name)
        if callable(value) and name.startswith(("Set", "Clear", "Update", "Remove")):
            def guarded(*args, **kwargs):
                self._disable_native_caches()
                return value(*args, **kwargs)

            return guarded
        return value


def MolFromSmiles(*args, **kwargs):
    result = _rdchem.MolFromSmiles(*args, **kwargs)
    return None if result is None else Mol(result)


def MolFromSmarts(*args, **kwargs):
    result = _rdchem.MolFromSmarts(*args, **kwargs)
    return None if result is None else Mol(result)


def MolToSmiles(mol, *args, **kwargs):
    return _rdchem.MolToSmiles(_native(mol), *args, **kwargs)


def HasSubstructMatch(mol, query, *args, **kwargs):
    return Mol(_native(mol)).HasSubstructMatch(query, *args, **kwargs)


def GetSubstructMatch(mol, query, *args, **kwargs):
    return Mol(_native(mol)).GetSubstructMatch(query, *args, **kwargs)


def GetSubstructMatches(mol, query, *args, **kwargs):
    return Mol(_native(mol)).GetSubstructMatches(query, *args, **kwargs)


from . import AllChem, rdFingerprintGenerator, rdMolDescriptors  # noqa: E402
from ._fingerprints import GetMorganFingerprintAsBitVect  # noqa: E402

__all__ = [
    "AllChem",
    "GetMorganFingerprintAsBitVect",
    "GetSubstructMatch",
    "GetSubstructMatches",
    "HasSubstructMatch",
    "Mol",
    "MolFromSmarts",
    "MolFromSmiles",
    "MolToSmiles",
    "rdFingerprintGenerator",
    "rdMolDescriptors",
]
