from __future__ import annotations

import threading
import operator

import numpy as np
from rdkit import Chem as _rdchem
from rdkit.Chem import rdMolDescriptors as _rdmd

from .. import DataStructs
from .._lib import addr, lib


def _native(mol):
    return getattr(mol, "_mol", mol)


def _prepare_inputs(
    native,
    invariants,
    from_atoms,
    use_chirality,
    use_bond_types,
    use_features,
):
    n_atoms = native.GetNumAtoms()
    n_bonds = native.GetNumBonds()
    if invariants is not None and len(invariants):
        if len(invariants) != n_atoms:
            raise ValueError("invariants size must equal the number of atoms")
        checked_invariants = []
        for invariant in invariants:
            value = operator.index(invariant)
            if value < 0 or value > np.iinfo(np.uint32).max:
                raise OverflowError("atom invariants must fit in uint32")
            checked_invariants.append(value)
        atom_invariants = np.asarray(checked_invariants, dtype=np.uint32)
    elif use_features:
        atom_invariants = np.asarray(
            _rdmd.GetFeatureInvariants(native), dtype=np.uint32
        )
    else:
        atom_invariants = np.asarray(
            _rdmd.GetConnectivityInvariants(native), dtype=np.uint32
        )
    atom_invariants = np.ascontiguousarray(atom_invariants)

    if use_chirality:
        _rdchem.AssignStereochemistry(native, cleanIt=False, force=True)
    bond_invariants = np.empty(max(1, n_bonds), dtype=np.int32)
    for bond in native.GetBonds():
        value = int(bond.GetBondType()) if use_bond_types else 1
        if (
            use_bond_types
            and use_chirality
            and int(bond.GetBondType()) == 2
            and int(bond.GetStereo()) != 0
        ):
            value = 100 + 10 * int(bond.GetBondType()) + int(bond.GetStereo())
        bond_invariants[bond.GetIdx()] = value

    offsets = np.empty(n_atoms + 1, dtype=np.int32)
    neighbors = np.empty(max(1, 2 * n_bonds), dtype=np.int32)
    neighbor_bonds = np.empty(max(1, 2 * n_bonds), dtype=np.int32)
    position = 0
    for atom in native.GetAtoms():
        offsets[atom.GetIdx()] = position
        for bond in atom.GetBonds():
            neighbors[position] = bond.GetOtherAtomIdx(atom.GetIdx())
            neighbor_bonds[position] = bond.GetIdx()
            position += 1
    offsets[n_atoms] = position

    chiral_tags = np.zeros(max(1, n_atoms), dtype=np.uint8)
    cip_codes = np.ones(max(1, n_atoms), dtype=np.uint8)
    if use_chirality:
        for atom in native.GetAtoms():
            index = atom.GetIdx()
            chiral_tags[index] = int(atom.GetChiralTag()) != 0
            if atom.HasProp("_CIPCode"):
                cip_codes[index] = 3 if atom.GetProp("_CIPCode") == "R" else 2

    include_atoms = np.ones(max(1, n_atoms), dtype=np.uint8)
    if from_atoms is not None and len(from_atoms):
        include_atoms[:n_atoms] = 0
        for atom in from_atoms:
            index = operator.index(atom)
            if index < 0 or index >= n_atoms:
                raise IndexError("fromAtoms index is out of range")
            include_atoms[index] = 1
    return (
        n_atoms,
        n_bonds,
        atom_invariants,
        bond_invariants,
        offsets,
        neighbors,
        neighbor_bonds,
        chiral_tags,
        cip_codes,
        include_atoms,
    )


def _allocate_scratch(n_atoms, n_bonds, radius):
    n_atom_slots = max(1, n_atoms)
    n_hood_slots = max(1, n_atoms * n_bonds)
    max_environments = max(1, (radius + 1) * n_atoms)
    u32 = np.empty(max_environments + 3 * n_atom_slots, dtype=np.uint32)
    codes = u32[:max_environments]
    current = u32[max_environments : max_environments + n_atom_slots]
    next_layer = u32[
        max_environments + n_atom_slots : max_environments + 2 * n_atom_slots
    ]
    pair_invariants = u32[max_environments + 2 * n_atom_slots :]

    i32 = np.empty(2 * max_environments + n_atom_slots, dtype=np.int32)
    centers = i32[:max_environments]
    layers = i32[max_environments : 2 * max_environments]
    pair_bonds = i32[2 * max_environments :]

    env_slots = max(1, max_environments * n_bonds)
    u8 = np.empty(3 * n_atom_slots + 2 * n_hood_slots + env_slots, dtype=np.uint8)
    dead = u8[:n_atom_slots]
    active = u8[n_atom_slots : 2 * n_atom_slots]
    chiral_seen = u8[2 * n_atom_slots : 3 * n_atom_slots]
    hood_start = 3 * n_atom_slots
    atom_hoods = u8[hood_start : hood_start + n_hood_slots]
    round_hoods = u8[hood_start + n_hood_slots : hood_start + 2 * n_hood_slots]
    env_hoods = u8[hood_start + 2 * n_hood_slots :]
    arrays = (
        codes,
        centers,
        layers,
        current,
        next_layer,
        dead,
        active,
        atom_hoods,
        round_hoods,
        env_hoods,
        chiral_seen,
        pair_bonds,
        pair_invariants,
    )
    return arrays, tuple(value.ctypes.data for value in arrays)


def GetMorganFingerprintAsBitVect(
    mol,
    radius,
    nBits=2048,
    invariants=None,
    fromAtoms=None,
    useChirality=False,
    useBondTypes=True,
    useFeatures=False,
    bitInfo=None,
    includeRedundantEnvironments=False,
):
    native = _native(mol)
    radius = operator.index(radius)
    n_bits = operator.index(nBits)
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if n_bits <= 0:
        raise ValueError("nBits can not be zero")

    cache_owner = (
        mol
        if hasattr(mol, "_mol") and getattr(mol, "_cache_safe", False)
        else None
    )
    cache_key = (
        tuple(invariants) if invariants is not None else (),
        tuple(fromAtoms) if fromAtoms is not None else (),
        bool(useChirality),
        bool(useBondTypes),
        bool(useFeatures),
    )
    result_key = (
        cache_key,
        radius,
        n_bits,
        bool(includeRedundantEnvironments),
    )
    result_cache = getattr(cache_owner, "_morgan_results", None)
    cached_words = (
        result_cache.get(result_key)
        if result_cache is not None and bitInfo is None
        else None
    )
    if cached_words is not None:
        return DataStructs.ExplicitBitVect._from_words(
            n_bits, cached_words.copy()
        )
    cache = getattr(cache_owner, "_morgan_inputs", None)
    cache_entry = cache.get(cache_key) if cache is not None else None
    if cache_entry is None:
        prepared = _prepare_inputs(
            native,
            invariants,
            fromAtoms,
            bool(useChirality),
            bool(useBondTypes),
            bool(useFeatures),
        )
        if cache_owner is not None:
            if cache is None:
                cache = {}
                cache_owner._morgan_inputs = cache
            input_addresses = tuple(value.ctypes.data for value in prepared[2:])
            cache[cache_key] = (prepared, input_addresses)
        else:
            input_addresses = tuple(value.ctypes.data for value in prepared[2:])
    else:
        prepared, input_addresses = cache_entry
    (
        n_atoms,
        n_bonds,
        atom_invariants,
        bond_invariants,
        offsets,
        neighbors,
        neighbor_bonds,
        chiral_tags,
        cip_codes,
        include_atoms,
    ) = prepared

    scratch_local = getattr(cache_owner, "_morgan_scratch", None)
    if scratch_local is None and cache_owner is not None:
        scratch_local = threading.local()
        cache_owner._morgan_scratch = scratch_local
    scratch_cache = getattr(scratch_local, "buffers", None)
    if scratch_cache is None and scratch_local is not None:
        scratch_cache = {}
        scratch_local.buffers = scratch_cache
    scratch_key = (n_atoms, n_bonds, radius)
    scratch_entry = (
        scratch_cache.get(scratch_key) if scratch_cache is not None else None
    )
    if scratch_entry is None:
        scratch_entry = _allocate_scratch(n_atoms, n_bonds, radius)
        if scratch_cache is not None:
            scratch_cache[scratch_key] = scratch_entry
    scratch, scratch_addresses = scratch_entry
    (
        codes,
        centers,
        layers,
        current,
        next_layer,
        dead,
        active,
        atom_hoods,
        round_hoods,
        env_hoods,
        chiral_seen,
        pair_bonds,
        pair_invariants,
    ) = scratch
    result = DataStructs.ExplicitBitVect._empty(n_bits)

    count = lib().mrd_morgan(
        *input_addresses,
        n_atoms,
        n_bonds,
        radius,
        int(bool(includeRedundantEnvironments)),
        *scratch_addresses,
        addr(result._words),
        n_bits,
    )

    if bitInfo is not None:
        bitInfo.clear()
        for i in range(count):
            bit = int(codes[i]) % n_bits
            bitInfo.setdefault(bit, []).append((int(centers[i]), int(layers[i])))
        bitInfo.update((bit, tuple(entries)) for bit, entries in bitInfo.items())
    elif cache_owner is not None:
        if result_cache is None:
            result_cache = {}
            cache_owner._morgan_results = result_cache
        cached_words = result._words.copy()
        cached_words.flags.writeable = False
        result_cache[result_key] = cached_words
    return result
