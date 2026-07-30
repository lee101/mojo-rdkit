import pytest
from rdkit import Chem as rdChem
from rdkit.Chem import rdMolDescriptors as rdDescriptors

from mojordkit import Chem


SMILES = [
    "C",
    "CCO",
    "CC(C)C",
    "C1CCCCC1",
    "c1ccccc1",
    "c1ncccc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "N[C@@H](C)C(=O)O",
    "F/C=C/F",
    "F/C=C\\F",
    "[Na+].[Cl-]",
    "O=C(O)c1ccccc1O",
]


@pytest.mark.parametrize("smiles", SMILES)
@pytest.mark.parametrize("radius", [0, 1, 2, 3])
def test_morgan_default_parity(smiles, radius):
    ours = Chem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), radius)
    theirs = rdDescriptors.GetMorganFingerprintAsBitVect(
        rdChem.MolFromSmiles(smiles), radius
    )
    assert ours.GetNumBits() == theirs.GetNumBits()
    assert ours.GetOnBits() == tuple(theirs.GetOnBits())


@pytest.mark.parametrize(
    "option",
    [
        {"useFeatures": True},
        {"useBondTypes": False},
        {"includeRedundantEnvironments": True},
        {"fromAtoms": [0]},
        {"nBits": 128},
        {"nBits": 4093},
    ],
)
def test_morgan_option_parity(option):
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    ours = Chem.GetMorganFingerprintAsBitVect(
        Chem.MolFromSmiles(smiles), 3, **option
    )
    theirs = rdDescriptors.GetMorganFingerprintAsBitVect(
        rdChem.MolFromSmiles(smiles), 3, **option
    )
    assert ours.GetOnBits() == tuple(theirs.GetOnBits())


@pytest.mark.parametrize("smiles", ["N[C@@H](C)C(=O)O", "F/C=C/F", "F/C=C\\F"])
def test_morgan_chirality_parity(smiles):
    ours = Chem.GetMorganFingerprintAsBitVect(
        Chem.MolFromSmiles(smiles), 3, useChirality=True
    )
    theirs = rdDescriptors.GetMorganFingerprintAsBitVect(
        rdChem.MolFromSmiles(smiles), 3, useChirality=True
    )
    assert ours.GetOnBits() == tuple(theirs.GetOnBits())


def test_morgan_custom_invariants_parity():
    smiles = "CCOC"
    invariants = [10, 20, 30, 40]
    ours = Chem.GetMorganFingerprintAsBitVect(
        Chem.MolFromSmiles(smiles), 2, invariants=invariants
    )
    theirs = rdDescriptors.GetMorganFingerprintAsBitVect(
        rdChem.MolFromSmiles(smiles), 2, invariants=invariants
    )
    assert ours.GetOnBits() == tuple(theirs.GetOnBits())


def test_morgan_bit_info_parity():
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    ours_info = {}
    theirs_info = {}
    Chem.GetMorganFingerprintAsBitVect(
        Chem.MolFromSmiles(smiles), 3, bitInfo=ours_info
    )
    rdDescriptors.GetMorganFingerprintAsBitVect(
        rdChem.MolFromSmiles(smiles), 3, bitInfo=theirs_info
    )
    assert ours_info == theirs_info


def test_morgan_generator_api_parity():
    smiles = "COc1ccccc1"
    generator = Chem.rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=1024
    )
    ours = generator.GetFingerprint(Chem.MolFromSmiles(smiles))
    theirs = rdDescriptors.GetMorganFingerprintAsBitVect(
        rdChem.MolFromSmiles(smiles), 2, nBits=1024
    )
    assert ours.GetOnBits() == tuple(theirs.GetOnBits())


def test_morgan_module_aliases():
    mol = Chem.MolFromSmiles("CCO")
    expected = Chem.GetMorganFingerprintAsBitVect(mol, 2).GetOnBits()
    assert Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol, 2
    ).GetOnBits() == expected
    assert Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 2).GetOnBits() == expected


def test_invalid_fingerprint_arguments():
    mol = Chem.MolFromSmiles("CC")
    with pytest.raises(ValueError):
        Chem.GetMorganFingerprintAsBitVect(mol, -1)
    with pytest.raises(ValueError):
        Chem.GetMorganFingerprintAsBitVect(mol, 2, nBits=0)
    with pytest.raises(ValueError):
        Chem.GetMorganFingerprintAsBitVect(mol, 2, invariants=[1])
    with pytest.raises(TypeError):
        Chem.GetMorganFingerprintAsBitVect(mol, 1.5)
    with pytest.raises(TypeError):
        Chem.GetMorganFingerprintAsBitVect(mol, 2, invariants=[1.5, 2])
    with pytest.raises(OverflowError):
        Chem.GetMorganFingerprintAsBitVect(mol, 2, invariants=[-1, 2])
    with pytest.raises(IndexError):
        Chem.GetMorganFingerprintAsBitVect(mol, 2, fromAtoms=[-1])


def test_morgan_cache_invalidates_after_atom_mutation():
    ours_mol = Chem.MolFromSmiles("CCO")
    Chem.GetMorganFingerprintAsBitVect(ours_mol, 2)
    ours_mol.GetAtomWithIdx(0).SetAtomicNum(7)
    theirs_mol = rdChem.MolFromSmiles("CCO")
    theirs_mol.GetAtomWithIdx(0).SetAtomicNum(7)
    ours = Chem.GetMorganFingerprintAsBitVect(ours_mol, 2)
    theirs = rdDescriptors.GetMorganFingerprintAsBitVect(theirs_mol, 2)
    assert ours.GetOnBits() == tuple(theirs.GetOnBits())


def test_morgan_cached_result_is_independent():
    mol = Chem.MolFromSmiles("CCO")
    first = Chem.GetMorganFingerprintAsBitVect(mol, 2)
    expected = first.GetOnBits()
    first.SetBit(17)
    second = Chem.GetMorganFingerprintAsBitVect(mol, 2)
    assert second.GetOnBits() == expected
