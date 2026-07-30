import pytest
from rdkit import Chem as rdChem

from mojordkit import Chem


TARGET = "CC(=O)Oc1ccccc1C(=O)O"
SMARTS = [
    "c1ccccc1",
    "[#6]",
    "[O,N]",
    "[!#6]",
    "C(=O)O",
    "*~*",
    "[R]",
    "[r6]",
    "[D2]",
    "[C;H3]",
    "[C,N;!H0]",
    "C-,=C",
]


@pytest.mark.parametrize("smarts", SMARTS)
def test_smarts_substructure_parity(smarts):
    ours = Chem.MolFromSmiles(TARGET).GetSubstructMatches(
        Chem.MolFromSmarts(smarts)
    )
    theirs = rdChem.MolFromSmiles(TARGET).GetSubstructMatches(
        rdChem.MolFromSmarts(smarts)
    )
    assert ours == theirs


@pytest.mark.parametrize("smarts", ["CC", "C(C)C", "c1ccccc1"])
def test_nonunique_match_set_parity(smarts):
    target = "CC(C)c1ccccc1"
    ours = Chem.MolFromSmiles(target).GetSubstructMatches(
        Chem.MolFromSmarts(smarts), uniquify=False
    )
    theirs = rdChem.MolFromSmiles(target).GetSubstructMatches(
        rdChem.MolFromSmarts(smarts), uniquify=False
    )
    assert set(ours) == set(theirs)


def test_has_and_first_match_parity():
    ours = Chem.MolFromSmiles(TARGET)
    theirs = rdChem.MolFromSmiles(TARGET)
    for smarts in ["C(=O)O", "N#N"]:
        ours_query = Chem.MolFromSmarts(smarts)
        theirs_query = rdChem.MolFromSmarts(smarts)
        assert ours.HasSubstructMatch(ours_query) == theirs.HasSubstructMatch(
            theirs_query
        )
        assert ours.GetSubstructMatch(ours_query) == theirs.GetSubstructMatch(
            theirs_query
        )


def test_molecule_query_parity():
    ours = Chem.MolFromSmiles("CCOC").GetSubstructMatches(
        Chem.MolFromSmiles("CO")
    )
    theirs = rdChem.MolFromSmiles("CCOC").GetSubstructMatches(
        rdChem.MolFromSmiles("CO")
    )
    assert ours == theirs


def test_max_matches_parity():
    ours = Chem.MolFromSmiles("CCCCCCCC").GetSubstructMatches(
        Chem.MolFromSmarts("C"), maxMatches=3
    )
    theirs = rdChem.MolFromSmiles("CCCCCCCC").GetSubstructMatches(
        rdChem.MolFromSmarts("C"), maxMatches=3
    )
    assert ours == theirs


def test_zero_max_matches_means_unlimited():
    ours = Chem.MolFromSmiles("CCCC").GetSubstructMatches(
        Chem.MolFromSmarts("C"), maxMatches=0
    )
    theirs = rdChem.MolFromSmiles("CCCC").GetSubstructMatches(
        rdChem.MolFromSmarts("C"), maxMatches=0
    )
    assert ours == theirs


def test_invalid_max_matches_rejected():
    mol = Chem.MolFromSmiles("CC")
    query = Chem.MolFromSmarts("C")
    with pytest.raises(OverflowError):
        mol.GetSubstructMatches(query, maxMatches=-1)
    with pytest.raises(TypeError):
        mol.GetSubstructMatches(query, maxMatches=1.5)


def test_recursive_smarts_fallback_parity():
    smarts = "[$(C=O)]"
    ours = Chem.MolFromSmiles(TARGET).GetSubstructMatches(
        Chem.MolFromSmarts(smarts)
    )
    theirs = rdChem.MolFromSmiles(TARGET).GetSubstructMatches(
        rdChem.MolFromSmarts(smarts)
    )
    assert ours == theirs


def test_chiral_match_fallback_parity():
    target = "N[C@@H](C)C(=O)O"
    query = "N[C@@H](C)C(=O)O"
    ours = Chem.MolFromSmiles(target).GetSubstructMatches(
        Chem.MolFromSmarts(query), useChirality=True
    )
    theirs = rdChem.MolFromSmiles(target).GetSubstructMatches(
        rdChem.MolFromSmarts(query), useChirality=True
    )
    assert ours == theirs


def test_module_level_search_helpers():
    target = Chem.MolFromSmiles("CCO")
    query = Chem.MolFromSmarts("CO")
    assert Chem.HasSubstructMatch(target, query)
    assert Chem.GetSubstructMatch(target, query) == (1, 2)
    assert Chem.GetSubstructMatches(target, query) == ((1, 2),)


def test_smiles_roundtrip():
    mol = Chem.MolFromSmiles("OC(C)C")
    assert Chem.MolToSmiles(mol) == "CC(C)O"


def test_substructure_cache_invalidates_after_atom_mutation():
    ours = Chem.MolFromSmiles("CCO")
    query = Chem.MolFromSmarts("[#7]")
    assert ours.GetSubstructMatches(query) == ()
    ours.GetAtomWithIdx(0).SetAtomicNum(7)

    theirs = rdChem.MolFromSmiles("CCO")
    theirs.GetAtomWithIdx(0).SetAtomicNum(7)
    assert ours.GetSubstructMatches(query) == theirs.GetSubstructMatches(
        rdChem.MolFromSmarts("[#7]")
    )
