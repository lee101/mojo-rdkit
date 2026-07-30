import numpy as np
import pytest
from rdkit import DataStructs as rdDataStructs

from mojordkit import DataStructs


def vectors():
    ours_a = DataStructs.ExplicitBitVect(257, [0, 1, 64, 128, 256])
    ours_b = DataStructs.ExplicitBitVect(257, [1, 3, 64, 129, 256])
    return ours_a, ours_b, ours_a.to_rdkit(), ours_b.to_rdkit()


@pytest.mark.parametrize(
    ("ours_name", "upstream_name"),
    [
        ("TanimotoSimilarity", "TanimotoSimilarity"),
        ("DiceSimilarity", "DiceSimilarity"),
        ("CosineSimilarity", "CosineSimilarity"),
        ("SokalSimilarity", "SokalSimilarity"),
        ("RusselSimilarity", "RusselSimilarity"),
        ("KulczynskiSimilarity", "KulczynskiSimilarity"),
        ("McConnaugheySimilarity", "McConnaugheySimilarity"),
    ],
)
def test_similarity_parity(ours_name, upstream_name):
    ours_a, ours_b, theirs_a, theirs_b = vectors()
    ours = getattr(DataStructs, ours_name)(ours_a, ours_b)
    theirs = getattr(rdDataStructs, upstream_name)(theirs_a, theirs_b)
    assert ours == pytest.approx(theirs)


@pytest.mark.parametrize("metric", ["TanimotoSimilarity", "DiceSimilarity"])
def test_bulk_similarity_parity(metric):
    query = DataStructs.ExplicitBitVect(130, [0, 64, 129])
    others = [
        DataStructs.ExplicitBitVect(130, []),
        DataStructs.ExplicitBitVect(130, [0, 64]),
        DataStructs.ExplicitBitVect(130, [1, 2, 3]),
    ]
    upstream_query = query.to_rdkit()
    upstream_others = [item.to_rdkit() for item in others]
    ours = getattr(DataStructs, f"Bulk{metric}")(query, others)
    theirs = getattr(rdDataStructs, f"Bulk{metric}")(
        upstream_query, upstream_others
    )
    assert ours == pytest.approx(theirs)


def test_return_distance_parity():
    ours_a, ours_b, theirs_a, theirs_b = vectors()
    assert DataStructs.TanimotoSimilarity(
        ours_a, ours_b, returnDistance=True
    ) == pytest.approx(
        rdDataStructs.TanimotoSimilarity(
            theirs_a, theirs_b, returnDistance=True
        )
    )


def test_zero_vector_parity():
    ours = DataStructs.ExplicitBitVect(128)
    theirs = rdDataStructs.ExplicitBitVect(128)
    assert DataStructs.TanimotoSimilarity(ours, ours) == rdDataStructs.TanimotoSimilarity(
        theirs, theirs
    )


def test_explicit_bit_vector_operations():
    vector = DataStructs.ExplicitBitVect(70)
    assert vector.SetBit(0) is False
    assert vector.SetBit(69) is False
    assert vector.SetBit(69) is True
    assert vector.GetOnBits() == (0, 69)
    assert vector.GetNumOnBits() == 2
    assert vector.UnSetBit(0) is True
    assert vector[-1] == 1
    other = DataStructs.ExplicitBitVect(70, [1, 69])
    assert (vector & other).GetOnBits() == (69,)
    assert (vector | other).GetOnBits() == (1, 69)
    assert (vector ^ other).GetOnBits() == (1,)


def test_upstream_vector_interop():
    _, _, theirs_a, theirs_b = vectors()
    ours = DataStructs.TanimotoSimilarity(theirs_a, theirs_b)
    theirs = rdDataStructs.TanimotoSimilarity(theirs_a, theirs_b)
    assert ours == pytest.approx(theirs)


def test_fingerprint_similarity_alias():
    ours_a, ours_b, theirs_a, theirs_b = vectors()
    assert DataStructs.FingerprintSimilarity(ours_a, ours_b) == pytest.approx(
        rdDataStructs.FingerprintSimilarity(theirs_a, theirs_b)
    )


def test_from_words_rejects_narrowing_and_masks_padding():
    with pytest.raises(TypeError):
        DataStructs.ExplicitBitVect._from_words(
            64, np.array([1], dtype=np.int64)
        )
    vector = DataStructs.ExplicitBitVect._from_words(
        65, np.array([0, np.iinfo(np.uint64).max], dtype=np.uint64)
    )
    assert vector.GetOnBits() == (64,)


def test_size_mismatch_rejected():
    with pytest.raises(ValueError):
        DataStructs.DiceSimilarity(
            DataStructs.ExplicitBitVect(64),
            DataStructs.ExplicitBitVect(65),
        )


def test_similarity_simd_tail_parity():
    size = 13 * 64
    ours_a = DataStructs.ExplicitBitVect(size, [0, 63, 64, 511, 831])
    ours_b = DataStructs.ExplicitBitVect(size, [0, 64, 129, 511, 830])
    theirs_a = ours_a.to_rdkit()
    theirs_b = ours_b.to_rdkit()
    assert DataStructs.TanimotoSimilarity(ours_a, ours_b) == pytest.approx(
        rdDataStructs.TanimotoSimilarity(theirs_a, theirs_b)
    )


def test_bulk_cache_tracks_vector_mutation():
    query = DataStructs.ExplicitBitVect(128, [1, 64])
    target = DataStructs.ExplicitBitVect(128, [1])
    vectors = [target]
    assert DataStructs.BulkTanimotoSimilarity(query, vectors) == [0.5]
    target.SetBit(64)
    assert DataStructs.BulkTanimotoSimilarity(query, vectors) == [1.0]


@pytest.mark.parametrize("n_targets", [8191, 8192])
def test_bulk_parallel_work_threshold(n_targets):
    n_words = 128
    size = n_words * 64
    query_words = np.zeros(n_words, dtype=np.uint64)
    query_words[0] = 1
    target_words = np.zeros((n_targets, n_words), dtype=np.uint64)
    target_words[::2, 0] = 1
    query = DataStructs.ExplicitBitVect._from_words(size, query_words)
    targets = [
        DataStructs.ExplicitBitVect._from_words(size, row)
        for row in target_words
    ]
    scores = DataStructs.BulkTanimotoSimilarity(query, targets)
    assert scores[0::2] == [1.0] * ((n_targets + 1) // 2)
    assert scores[1::2] == [0.0] * (n_targets // 2)
