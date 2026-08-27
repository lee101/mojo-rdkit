# mojo-rdkit

`mojo-rdkit` is a focused Mojo port of compute-heavy RDKit operations for
molecular fingerprints and substructure search. It keeps RDKit's chemistry
parsing and perception layer, then passes compact graph and bit-vector buffers
to an allocation-free Mojo shared library.

The Python package is named `mojordkit`, so it can be installed beside RDKit
for parity testing and gradual adoption. Covered functions retain RDKit names,
argument order, defaults, and return behavior.

## Covered subset

- `Chem.GetMorganFingerprintAsBitVect`,
  `Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect`, and
  `Chem.AllChem.GetMorganFingerprintAsBitVect`
- Morgan radii, bit sizes, connectivity or feature invariants, custom atom
  invariants, selected centers, bond types, chirality, redundant environments,
  and `bitInfo`
- `Chem.rdFingerprintGenerator.GetMorganGenerator(...).GetFingerprint(...)`
  for the common non-counted generator configuration
- `DataStructs.ExplicitBitVect` and Tanimoto, Dice, Cosine, Sokal, Russel,
  Kulczynski, and McConnaughey similarities
- `BulkTanimotoSimilarity`, `BulkDiceSimilarity`, and
  `FingerprintSimilarity`
- `Mol.GetSubstructMatch`, `Mol.GetSubstructMatches`, and
  `Mol.HasSubstructMatch` for molecule queries and local SMARTS atom/bond
  predicates, including uniqueness and `maxMatches`

RDKit remains a dependency for SMILES/SMARTS parsing, aromaticity, ring and
stereochemistry perception, and initial Morgan atom invariants. Recursive
SMARTS, chiral matching, and query-query matching are delegated to RDKit
because those predicates require molecule-level query context. Sparse/count
fingerprints, count simulation, RDK, Atom Pair, Topological Torsion, MACCS,
pharmacophore fingerprints, reactions, conformers, descriptors, depiction,
and the rest of RDKit are not covered.

## Install and verify

```bash
pixi install
pixi run build
pixi run test
```

The build task compiles `src/kernels.mojo` with `mojo build --emit shared-lib`
to `dist/libmojo-rdkit.so`. The test suite compares results directly with the
real conda-forge RDKit package.

## Usage

```python
from mojordkit import Chem, DataStructs

aspirin = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
salicylic_acid = Chem.MolFromSmiles("O=C(O)c1ccccc1O")

aspirin_fp = Chem.GetMorganFingerprintAsBitVect(aspirin, radius=2, nBits=2048)
salicylic_fp = Chem.GetMorganFingerprintAsBitVect(
    salicylic_acid, radius=2, nBits=2048
)
print(DataStructs.TanimotoSimilarity(aspirin_fp, salicylic_fp))

carboxyl = Chem.MolFromSmarts("C(=O)O")
print(aspirin.GetSubstructMatches(carboxyl))
```

Run it inside the environment with `pixi run python example.py`.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz using
Python 3.13.14. Values are the best of five warm runs. Molecule parsing and
benchmark fixture construction are outside the timed regions.

| operation | mojo-rdkit | RDKit | RDKit time / Mojo time |
|---|---:|---:|---:|
| Morgan radius=2, 800 molecules | 4.544 ms | 9.137 ms | 2.01x faster |
| Bulk Tanimoto, 5,000 x 2,048-bit | 0.480 ms | 0.613 ms | 1.28x faster |
| Substructure C8 in C80, all matches | 0.094 ms | 0.150 ms | 1.60x faster |

Warm calls reuse immutable molecule topology, query compatibility, and native
scratch buffers. Returned fingerprints and match tuples remain independent,
and exposing mutable atoms or bonds disables molecule caching. Bulk similarity
keeps shared contiguous word matrices zero-copy across the FFI boundary and
caches immutable query and target population counts.

No GPU path is included. Similarity is a low-arithmetic-intensity streaming
bit scan, while Morgan and substructure kernels operate on small, branchy graph
frontiers. This port uses CPU SIMD and size-thresholded CPU parallelism.

## How it works

Python turns each molecule into contiguous NumPy buffers: CSR atom adjacency,
bond invariants, atom invariants, and query compatibility matrices. Buffer
addresses cross the C ABI as 64-bit integers and are reconstructed as mutable
typed pointers inside Mojo. Python owns every input, output, and scratch
allocation.

Morgan environments use RDKit-compatible 32-bit hash combining. Each radius
expands a bond-set neighborhood, sorts `(bond invariant, neighbor invariant)`
pairs, suppresses duplicate environments, and writes codes plus `bitInfo`
metadata. Fingerprints are stored as contiguous little-endian `uint64` words.
Similarity kernels scan those words directly using SIMD popcount chunks,
lane-local accumulation, and a scalar remainder. Large independent target
batches are split into native chunks across a bounded persistent CPU worker
pool after a 1,048,576-word work threshold. Substructure search uses an iterative
injective graph mapping, target CSR neighbor traversal, and atom and bond
predicate matrices without allocation or recursion inside the shared library.

The entire native implementation is one Mojo compilation unit to avoid paying
the fixed shared-library build cost multiple times.
