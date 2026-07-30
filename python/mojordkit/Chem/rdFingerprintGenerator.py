from __future__ import annotations

from ._fingerprints import GetMorganFingerprintAsBitVect


class FingerprintGenerator32:
    def __init__(
        self,
        radius=3,
        includeChirality=False,
        useBondTypes=True,
        fpSize=2048,
        includeRedundantEnvironments=False,
    ):
        self.radius = radius
        self.includeChirality = includeChirality
        self.useBondTypes = useBondTypes
        self.fpSize = fpSize
        self.includeRedundantEnvironments = includeRedundantEnvironments

    def GetFingerprint(
        self,
        mol,
        fromAtoms=None,
        ignoreAtoms=None,
        confId=-1,
        customAtomInvariants=None,
        customBondInvariants=None,
        additionalOutput=None,
    ):
        if (
            (ignoreAtoms is not None and len(ignoreAtoms))
            or (customBondInvariants is not None and len(customBondInvariants))
            or additionalOutput is not None
        ):
            raise NotImplementedError(
                "ignoreAtoms, customBondInvariants, and AdditionalOutput are not covered"
            )
        return GetMorganFingerprintAsBitVect(
            mol,
            self.radius,
            nBits=self.fpSize,
            invariants=customAtomInvariants,
            fromAtoms=fromAtoms,
            useChirality=self.includeChirality,
            useBondTypes=self.useBondTypes,
            includeRedundantEnvironments=self.includeRedundantEnvironments,
        )


def GetMorganGenerator(
    radius=3,
    countSimulation=False,
    includeChirality=False,
    useBondTypes=True,
    onlyNonzeroInvariants=False,
    includeRingMembership=True,
    countBounds=None,
    fpSize=2048,
    atomInvariantsGenerator=None,
    bondInvariantsGenerator=None,
):
    if (
        countSimulation
        or onlyNonzeroInvariants
        or not includeRingMembership
        or countBounds is not None
        or atomInvariantsGenerator is not None
        or bondInvariantsGenerator is not None
    ):
        raise NotImplementedError("the requested non-default Morgan generator option is not covered")
    return FingerprintGenerator32(
        radius, includeChirality, useBondTypes, fpSize
    )
