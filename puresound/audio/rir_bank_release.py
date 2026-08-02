"""Compatibility shim.

The implementation moved to :mod:`puresound.audio.rir.bank.release` during R6 of
``RIR_MODULARIZATION_PLAN.md``.  This module re-exports it unchanged; prefer
the new path in new code.
"""

from puresound.audio.rir.bank.release import (
    DEFAULT_RELEASE_MANIFEST_NAME,
    RIRBankReleaseManifest,
    RIR_BANK_DISTRIBUTION_SCHEMA_VERSION,
    RIR_BANK_RELEASE_SCHEMA_VERSION,
    ReleaseIndex,
    ReleaseRecipe,
    ReleaseVariant,
    audit_m6_variant_release,
    build_m6_variant_release,
    compute_bank_distribution,
)

__all__ = [
    "DEFAULT_RELEASE_MANIFEST_NAME",
    "RIRBankReleaseManifest",
    "RIR_BANK_DISTRIBUTION_SCHEMA_VERSION",
    "RIR_BANK_RELEASE_SCHEMA_VERSION",
    "ReleaseIndex",
    "ReleaseRecipe",
    "ReleaseVariant",
    "audit_m6_variant_release",
    "build_m6_variant_release",
    "compute_bank_distribution",
]
