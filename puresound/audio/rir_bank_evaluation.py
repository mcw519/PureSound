"""Compatibility shim.

The implementation moved to :mod:`puresound.audio.rir.bank.evaluation` during R6 of
``RIR_MODULARIZATION_PLAN.md``.  This module re-exports it unchanged; prefer
the new path in new code.
"""

from puresound.audio.rir.bank.evaluation import (
    _paired_t_confidence_interval,
    M6_DOWNSTREAM_SCHEMA_VERSION,
    M6_EVALUATION_SCHEMA_VERSION,
    M6_LISTENING_SCHEMA_VERSION,
    M6_THROUGHPUT_SCHEMA_VERSION,
    compare_release_distributions,
    evaluate_m6_release,
    validate_downstream_report,
    validate_listening_report,
    validate_throughput_report,
)

__all__ = [
    "M6_DOWNSTREAM_SCHEMA_VERSION",
    "M6_EVALUATION_SCHEMA_VERSION",
    "M6_LISTENING_SCHEMA_VERSION",
    "M6_THROUGHPUT_SCHEMA_VERSION",
    "compare_release_distributions",
    "evaluate_m6_release",
    "validate_downstream_report",
    "validate_listening_report",
    "validate_throughput_report",
]

# The M6.5 validator reuses this statistical helper directly; the R0 inventory
# records it as the one cross-module private reference outside hybrid_rir.
__all__.append("_paired_t_confidence_interval")
