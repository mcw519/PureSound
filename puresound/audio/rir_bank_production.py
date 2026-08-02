"""Compatibility shim.

The implementation moved to :mod:`puresound.audio.rir.bank.production` during R6 of
``RIR_MODULARIZATION_PLAN.md``.  This module re-exports it unchanged; prefer
the new path in new code.
"""

from puresound.audio.rir.bank.production import (
    DEFAULT_PRODUCTION_DECISION_NAME,
    M6_PRODUCTION_DECISION_SCHEMA_VERSION,
    M6_PRODUCTION_EVIDENCE_SCHEMA_VERSION,
    M6_PRODUCTION_SIGNOFF_SCHEMA_VERSION,
    PRODUCTION_DECISION_CHECK_NAMES,
    REQUIRED_EVIDENCE_KINDS,
    REQUIRED_RECIPE_IDS,
    audit_m6_production_evidence,
    build_m6_production_decision,
    validate_m6_production_certificate,
)

__all__ = [
    "DEFAULT_PRODUCTION_DECISION_NAME",
    "M6_PRODUCTION_DECISION_SCHEMA_VERSION",
    "M6_PRODUCTION_EVIDENCE_SCHEMA_VERSION",
    "M6_PRODUCTION_SIGNOFF_SCHEMA_VERSION",
    "PRODUCTION_DECISION_CHECK_NAMES",
    "REQUIRED_EVIDENCE_KINDS",
    "REQUIRED_RECIPE_IDS",
    "audit_m6_production_evidence",
    "build_m6_production_decision",
    "validate_m6_production_certificate",
]
