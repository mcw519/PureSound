"""Compatibility shim.

The implementation moved to :mod:`puresound.audio.rir.bank.qc` during R6 of
``RIR_MODULARIZATION_PLAN.md``.  This module re-exports it unchanged; prefer
the new path in new code.
"""

from puresound.audio.rir.bank.qc import (
    DEFAULT_QC_SUMMARY_NAME,
    RIRBankQCPolicy,
    RIR_BANK_ITEM_QC_SCHEMA_VERSION,
    RIR_BANK_QC_POLICY_ID,
    RIR_BANK_QC_RELEASE_SCHEMA_VERSION,
    audit_rir_bank_qc_release,
    evaluate_rir_bank_item,
    run_rir_bank_qc,
)

__all__ = [
    "DEFAULT_QC_SUMMARY_NAME",
    "RIRBankQCPolicy",
    "RIR_BANK_ITEM_QC_SCHEMA_VERSION",
    "RIR_BANK_QC_POLICY_ID",
    "RIR_BANK_QC_RELEASE_SCHEMA_VERSION",
    "audit_rir_bank_qc_release",
    "evaluate_rir_bank_item",
    "run_rir_bank_qc",
]
