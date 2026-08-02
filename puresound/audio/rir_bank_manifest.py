"""Compatibility shim.

The implementation moved to :mod:`puresound.audio.rir.bank.schema` during R6 of
``RIR_MODULARIZATION_PLAN.md``.  This module re-exports it unchanged; prefer
the new path in new code.
"""

from puresound.audio.rir.bank.schema import (
    BANK_SPLITS,
    BankGeneratorProvenance,
    BankRendererProfile,
    BankSplitIndex,
    BankSplitPolicy,
    M6_SPLIT_POLICY,
    RIRBankItem,
    RIRBankManifest,
    RIR_BANK_AUDIT_SCHEMA_VERSION,
    RIR_BANK_MANIFEST_SCHEMA_VERSION,
    audit_rir_bank_manifest,
    canonical_json_bytes,
    canonical_json_sha256,
    canonicalize_float_wav_header,
    sha256_file,
    split_index_rows,
    task_plan_rows,
    write_split_indexes,
)

__all__ = [
    "BANK_SPLITS",
    "BankGeneratorProvenance",
    "BankRendererProfile",
    "BankSplitIndex",
    "BankSplitPolicy",
    "M6_SPLIT_POLICY",
    "RIRBankItem",
    "RIRBankManifest",
    "RIR_BANK_AUDIT_SCHEMA_VERSION",
    "RIR_BANK_MANIFEST_SCHEMA_VERSION",
    "audit_rir_bank_manifest",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "canonicalize_float_wav_header",
    "sha256_file",
    "split_index_rows",
    "task_plan_rows",
    "write_split_indexes",
]
