"""Bank layer: dataset storage, manifests, QC, release, and promotion.

Layer 3.  ``schema`` is pure data plus hashes and the deterministic split
policy; ``storage`` is the filesystem adapter; ``loader`` is what training
consumes; ``qc`` / ``release`` / ``evaluation`` / ``production`` are the M6
pipeline stages, each fail-closed on missing evidence.

Deliberately empty of imports.  ``schema`` carries the manifest contract and
must stay importable without ``torch``/``torchaudio``, which ``loader`` and
``storage`` both need; re-exporting them here would drag the heavy stack into
every manifest reader.  Import the submodule you need.
"""
