"""R0 API inventory: freeze the RIR public surface before any code moves.

``RIR_MODULARIZATION_PLAN.md`` §8 item 1 and item 5 require an inventory of the
existing API and a public/legacy/test-only classification of the private
helpers that outside code reaches into.  This module *is* that inventory: the
tables below are the frozen contract, and the tests assert the code still
matches them.

Changing a table is allowed — that is how the surface evolves — but it has to
be a deliberate edit, reviewed alongside whatever migration motivated it.  A
symbol quietly disappearing during R1..R7 fails here instead of failing in a
downstream recipe.

The classification was produced by an AST scan of ``test/``, ``egs/`` and
``puresound/`` for cross-module references, not by reading docstrings.
"""

from __future__ import annotations

import ast
import importlib
import pathlib

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCAN_ROOTS = ("test", "egs", "puresound")


# --------------------------------------------------------------------------
# Frozen public surface
# --------------------------------------------------------------------------

#: Public ``hybrid_rir`` symbols reached by the production CLI under ``egs/``.
#: These carry the strongest compatibility guarantee: R7's shim must re-export
#: every one of them.
HYBRID_RIR_PUBLIC_USED_BY_EGS: frozenset[str] = frozenset(
    {
        "AnalyticModalLowFrequencyBackend",
        "GpuARDPytARDBackend",
        "GpuARDPytARDCuPyBackend",
        "HybridRIRConfig",
        "HybridRIRScene",
        "ImpedanceModalLowFrequencyBackend",
        "PathEventFDNHighFrequencyBackend",
        "PathEventHighFrequencyBackend",
        "PolygonObstacle",
        "PyroomacousticsHighFrequencyBackend",
        "generate_hybrid_rir",
        "hybrid_crossover",
        "sample_material_first_rir_scene",
        "sample_polygon_obstacles",
        "upgrade_hybrid_scene_to_v2",
    }
)

#: Public symbols currently reached only by tests.  Still public — three of
#: them are named in the plan's §2.3 must-preserve list — but they have no
#: production consumer, so R7 may narrow them with less risk.
HYBRID_RIR_PUBLIC_USED_BY_TESTS_ONLY: frozenset[str] = frozenset(
    {
        "apply_obstacle_high_frequency_effects",
        "material_modal_damping_metadata",
        "obstacle_effects_metadata",
        "sample_hybrid_rir_scene",
        "write_hybrid_rir_dataset_item",
    }
)

#: Public symbols with no consumer anywhere in the repository.
#:
#: ``RIRBackend`` is the backend Protocol; it is unreferenced only because
#: backends are structurally typed.  R0 restates it as
#: ``puresound.audio.rir.contracts.BackendCapabilities`` plus the protocol
#: shape, and R2 should make renderers depend on it explicitly.
#:
#: ``PytARDWaveBackend`` is genuinely unreferenced — a dead-code candidate to
#: resolve in R7 rather than to carry into the new package.
HYBRID_RIR_PUBLIC_UNUSED: frozenset[str] = frozenset(
    {
        "PytARDWaveBackend",
        "RIRBackend",
    }
)

HYBRID_RIR_PUBLIC_SURFACE: frozenset[str] = (
    HYBRID_RIR_PUBLIC_USED_BY_EGS
    | HYBRID_RIR_PUBLIC_USED_BY_TESTS_ONLY
    | HYBRID_RIR_PUBLIC_UNUSED
)


# --------------------------------------------------------------------------
# Frozen private-helper classification (plan §8 item 5)
# --------------------------------------------------------------------------

#: Private helpers the production CLI imports.  These are public API in all but
#: name; R1/R2 must give them a real home and a real name rather than leaving
#: ``egs/`` reaching through the underscore.
HYBRID_RIR_PRIVATE_PROMOTE_TO_PUBLIC: frozenset[str] = frozenset(
    {
        "_distance_point_to_polygon",
        "_hybrid_crossover_with_metadata",
        "_max_room_horizontal_distance_from_point",
        "_min_feasible_rt60",
        "_sample_point",
        "_sample_source_in_horizontal_shell",
    }
)

#: Private helpers only tests import.  R2 may keep these private if the
#: behaviour they pin is covered through a public entry point instead.
HYBRID_RIR_PRIVATE_TEST_ONLY: frozenset[str] = frozenset(
    {
        "_align_high_band_direct",
        "_apply_rt60_decay_envelope",
        "_calibrate_pytard_signal",
        "_clip_rir_before_physical_arrival",
        "_obstacle_floor_coverage",
        "_polygons_overlap",
        "_pytard_green_delta_excitation",
        "_solve_modal_ard",
    }
)

HYBRID_RIR_PRIVATE_REACHED_EXTERNALLY: frozenset[str] = (
    HYBRID_RIR_PRIVATE_PROMOTE_TO_PUBLIC | HYBRID_RIR_PRIVATE_TEST_ONLY
)

#: Private helpers in other RIR modules that outside code imports.
#:
#: The M6.5 validator still imports this from the flat ``rir_bank_evaluation``
#: path; the shim re-exports it explicitly.  The entry disappears once that
#: validator moves to ``puresound.audio.rir.bank.evaluation``.
OTHER_PRIVATE_REACHED_EXTERNALLY: frozenset[tuple[str, str]] = frozenset(
    {
        ("rir_bank_evaluation", "_paired_t_confidence_interval"),
    }
)

#: Modules whose ``__all__`` is the declared surface.  ``hybrid_rir`` is absent
#: on purpose: it has no ``__all__``, which is exactly why this file exists.
MODULES_WITH_EXPLICIT_ALL: tuple[str, ...] = (
    "multiband_fdn",
    "rir_bank_evaluation",
    "rir_bank_manifest",
    "rir_bank_production",
    "rir_bank_qc",
    "rir_bank_release",
    "rir_late_coupling",
    "rir_materials",
    "rir_metrics",
    "rir_path_events",
    "rir_scene",
    "spatial_rir",
)


def _module_symbols(module_name: str) -> tuple[frozenset[str], frozenset[str]]:
    """Return (public, private) symbols a module actually defines itself."""

    module = importlib.import_module(f"puresound.audio.{module_name}")
    own = {
        name
        for name, obj in vars(module).items()
        if not name.startswith("__")
        and getattr(obj, "__module__", None) == module.__name__
    }
    public = frozenset(n for n in own if not n.startswith("_"))
    private = frozenset(n for n in own if n.startswith("_"))
    return public, private


def _is_compatibility_shim(path: pathlib.Path) -> bool:
    """True for the R1..R7 re-export modules, which declare themselves as such."""

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return False
    docstring = ast.get_docstring(tree) or ""
    return docstring.startswith("Compatibility shim")


def _module_name_of(path: pathlib.Path) -> str | None:
    """Dotted module name for a file inside the ``puresound`` package."""

    relative = path.relative_to(REPO_ROOT)
    if relative.parts[0] != "puresound":
        return None
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _cross_module_private_references() -> set[tuple[str, str]]:
    """AST-scan for ``from puresound.audio.X import _y`` across package lines.

    Siblings inside one package sharing a private helper is ordinary Python and
    is not reported: ``rir.path_events.generator`` importing
    ``rir.path_events.geometry._ordered_shoebox_image_path`` keeps that helper
    package-internal, which is exactly where it belongs.  What this flags is a
    module reaching into the privates of a *different* package — the coupling
    the migration is meant to remove.
    """

    found: set[tuple[str, str]] = set()
    for root in SCAN_ROOTS:
        for path in (REPO_ROOT / root).rglob("*.py"):
            if _is_compatibility_shim(path):
                # A shim exists to re-export, private compatibility aliases
                # included.  Flagging it would flag the mechanism that keeps
                # the legacy import paths alive.
                continue
            importer = _module_name_of(path)
            importer_package = (
                importer.rsplit(".", 1)[0] if importer and "." in importer else importer
            )
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                if not node.module or not node.module.startswith("puresound.audio"):
                    continue
                if node.module == importer:
                    continue
                target_package = (
                    node.module.rsplit(".", 1)[0]
                    if "." in node.module
                    else node.module
                )
                if importer_package is not None and target_package == importer_package:
                    continue
                for alias in node.names:
                    if alias.name.startswith("_"):
                        found.add((node.module.split(".")[-1], alias.name))
    return found


class TestPublicSurface:
    def test_hybrid_rir_declares_the_frozen_surface(self):
        """``__all__`` is the contract, whether or not the code still lives here.

        R1 started moving implementations into ``puresound.audio.rir`` and
        re-exporting them.  A re-exported symbol is no longer *defined* in
        ``hybrid_rir``, but it is still part of its public surface, so the
        surface is pinned by ``__all__`` rather than by definition site.
        """

        module = importlib.import_module("puresound.audio.hybrid_rir")
        declared = set(getattr(module, "__all__"))
        assert declared == set(HYBRID_RIR_PUBLIC_SURFACE), (
            "hybrid_rir.__all__ drifted from the R0 inventory: "
            f"added={sorted(declared - HYBRID_RIR_PUBLIC_SURFACE)} "
            f"removed={sorted(HYBRID_RIR_PUBLIC_SURFACE - declared)}"
        )

    def test_hybrid_rir_defines_no_public_symbol_outside_the_inventory(self):
        """Catch a genuinely new public definition, not a migration re-export."""

        public, _ = _module_symbols("hybrid_rir")
        added = public - HYBRID_RIR_PUBLIC_SURFACE
        assert not added, (
            f"hybrid_rir gained public symbols not in the R0 inventory: {sorted(added)}. "
            "Add them to the appropriate table if this is intended."
        )

    @pytest.mark.parametrize("symbol", sorted(HYBRID_RIR_PUBLIC_SURFACE))
    def test_every_frozen_public_symbol_is_importable(self, symbol):
        module = importlib.import_module("puresound.audio.hybrid_rir")
        assert hasattr(module, symbol)

    def test_plan_must_preserve_list_is_covered(self):
        """Every API named in the plan's §2.3 is in the frozen surface."""

        plan_required = {
            "HybridRIRConfig",
            "HybridRIRScene",
            "PolygonObstacle",
            "AnalyticModalLowFrequencyBackend",
            "ImpedanceModalLowFrequencyBackend",
            "GpuARDPytARDBackend",
            "GpuARDPytARDCuPyBackend",
            "PyroomacousticsHighFrequencyBackend",
            "PathEventHighFrequencyBackend",
            "PathEventFDNHighFrequencyBackend",
            "generate_hybrid_rir",
            "sample_hybrid_rir_scene",
            "sample_material_first_rir_scene",
            "upgrade_hybrid_scene_to_v2",
            "hybrid_crossover",
            "write_hybrid_rir_dataset_item",
        }
        assert plan_required <= HYBRID_RIR_PUBLIC_SURFACE

    def test_classification_tables_are_disjoint(self):
        assert not (
            HYBRID_RIR_PUBLIC_USED_BY_EGS & HYBRID_RIR_PUBLIC_USED_BY_TESTS_ONLY
        )
        assert not (HYBRID_RIR_PUBLIC_USED_BY_EGS & HYBRID_RIR_PUBLIC_UNUSED)
        assert not (
            HYBRID_RIR_PUBLIC_USED_BY_TESTS_ONLY & HYBRID_RIR_PUBLIC_UNUSED
        )
        assert not (
            HYBRID_RIR_PRIVATE_PROMOTE_TO_PUBLIC & HYBRID_RIR_PRIVATE_TEST_ONLY
        )

    @pytest.mark.parametrize("module_name", MODULES_WITH_EXPLICIT_ALL)
    def test_declared_all_matches_defined_public_symbols(self, module_name):
        module = importlib.import_module(f"puresound.audio.{module_name}")
        declared = set(getattr(module, "__all__"))
        missing = {name for name in declared if not hasattr(module, name)}
        assert not missing, f"{module_name}.__all__ names absent symbols: {sorted(missing)}"


class TestPrivateHelperClassification:
    def test_private_helpers_reached_externally_are_classified(self):
        references = _cross_module_private_references()
        hybrid = {sym for mod, sym in references if mod == "hybrid_rir"}
        unclassified = hybrid - HYBRID_RIR_PRIVATE_REACHED_EXTERNALLY
        assert not unclassified, (
            "new code reaches into hybrid_rir private helpers without a "
            f"classification: {sorted(unclassified)}. Classify them as "
            "promote-to-public or test-only before the migration moves them."
        )

    def test_classified_helpers_are_still_reachable(self):
        """External callers must keep working, wherever the code now lives.

        R1 moved some of these into ``puresound.audio.rir.scene`` and aliased
        them back, so reachability — not definition site — is the contract.
        """

        module = importlib.import_module("puresound.audio.hybrid_rir")
        missing = sorted(
            name
            for name in HYBRID_RIR_PRIVATE_REACHED_EXTERNALLY
            if not hasattr(module, name)
        )
        assert not missing, (
            f"classified private helpers are no longer importable: {missing}. "
            "Their external callers are now broken."
        )

    def test_other_module_private_references_are_classified(self):
        references = _cross_module_private_references()
        others = {(mod, sym) for mod, sym in references if mod != "hybrid_rir"}
        unclassified = others - OTHER_PRIVATE_REACHED_EXTERNALLY
        assert not unclassified, (
            f"unclassified cross-module private imports: {sorted(unclassified)}"
        )

    def test_promotion_candidates_are_reached_from_production_code(self):
        """Helpers marked promote-to-public must have a non-test caller."""

        production_hits: set[str] = set()
        for path in (REPO_ROOT / "egs").rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == (
                    "puresound.audio.hybrid_rir"
                ):
                    production_hits.update(
                        a.name for a in node.names if a.name.startswith("_")
                    )
        assert HYBRID_RIR_PRIVATE_PROMOTE_TO_PUBLIC <= production_hits, (
            "these are classified as production-reached but no egs/ caller "
            f"imports them: {sorted(HYBRID_RIR_PRIVATE_PROMOTE_TO_PUBLIC - production_hits)}"
        )
