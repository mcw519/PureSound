"""The public surface of ``puresound.audio.rir``, enforced.

What it guarantees:

- every layer module declares ``__all__``, and every declared name resolves;
- ``render.hybrid`` stays a thin orchestration module;
- no module reaches into the privates of a *different* package unless the
  reference is recorded below with a reason, and every recorded one resolves.

Changing a table is allowed -- that is how the surface evolves -- but it has to
be a deliberate edit reviewed alongside whatever motivated it.
"""

from __future__ import annotations

import ast
import importlib
import pathlib

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
RIR_PACKAGE = REPO_ROOT / "puresound" / "audio" / "rir"
SCAN_ROOTS = ("test", "egs", "puresound")


#: Every module expected to declare an explicit ``__all__``.  Leaf modules that
#: exist only to be imported by a sibling (``metrics.core``) are exempt.
MODULES_WITH_EXPLICIT_ALL: tuple[str, ...] = (
    "puresound.audio.rir.api",
    "puresound.audio.rir.contracts",
    "puresound.audio.rir.bank.storage",
    "puresound.audio.rir.metrics",
    "puresound.audio.rir.path_events",
    "puresound.audio.rir.path_events.schema",
    "puresound.audio.rir.path_events.geometry",
    "puresound.audio.rir.path_events.renderer",
    "puresound.audio.rir.render.arrays",
    "puresound.audio.rir.render.backend",
    "puresound.audio.rir.render.crossover",
    "puresound.audio.rir.render.high_frequency",
    "puresound.audio.rir.render.hybrid",
    "puresound.audio.rir.render.low_frequency",
    "puresound.audio.rir.scene",
    "puresound.audio.rir.scene.geometry",
    "puresound.audio.rir.scene.materials",
    "puresound.audio.rir.scene.sampling",
    "puresound.audio.rir.scene.schema",
)

#: ``render.hybrid`` is orchestration and owns exactly one public entry point;
#: backends are imported from their own modules.
HYBRID_PUBLIC_SURFACE: frozenset[str] = frozenset({"generate_hybrid_rir"})

#: Cross-package private imports.  Siblings inside one package sharing a
#: private helper is ordinary Python and is not recorded here; this table is
#: for a module reaching across a package boundary.
#:
#: The release-evaluation validator reuses this statistical helper directly.
#:
#: The two geometry helpers are reached by a test, not by shipping code.
#: ``apply_scene_object_visibility`` short-circuits on axis-aligned bounds
#: before the exact prism test, and that is only sound if the cheap test never
#: rejects a real intersection.  Asserting that invariant means calling the
#: cheap test and the exact test on the same input, which needs both names.
#: Promoting them would put an implementation detail in the public surface;
#: testing only through the public function would not isolate the property.
#:
#: ``_emission`` is reached only by dynamic-scene tests. Complete speech-clip
#: placement, gain and partial noise loops are asserted before propagation;
#: testing their sample boundaries after fractional delays and room filters
#: would conflate emission scheduling with acoustic rendering. It remains an
#: implementation detail rather than a public dry-audio API.
#:
#: ``_recipe_semantics_valid`` and ``_production_decision_components`` are reached
#: by tests, not by shipping code.  The measured ingest and the evidence producers
#: exist to turn specific production-decision checks from False into True, and
#: these two are the decision's own definitions of those checks — the first for whether the
#: real and mixed recipes are ready, the second for the full check set including
#: renderer approval.  Asserting against a re-implementation in the test would let
#: producer and decision drift apart, which is the failure the tests exist to catch.
CROSS_PACKAGE_PRIVATE_IMPORTS: frozenset[tuple[str, str]] = frozenset(
    {
        ("puresound.audio.rir.bank.evaluation", "_paired_t_confidence_interval"),
        ("puresound.audio.rir.bank.production", "_production_decision_components"),
        ("puresound.audio.rir.bank.production", "_recipe_semantics_valid"),
        ("puresound.audio.rir.path_events.geometry", "_object_bounds"),
        ("puresound.audio.rir.path_events.geometry", "_segment_misses_bounds"),
        ("puresound.audio.rir.render.dynamic", "_emission"),
    }
)


def _module_name_of(path: pathlib.Path) -> str | None:
    relative = path.relative_to(REPO_ROOT)
    if relative.parts[0] != "puresound":
        return None
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _iter_python_files():
    for root in SCAN_ROOTS:
        yield from (REPO_ROOT / root).rglob("*.py")


def _cross_package_private_references() -> set[tuple[str, str]]:
    """``from puresound.audio.X import _y`` where X is in another package."""

    found: set[tuple[str, str]] = set()
    for path in _iter_python_files():
        importer = _module_name_of(path)
        importer_package = (
            importer.rsplit(".", 1)[0] if importer and "." in importer else importer
        )
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level:
                continue
            if not node.module or not node.module.startswith("puresound.audio"):
                continue
            if node.module == importer:
                continue
            target_package = (
                node.module.rsplit(".", 1)[0] if "." in node.module else node.module
            )
            if importer_package is not None and target_package == importer_package:
                continue
            for alias in node.names:
                if alias.name.startswith("_"):
                    found.add((node.module, alias.name))
    return found


@pytest.mark.parametrize("module_name", MODULES_WITH_EXPLICIT_ALL)
def test_module_declares_all_and_every_declared_name_resolves(module_name):
    module = importlib.import_module(module_name)
    assert hasattr(module, "__all__"), f"{module_name} must declare __all__"
    missing = sorted(n for n in module.__all__ if not hasattr(module, n))
    assert not missing, f"{module_name}.__all__ names absent symbols: {missing}"


def test_hybrid_owns_only_the_orchestration_entry_point():
    module = importlib.import_module("puresound.audio.rir.render.hybrid")
    assert set(module.__all__) == set(HYBRID_PUBLIC_SURFACE), (
        "render.hybrid should stay a thin orchestration module; import "
        "backends from their own modules instead of re-exporting them here."
    )


def test_cross_package_private_imports_are_recorded_and_resolve():
    found = _cross_package_private_references()
    unrecorded = sorted(found - CROSS_PACKAGE_PRIVATE_IMPORTS)
    assert not unrecorded, (
        f"new cross-package private imports: {unrecorded}. Either give the "
        "helper a public name in its own module, or record it here with a "
        "reason."
    )
    for module_name, symbol in sorted(CROSS_PACKAGE_PRIVATE_IMPORTS):
        module = importlib.import_module(module_name)
        assert hasattr(module, symbol), (
            f"{module_name}.{symbol} is recorded as an external dependency "
            "but no longer exists"
        )
