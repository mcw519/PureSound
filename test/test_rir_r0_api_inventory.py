"""RIR API inventory: the package's public surface, frozen and enforced.

Started life as the R0 pre-migration freeze of ``puresound.audio.hybrid_rir``
(``RIR_EXP_LOG.md`` §8 items 1 and 5).  That module is gone: the
migration finished, the compatibility shims were removed, and every caller now
imports from ``puresound.audio.rir``.  The file kept its job and changed its
subject.

What it still guarantees:

- every layer module declares ``__all__``, and every declared name resolves;
- no module reaches into the privates of a *different* package;
- symbols recorded as having no consumer really have none, so a dead-code
  claim cannot quietly become false.

Changing a table is allowed — that is how the surface evolves — but it has to
be a deliberate edit reviewed alongside whatever motivated it.
"""

from __future__ import annotations

import ast
import importlib
import pathlib

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
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

#: ``render.hybrid`` is orchestration and owns exactly one public entry point.
#: It used to re-export twenty-one more so the ``hybrid_rir`` shim could serve
#: them; with the shim gone, those callers import the canonical module instead.
HYBRID_PUBLIC_SURFACE: frozenset[str] = frozenset({"generate_hybrid_rir"})

#: Cross-package private imports.  Siblings inside one package sharing a
#: private helper is ordinary Python and is not recorded here; this table is
#: for a module reaching across a package boundary.
#:
#: The M6.5 validator reuses this statistical helper directly.  It is the last
#: production one left — the migration turned the other fifteen into real
#: public names.
#:
#: The two geometry helpers are reached by a test, not by shipping code.
#: ``apply_scene_object_visibility`` short-circuits on axis-aligned bounds
#: before the exact prism test, and that is only sound if the cheap test never
#: rejects a real intersection.  Asserting that invariant means calling the
#: cheap test and the exact test on the same input, which needs both names.
#: Promoting them would put an implementation detail in the public surface;
#: testing only through the public function would not isolate the property.
#:
#: ``_recipe_semantics_valid`` and ``_production_decision_components`` are reached
#: by tests, not by shipping code.  The measured ingest and the evidence producers
#: exist to turn specific M6.6 decision checks from False into True, and these two
#: are the decision's own definitions of those checks — the first for whether the
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
    }
)

#: Public symbols with no consumer anywhere in the repository.
#:
#: ``PytARDWaveBackend`` is a genuine dead-code candidate: it predates the
#: ``GpuARDPytARDBackend`` adapter and nothing constructs it.  It is kept
#: listed rather than deleted so the claim stays checked — if something starts
#: using it, this test says so; if it is removed, this entry goes with it.
#:
#: ``_sample_source_in_shell`` is its private counterpart in scene sampling,
#: superseded by ``sample_source_in_horizontal_shell``.
KNOWN_UNUSED: frozenset[tuple[str, str]] = frozenset(
    {
        ("puresound.audio.rir.render.low_frequency.pytard", "PytARDWaveBackend"),
        ("puresound.audio.rir.scene.sampling", "_sample_source_in_shell"),
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


def _imported_names(module: str) -> set[str]:
    """Names imported from ``module`` by code outside its own package.

    A package ``__init__`` re-exporting one of its own submodule's symbols is
    not a consumer — that is just the package declaring its surface — so the
    same-package exclusion used for private imports applies here too.
    """

    package = module.rsplit(".", 1)[0] if "." in module else module
    names: set[str] = set()
    for path in _iter_python_files():
        importer = _module_name_of(path)
        if importer == module:
            continue
        if importer is not None and (
            importer == package or importer.startswith(package + ".")
        ):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == module:
                names.update(alias.name for alias in node.names)
    return names


class TestDeclaredSurface:
    @pytest.mark.parametrize("module_name", MODULES_WITH_EXPLICIT_ALL)
    def test_module_declares_all(self, module_name):
        module = importlib.import_module(module_name)
        assert hasattr(module, "__all__"), f"{module_name} must declare __all__"

    @pytest.mark.parametrize("module_name", MODULES_WITH_EXPLICIT_ALL)
    def test_every_declared_name_resolves(self, module_name):
        module = importlib.import_module(module_name)
        missing = sorted(n for n in module.__all__ if not hasattr(module, n))
        assert not missing, f"{module_name}.__all__ names absent symbols: {missing}"

    def test_hybrid_owns_only_the_orchestration_entry_point(self):
        module = importlib.import_module("puresound.audio.rir.render.hybrid")
        assert set(module.__all__) == set(HYBRID_PUBLIC_SURFACE), (
            "render.hybrid should stay a thin orchestration module; re-exporting "
            "backends from here is what the compatibility shim used to do."
        )

    def test_api_convenience_reexports_resolve(self):
        """``api`` is a convenience bundle, not a stability boundary.

        It used to be described as the package's stable public entry point and
        this test asserted a list of "essential" names to hold that line. The
        line was fictional: nothing imports the module, and it omits 25 of the
        40 symbols the top-level ``egs/rir_generation`` scripts actually need.
        Enforcing a contract no caller depends on only makes the contract
        harder to change for no one's benefit -- see the module docstring.

        What is still worth checking is that the re-exports are not broken:
        a name in ``__all__`` that no longer resolves means a layer module
        renamed something and this bundle silently rotted.
        """
        api = importlib.import_module("puresound.audio.rir.api")
        missing = sorted(n for n in api.__all__ if not hasattr(api, n))
        assert not missing, f"api.__all__ names absent symbols: {missing}"


class TestPrivateBoundaries:
    def test_cross_package_private_imports_are_recorded(self):
        found = _cross_package_private_references()
        unrecorded = sorted(found - CROSS_PACKAGE_PRIVATE_IMPORTS)
        assert not unrecorded, (
            f"new cross-package private imports: {unrecorded}. Either give the "
            "helper a public name in its own module, or record it here with a "
            "reason."
        )

    def test_recorded_private_imports_still_resolve(self):
        for module_name, symbol in sorted(CROSS_PACKAGE_PRIVATE_IMPORTS):
            module = importlib.import_module(module_name)
            assert hasattr(module, symbol), (
                f"{module_name}.{symbol} is recorded as an external dependency "
                "but no longer exists"
            )

    def test_no_legacy_flat_rir_modules_remain(self):
        """The migration removed the shims; nothing should recreate them."""

        legacy = sorted(
            path.name
            for path in (REPO_ROOT / "puresound" / "audio").glob("*.py")
            if path.read_text(encoding="utf-8").startswith('"""Compatibility shim')
        )
        assert not legacy, (
            f"compatibility shims reappeared: {legacy}. New code should import "
            "from puresound.audio.rir directly."
        )


class TestKnownUnused:
    @pytest.mark.parametrize("module_name,symbol", sorted(KNOWN_UNUSED))
    def test_still_exists(self, module_name, symbol):
        module = importlib.import_module(module_name)
        assert hasattr(module, symbol), (
            f"{module_name}.{symbol} was removed — drop it from KNOWN_UNUSED too"
        )

    @pytest.mark.parametrize("module_name,symbol", sorted(KNOWN_UNUSED))
    def test_still_has_no_consumer(self, module_name, symbol):
        if symbol.startswith("_"):
            consumers = {
                mod
                for mod, name in _cross_package_private_references()
                if name == symbol
            }
        else:
            consumers = (
                {module_name} if symbol in _imported_names(module_name) else set()
            )
        assert not consumers, (
            f"{module_name}.{symbol} is recorded as unused but {sorted(consumers)} "
            "imports it now; remove it from KNOWN_UNUSED."
        )
