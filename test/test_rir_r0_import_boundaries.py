"""R0 import boundaries: enforce the layering before code starts moving.

``RIR_MODULARIZATION_PLAN.md`` §3.1 defines the dependency direction::

    api / CLI adapter  ->  render, calibration, bank  ->  path_events, scene,
    metrics  ->  physics, contracts  ->  numpy / scipy

Two kinds of guard live here:

1. **Forward guard** — every module under ``puresound/audio/rir/`` is checked
   against the layer table, so a cross-layer import cannot be introduced during
   R1..R7.  Today only ``contracts`` exists, which is the point: the rule is in
   place before the code arrives.
2. **Regression guard** — the flat modules that are *already* free of heavy or
   optional dependencies must stay that way.  ``scene``/``materials``/
   ``bank_manifest`` currently import neither ``torch`` nor a renderer, and the
   plan requires that to hold after the migration.
"""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
RIR_PACKAGE = REPO_ROOT / "puresound" / "audio" / "rir"

#: Layer rank per plan §3.1.  A module may import from its own rank or lower;
#: importing a higher rank is a boundary violation.
LAYER_RANK: dict[str, int] = {
    "contracts": 0,
    "physics": 1,
    "scene": 2,
    "path_events": 2,
    "metrics": 2,
    "render": 3,
    "calibration": 3,
    "bank": 3,
    "api": 4,
}

#: Modules ``contracts`` may never pull in, directly or transitively at import
#: time.  Plan §3.1 rule 1 plus the NumPy/Torch boundary in the risk table.
CONTRACTS_FORBIDDEN_ROOTS = ("torch", "torchaudio", "pyroomacoustics", "cupy")

#: Flat modules that import cleanly today and must keep doing so.
#: ``scipy`` is permitted — plan §3.1 puts ``numpy/scipy`` at the bottom layer.
PURE_FLAT_MODULES = (
    "puresound.audio.rir_scene",
    "puresound.audio.rir_materials",
    "puresound.audio.rir_bank_manifest",
)

#: Optional backends that must never be imported merely by importing a module.
#: Plan §3.1 rule 8 requires lazy import.
OPTIONAL_BACKEND_ROOTS = ("pyroomacoustics", "cupy", "torch", "torchaudio")

#: Canonical package modules that must import without the heavy stack.
#:
#: This is where the property really has to hold: the flat entries below are
#: shims that will eventually go away, but these are the modules everything
#: will import afterwards.  ``bank.schema`` is here deliberately — a manifest
#: reader must not need ``torch``, which is why ``bank/__init__.py`` imports
#: nothing.
PURE_PACKAGE_MODULES = (
    "puresound.audio.rir.contracts",
    "puresound.audio.rir.scene",
    "puresound.audio.rir.scene.schema",
    "puresound.audio.rir.scene.materials",
    "puresound.audio.rir.scene.geometry",
    "puresound.audio.rir.scene.sampling",
    "puresound.audio.rir.metrics",
    "puresound.audio.rir.path_events",
    "puresound.audio.rir.physics.propagation",
    "puresound.audio.rir.physics.impedance.admittance",
    "puresound.audio.rir.physics.wave.fdtd",
    "puresound.audio.rir.bank.schema",
)

#: Flat modules that must not drag an optional renderer in at import time,
#: even though they may legitimately use scipy.
NO_OPTIONAL_BACKEND_MODULES = PURE_FLAT_MODULES + (
    "puresound.audio.rir_metrics",
    "puresound.audio.rir_path_events",
    "puresound.audio.multiband_fdn",
    "puresound.audio.rir_late_coupling",
    "puresound.audio.rir_bank_qc",
    "puresound.audio.rir_bank_release",
    "puresound.audio.rir_bank_evaluation",
    "puresound.audio.rir_bank_production",
)


def _layer_of(path: pathlib.Path) -> str | None:
    """Layer name for a module inside the rir package, if it has one."""

    relative = path.relative_to(RIR_PACKAGE)
    head = relative.parts[0]
    if head.endswith(".py"):
        return head[: -len(".py")]
    return head


def _rir_modules() -> list[pathlib.Path]:
    if not RIR_PACKAGE.is_dir():
        return []
    return [
        path
        for path in sorted(RIR_PACKAGE.rglob("*.py"))
        if path.name != "__init__.py"
    ]


def _imported_rir_layers(tree: ast.AST) -> set[str]:
    """Layers referenced by ``puresound.audio.rir.*`` imports in a module."""

    layers: set[str] = set()
    prefix = "puresound.audio.rir."
    for node in ast.walk(tree):
        module_name: str | None = None
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            module_name = node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(prefix):
                    layers.add(alias.name[len(prefix) :].split(".")[0])
            continue
        if module_name and module_name.startswith(prefix):
            layers.add(module_name[len(prefix) :].split(".")[0])
    return layers


def _import_in_subprocess(module: str, forbidden_roots: tuple[str, ...]) -> list[str]:
    """Import ``module`` in a clean interpreter; return forbidden roots loaded."""

    code = (
        "import sys\n"
        f"import {module}\n"
        "roots = {name.split('.')[0] for name in sys.modules}\n"
        f"hits = sorted(roots & set({forbidden_roots!r}))\n"
        "print(','.join(hits))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"importing {module} failed:\n{result.stdout}\n{result.stderr}"
    )
    payload = result.stdout.strip().splitlines()
    last = payload[-1] if payload else ""
    return [item for item in last.split(",") if item]


class TestContractsLayer:
    def test_contracts_imports_only_stdlib_and_numpy(self):
        loaded = _import_in_subprocess(
            "puresound.audio.rir.contracts", CONTRACTS_FORBIDDEN_ROOTS
        )
        assert not loaded, (
            "puresound.audio.rir.contracts must stay at the bottom layer but "
            f"pulled in: {loaded}"
        )

    def test_contracts_module_has_no_puresound_imports(self):
        """The bottom layer must not depend on any other project module."""

        tree = ast.parse((RIR_PACKAGE / "contracts.py").read_text(encoding="utf-8"))
        offenders: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "puresound"
            ):
                offenders.append(node.module or "")
            elif isinstance(node, ast.Import):
                offenders.extend(
                    a.name for a in node.names if a.name.startswith("puresound")
                )
        assert not offenders, f"contracts imports project modules: {offenders}"

    def test_package_import_does_not_pull_optional_backends(self):
        loaded = _import_in_subprocess(
            "puresound.audio.rir", CONTRACTS_FORBIDDEN_ROOTS
        )
        assert not loaded, f"puresound.audio.rir pulled in: {loaded}"


class TestLayerDirection:
    def test_every_rir_module_lives_in_a_known_layer(self):
        unknown = [
            str(path.relative_to(REPO_ROOT))
            for path in _rir_modules()
            if _layer_of(path) not in LAYER_RANK
        ]
        assert not unknown, (
            f"modules outside the plan's layer table: {unknown}. "
            "Add the layer to LAYER_RANK or move the module."
        )

    def test_no_module_imports_a_higher_layer(self):
        violations: list[str] = []
        for path in _rir_modules():
            layer = _layer_of(path)
            if layer not in LAYER_RANK:
                continue
            rank = LAYER_RANK[layer]
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for imported in _imported_rir_layers(tree):
                if imported not in LAYER_RANK:
                    violations.append(
                        f"{path.relative_to(REPO_ROOT)} imports unknown layer "
                        f"'{imported}'"
                    )
                elif LAYER_RANK[imported] > rank:
                    violations.append(
                        f"{path.relative_to(REPO_ROOT)} ({layer}, rank {rank}) "
                        f"imports {imported} (rank {LAYER_RANK[imported]})"
                    )
        assert not violations, "layer direction violated:\n" + "\n".join(violations)


class TestPackageModuleGuards:
    @pytest.mark.parametrize("module", PURE_PACKAGE_MODULES)
    def test_canonical_modules_stay_free_of_heavy_dependencies(self, module):
        loaded = _import_in_subprocess(module, CONTRACTS_FORBIDDEN_ROOTS)
        assert not loaded, (
            f"{module} must import without the renderer stack, but pulled: {loaded}. "
            "If a package __init__ is re-exporting a heavy submodule, drop the "
            "re-export rather than relaxing this test."
        )


class TestFlatModuleRegressionGuards:
    @pytest.mark.parametrize("module", PURE_FLAT_MODULES)
    def test_pure_modules_stay_free_of_heavy_dependencies(self, module):
        loaded = _import_in_subprocess(module, CONTRACTS_FORBIDDEN_ROOTS)
        assert not loaded, (
            f"{module} imports cleanly today; it must not start pulling {loaded}"
        )

    @pytest.mark.parametrize("module", NO_OPTIONAL_BACKEND_MODULES)
    def test_optional_backends_are_lazily_imported(self, module):
        loaded = _import_in_subprocess(module, OPTIONAL_BACKEND_ROOTS)
        assert not loaded, (
            f"{module} must lazily import optional backends, but importing it "
            f"loaded: {loaded}"
        )
