import pytest

from egs.rir_generation import generate_hybrid_rir
from egs.rir_generation import generate_m6_bank
from puresound.audio.rir.render.high_frequency import (
    PathEventFDNHighFrequencyBackend,
    PyroomacousticsHighFrequencyBackend,
)
from puresound.audio.rir.scene.sampling import HybridRIRScene


def _scene(mic_pos, source_pos):
    return HybridRIRScene(
        room_dim=[4.0, 5.0, 2.8],
        rt60=0.4,
        mic_pos=mic_pos,
        source_pos=[source_pos],
        source_labels=["near_0"],
    )


def test_cli_defaults_and_backend_selection(tmp_path):
    """Bank emission is opt-in, and each entry point pins its own backend.

    ``generate_hybrid_rir`` stays on pyroomacoustics; the bank wrapper defaults
    to the path-event/FDN high band and material modal damping, whose decay
    shape is closer to measured rooms, and always passes ``--high-backend``
    explicitly.
    """
    args = generate_hybrid_rir._parse_args(["--output-dir", str(tmp_path)])
    assert args.emit_m6_manifest is False
    assert args.high_backend == "pyroomacoustics"

    wrapper = generate_m6_bank._build_parser().parse_args(
        ["--output-dir", str(tmp_path)]
    )
    assert wrapper.backend == "path-events-m4"
    assert wrapper.low_backend == "pytard-material"

    coupled = generate_hybrid_rir._make_high_backend(
        {
            "high_backend": "path-events-m4",
            "pra_max_order": 4,
            "pra_n_rays": 100,
            "fdn_mixing_time_ms": 24.0,
            "fdn_transition_ms": 16.0,
            "fdn_delay_lines": 16,
            "fdn_seed": 33,
        }
    )
    default = generate_hybrid_rir._make_high_backend(
        {"high_backend": "pyroomacoustics", "pra_max_order": 4, "pra_n_rays": 100}
    )
    assert isinstance(coupled, PathEventFDNHighFrequencyBackend)
    assert coupled.mixing_time_s == 0.024
    assert coupled.transition_duration_s == 0.016
    assert isinstance(default, PyroomacousticsHighFrequencyBackend)


def test_bank_wrapper_forwards_valid_distance_shells_and_rejects_impossible_ones(tmp_path):
    """The wrapper can build a boundary-coverage bank.

    The generator's default shells leave a gap between the near and far ranges,
    which is where the near/far decision boundary lies; filling it needs the
    shells reachable from the bank entry point.
    """
    args = generate_m6_bank._build_parser().parse_args(
        ["--output-dir", str(tmp_path), "--far-dist", "1.20", "2.10"]
    )

    assert args.near_dist is None
    assert args.far_dist == [1.20, 2.10]
    # Unset shells are omitted rather than echoed, so a manifest without the
    # flags means the generator's own defaults were used.
    assert generate_m6_bank._distance_flags(args.near_dist, args.far_dist) == [
        "--far-dist",
        "1.2",
        "2.1",
    ]
    assert generate_m6_bank._distance_flags(None, None) == []
    for bad in ([2.0, 1.0], [0.0, 1.0], [-1.0, 1.0], [1.5, 1.5]):
        with pytest.raises(SystemExit, match="needs 0 < MIN < MAX"):
            generate_m6_bank._distance_flags(None, bad)


def test_task_seed_and_acoustic_space_identity_are_stable():
    first = generate_hybrid_rir._m6_task_seed(1337, "room_000001_000002")
    assert first == generate_hybrid_rir._m6_task_seed(1337, "room_000001_000002")
    assert first != generate_hybrid_rir._m6_task_seed(1338, "room_000001_000002")

    # The acoustic space is the room, not where the source and receiver stand.
    assert generate_hybrid_rir._m6_acoustic_space_id(
        _scene([1.0, 1.0, 1.0], [1.5, 1.0, 1.4])
    ) == generate_hybrid_rir._m6_acoustic_space_id(
        _scene([2.0, 2.0, 1.0], [3.0, 2.0, 1.4])
    )


def test_task_contract_binds_code_revision(tmp_path):
    args = generate_hybrid_rir._parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--emit-m6-manifest",
            "--m6-code-revision",
            "revision-a",
        ]
    )
    tasks = [
        {
            "sample_id": "room_000000_000000",
            "room_id": "room_000000",
            "scene": _scene([1.0, 1.0, 1.0], [1.5, 1.0, 1.4]),
        }
    ]

    generate_hybrid_rir._attach_m6_task_contract(
        tasks,
        generate_hybrid_rir.HybridRIRConfig(num_near_sources=1, num_far_sources=0),
        args,
    )

    assert tasks[0]["m6"]["code_revision"] == "revision-a"
    assert "runtime_versions" in generate_hybrid_rir._m6_renderer_config(args)
