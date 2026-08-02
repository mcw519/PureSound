from egs.rir_generation import generate_hybrid_rir
from egs.rir_generation import generate_m6_bank
from puresound.audio.hybrid_rir import HybridRIRScene


def test_m6_emission_is_opt_in_and_default_backend_is_unchanged(tmp_path):
    args = generate_hybrid_rir._parse_args(["--output-dir", str(tmp_path)])

    assert args.emit_m6_manifest is False
    assert args.high_backend == "pyroomacoustics"


def test_m6_wrapper_defaults_to_material_modal_damping(tmp_path):
    args = generate_m6_bank._build_parser().parse_args(
        ["--output-dir", str(tmp_path)]
    )

    assert args.low_backend == "pytard-material"


def test_m6_task_seed_is_stable_and_base_seed_sensitive():
    first = generate_hybrid_rir._m6_task_seed(1337, "room_000001_000002")

    assert first == generate_hybrid_rir._m6_task_seed(
        1337, "room_000001_000002"
    )
    assert first != generate_hybrid_rir._m6_task_seed(
        1338, "room_000001_000002"
    )


def test_m6_acoustic_space_identity_excludes_source_receiver_positions():
    first = HybridRIRScene(
        room_dim=[4.0, 5.0, 2.8],
        rt60=0.4,
        mic_pos=[1.0, 1.0, 1.0],
        source_pos=[[1.5, 1.0, 1.4]],
        source_labels=["near_0"],
    )
    second = HybridRIRScene(
        room_dim=[4.0, 5.0, 2.8],
        rt60=0.4,
        mic_pos=[2.0, 2.0, 1.0],
        source_pos=[[3.0, 2.0, 1.4]],
        source_labels=["near_0"],
    )

    assert generate_hybrid_rir._m6_acoustic_space_id(
        first
    ) == generate_hybrid_rir._m6_acoustic_space_id(second)


def test_m6_task_contract_binds_code_revision(tmp_path):
    args = generate_hybrid_rir._parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--emit-m6-manifest",
            "--m6-code-revision",
            "revision-a",
        ]
    )
    scene = HybridRIRScene(
        room_dim=[4.0, 5.0, 2.8],
        rt60=0.4,
        mic_pos=[1.0, 1.0, 1.0],
        source_pos=[[1.5, 1.0, 1.4]],
        source_labels=["near_0"],
    )
    tasks = [
        {
            "sample_id": "room_000000_000000",
            "room_id": "room_000000",
            "scene": scene,
        }
    ]

    generate_hybrid_rir._attach_m6_task_contract(
        tasks,
        generate_hybrid_rir.HybridRIRConfig(num_near_sources=1, num_far_sources=0),
        args,
    )

    assert tasks[0]["m6"]["code_revision"] == "revision-a"
    assert "runtime_versions" in generate_hybrid_rir._m6_renderer_config(args)
