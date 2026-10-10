import json

import numpy as np
import soundfile as sf

from egs.rir_generation.tools.bank import build_bank_view

SAMPLE_RATE = 16000


def _write_item(root, name, *, rt60, far_tail_std):
    room = root / name
    room.mkdir(parents=True)
    rng = np.random.default_rng(0)
    rir = np.zeros((SAMPLE_RATE, 5), dtype=np.float32)
    for channel in range(5):
        near = channel < 2
        rir[100, channel] = 1.0 if near else 0.2
        std = 0.003 if near else far_tail_std
        rir[400:, channel] = std * rng.standard_normal(SAMPLE_RATE - 400)
    sf.write(room / f"{name}.wav", rir, SAMPLE_RATE, subtype="FLOAT")
    scene = {
        "channel_map": [
            {"channel": index, "label": label}
            for index, label in enumerate(("near_0", "near_1", "far_0", "far_1", "far_2"))
        ]
    }
    if rt60 is not None:
        scene["rt60"] = rt60
    (room / f"{name}.json").write_text(json.dumps({"scene": scene}), encoding="utf-8")


def _levels(bank, output):
    args = build_bank_view.build_parser().parse_args(
        ["levels", str(bank), str(output), "--workers", "1"]
    )
    args.func(args)


def test_levels_skips_items_without_rt60_instead_of_aborting(tmp_path):
    bank = tmp_path / "bank"
    _write_item(bank, "room_a", rt60=0.3, far_tail_std=0.02)
    _write_item(bank, "room_b", rt60=None, far_tail_std=0.02)

    _levels(bank, tmp_path / "views")

    summary = json.loads((tmp_path / "views" / "summary.json").read_text())
    assert summary["valid_items"] == 1
    assert [path.name for path in (tmp_path / "views" / "all" / "items").glob("*.wav")] == [
        "room_a.wav"
    ]


def test_levels_are_cumulative_and_stress_collects_the_rest(tmp_path):
    bank = tmp_path / "bank"
    _write_item(bank, "room_core", rt60=0.3, far_tail_std=0.02)
    _write_item(bank, "room_wide", rt60=0.8, far_tail_std=0.02)
    _write_item(bank, "room_stress", rt60=1.4, far_tail_std=0.02)

    _levels(bank, tmp_path / "views")

    def members(level):
        return sorted(
            path.stem for path in (tmp_path / "views" / level / "items").glob("*.wav")
        )

    assert members("core") == ["room_core"]
    assert members("expand") == ["room_core"]
    assert members("wide") == ["room_core", "room_wide"]
    assert members("stress") == ["room_stress"]
    assert members("all") == ["room_core", "room_stress", "room_wide"]
    assert all(
        path.resolve().is_file()
        for path in (tmp_path / "views" / "core" / "items").glob("*")
    )


def test_merge_prefixes_each_source_so_equal_stems_do_not_collide(tmp_path):
    bank = tmp_path / "bank"
    _write_item(bank, "room_a", rt60=0.3, far_tail_std=0.02)
    _levels(bank, tmp_path / "views_one")
    _levels(bank, tmp_path / "views_two")
    args = build_bank_view.build_parser().parse_args(
        [
            "merge",
            "--source",
            f"one={tmp_path / 'views_one' / 'all'}",
            "--source",
            f"two={tmp_path / 'views_two' / 'all'}",
            "--output",
            str(tmp_path / "merged"),
        ]
    )
    args.func(args)

    names = sorted(path.name for path in (tmp_path / "merged" / "items").glob("*"))
    assert names == ["one_room_a.json", "one_room_a.wav", "two_room_a.json", "two_room_a.wav"]
    assert (tmp_path / "merged" / "items" / "two_room_a.wav").resolve() == (
        bank / "room_a" / "room_a.wav"
    ).resolve()
