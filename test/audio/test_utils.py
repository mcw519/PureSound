"""``puresound.utils``: folder listing and configuration loading."""

import pytest
import yaml

from puresound.utils import iter_files_recursive, load_hparam, recursive_read_folder


def test_recursive_listing_pairs_each_name_with_its_full_path(tmp_path):
    (tmp_path / "a dir").mkdir()
    (tmp_path / "a dir" / "x y.wav").write_bytes(b"")
    (tmp_path / "z.wav").write_bytes(b"")
    (tmp_path / "skip.txt").write_bytes(b"")

    pairs = sorted(iter_files_recursive(str(tmp_path), ".wav"))
    lines = []
    recursive_read_folder(str(tmp_path), ".wav", lines)

    assert pairs == [
        ("x y.wav", str(tmp_path / "a dir" / "x y.wav")),
        ("z.wav", str(tmp_path / "z.wav")),
    ]
    assert sorted(lines) == sorted(f"{name} {path}" for name, path in pairs)


def test_load_hparam_merges_documents_in_order_and_skips_an_empty_one(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("a: 1\nb: [1, 2]\n---\nb: 3\n---\n", encoding="utf-8")

    assert load_hparam(str(path)) == {"a": 1, "b": 3}


@pytest.mark.parametrize(
    "value",
    ["!!python/name:os.getcwd", "!!python/tuple [1, 2]", "!!python/object/apply:os.getcwd []"],
)
def test_load_hparam_refuses_python_specific_tags(tmp_path, value):
    path = tmp_path / "config.yaml"
    path.write_text(f"x: {value}\n", encoding="utf-8")

    with pytest.raises(yaml.YAMLError):
        load_hparam(str(path))
