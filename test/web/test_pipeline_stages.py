"""The bilingual stage catalog the Pipeline screen explains each stage with."""

import json
from pathlib import Path

import puresound.web
from puresound.task.trace import STAGES

STATIC = Path(puresound.web.__file__).with_name("static")
REPO = Path(puresound.web.__file__).parents[2]


def test_every_synthesis_stage_is_explained_in_both_languages():
    catalog = json.loads((STATIC / "pipeline-stages.json").read_text(encoding="utf-8"))
    assert list(catalog["stages"]) == [spec.id for spec in STAGES]
    groups = {group["id"]: group for group in catalog["groups"]}
    assert set(groups) == {spec.group for spec in STAGES}
    for group in groups.values():
        assert group["title"]["en"] and group["title"]["zh"] and group["color"].startswith("#")
    for spec in STAGES:
        entry = catalog["stages"][spec.id]
        assert entry["group"] == spec.group, spec.id
        for field in ("title", "what", "model"):
            assert entry[field]["en"].strip() and entry[field]["zh"].strip(), (spec.id, field)
        assert (REPO / entry["doc"].split("#")[0]).is_file(), entry["doc"]
