import json

import pytest

from puresound.audio.rir_measured_calibration import (
    CampaignNotReadyError,
    deterministic_position_assignments,
    run_measured_campaign_fit,
)
from puresound.audio.rir_measurement_campaign import RIRMeasurementCampaign


def _template_campaign():
    with open(
        "egs/rir_generation/phases/m5_calibration/config/m5_measurement_campaign_template.json",
        encoding="utf-8",
    ) as handle:
        return RIRMeasurementCampaign.from_json(handle.read())


def test_measured_runner_fails_before_fitting_incomplete_campaign(tmp_path):
    campaign = _template_campaign()

    with pytest.raises(CampaignNotReadyError) as caught:
        run_measured_campaign_fit(campaign, tmp_path)

    assert caught.value.audit["ready_for_m5_inverse_calibration"] is False
    assert caught.value.audit["missing_asset_paths"]
    json.dumps(caught.value.audit, allow_nan=False)


def test_position_split_rejects_train_room_without_holdout_position():
    with pytest.raises(ValueError, match="at least two positions"):
        deterministic_position_assignments(_template_campaign())
