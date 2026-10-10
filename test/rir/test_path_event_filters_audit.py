import json

import pytest

from egs.rir_generation.phases.m3_wave_path.scripts.validate_path_event_filters import (
    _load_full_room_audit,
)


def _full_room_report(
    *,
    include_flag: bool,
    causal_max_order: int | None,
    legacy_key: bool = False,
) -> dict:
    realization = {
        "max_order": causal_max_order,
        "time_renderer_vs_analytic_complex_angle": {
            "complex_nrmse": 0.001,
            "complex_correlation": 0.9999,
            "transfer_energy_ratio": 1.0,
        },
    }
    diagnostic = (
        {"causal_path_event_first_order": {"max_order": 1}}
        if legacy_key
        else {"causal_path_event": {"max_order": causal_max_order}}
    )
    return {
        "schema_version": "puresound.full_room_crossover_validation.v1",
        "configuration": {
            "include_causal_path_event_first_order": include_flag,
            "causal_path_event_max_order": causal_max_order,
        },
        "cases": [
            {
                "case": {"case_id": "room"},
                (
                    "causal_path_event_first_order"
                    if legacy_key
                    else "causal_path_event"
                ): realization,
            }
        ],
        "diagnostic": diagnostic,
        "acceptance": {
            "protocol_completed": True,
            "full_room_complex_gate_accepted": False,
        },
    }


@pytest.mark.parametrize(
    ("include_flag", "causal_max_order", "legacy_key", "accepted"),
    [
        # `--include-causal-path-event-first-order`.
        (True, 1, False, True),
        # `--causal-path-event-max-order 1` supersedes the flag but still
        # carries the same first-order diagnostic.
        (False, 1, False, True),
        # A higher-order diagnostic must not be audited against the seven
        # first-order analytic images, even if the flag was also passed.
        (True, 12, False, False),
        (False, 12, False, False),
        # A report written before the diagnostic recorded its order.
        (True, None, True, True),
        (False, None, True, False),
    ],
)
def test_filter_audit_accepts_only_first_order_full_room_reports(
    tmp_path, include_flag, causal_max_order, legacy_key, accepted
):
    path = tmp_path / "full_room.json"
    path.write_text(
        json.dumps(
            _full_room_report(
                include_flag=include_flag,
                causal_max_order=causal_max_order,
                legacy_key=legacy_key,
            )
        ),
        encoding="utf-8",
    )

    if accepted:
        audit = _load_full_room_audit(path)
        assert audit["protocol_completed"] is True
        assert audit["renderer_vs_analytic_aggregate"][
            "maximum_complex_nrmse"
        ] == pytest.approx(0.001)
    else:
        with pytest.raises(ValueError, match="first-order"):
            _load_full_room_audit(path)
