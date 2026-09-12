"""Low-frequency wave and modal backends."""

from puresound.audio.rir.render.low_frequency.analytic_modal import (
    AnalyticModalLowFrequencyBackend,
)
from puresound.audio.rir.render.low_frequency.impedance_modal import (
    ImpedanceModalLowFrequencyBackend,
)
from puresound.audio.rir.render.low_frequency.modal_damping import (
    material_modal_damping_metadata,
    material_modal_decay_rates,
)
from puresound.audio.rir.render.low_frequency.pytard import (
    PYTARD_EXCITATION_POLICY,
    GpuARDPytARDBackend,
    GpuARDPytARDCuPyBackend,
    PytARDWaveBackend,
)

__all__ = [
    "PYTARD_EXCITATION_POLICY",
    "AnalyticModalLowFrequencyBackend",
    "GpuARDPytARDBackend",
    "GpuARDPytARDCuPyBackend",
    "ImpedanceModalLowFrequencyBackend",
    "PytARDWaveBackend",
    "material_modal_damping_metadata",
    "material_modal_decay_rates",
]
