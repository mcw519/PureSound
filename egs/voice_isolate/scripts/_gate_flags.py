"""Shared `--presence-gate` / `--onset-guard` plumbing for the eval scripts.

One place so every stage of run_full_benchmark.sh spells the operating
point the same way, and so a stage that forgets the flag is obviously
ungated rather than subtly differently gated.

The two are independent inference-only stages and can be combined: the
presence gain attenuates past the blend's ceiling, the onset guard hands
the dry input back before any talker is confirmed. `SISO.forward` applies
them in that order.
"""
from __future__ import annotations
def add_presence_gate_arg(parser):
    """`--presence-gate <readout.npz>` plus its operating point.

    The readout is fitted offline (see
    egs/voice_isolate/benchmarks/probes/b_traj_README.md) and is not part of the
    checkpoint, so it travels as a file. Omitted, every stage behaves exactly as
    before.
    """
    parser.add_argument("--presence-gate", default=None, metavar="READOUT_NPZ",
                        help="inference-only near-presence gain; path to a fitted readout")
    parser.add_argument("--presence-gate-head", action="store_true",
                        help="drive the same gain from the checkpoint's trained "
                             "vad_head instead of a fitted readout (needs a config "
                             "whose backbone enables vad_head)")
    parser.add_argument("--gate-b-hi", type=float, default=0.50,
                        help="dead-zone edge; at or above it the gain is exactly 1.0")
    parser.add_argument("--gate-b-lo", type=float, default=0.10)
    parser.add_argument("--gate-floor-db", type=float, default=-26.0)
    parser.add_argument("--gate-tau-up", type=float, default=0.05)
    parser.add_argument("--gate-tau-dn", type=float, default=1.0)
    return parser


def build_presence_gate(args):
    """The `PresenceGate` the flags describe, or None.

    Head-driven and readout-driven are mutually exclusive: they are two
    estimators for the same quantity, and accepting both would leave which one
    ran up to argument order.
    """
    from_head = bool(getattr(args, "presence_gate_head", False))
    readout = getattr(args, "presence_gate", None)
    if from_head and readout:
        raise SystemExit("pass --presence-gate OR --presence-gate-head, not both")
    if not (from_head or readout):
        return None
    from puresound.system.presence_gate import PresenceGate
    knobs = dict(b_hi=args.gate_b_hi, b_lo=args.gate_b_lo,
                 gain_floor_db=args.gate_floor_db,
                 tau_up_s=args.gate_tau_up, tau_dn_s=args.gate_tau_dn)
    if from_head:
        return PresenceGate(**knobs)
    return PresenceGate.load(readout, **knobs)


def add_onset_guard_arg(parser):
    """`--onset-guard` plus the four knobs the sweep actually moved.

    Nothing travels as a file here -- the guard reads only the input waveform,
    so the flag is a switch and the operating point is the default measured on
    the FIT set (`benchmarks/probes/onset_guard_sweep.py`,
    `anchor_gate_README` §2). Omitted, every stage behaves exactly as before.
    """
    parser.add_argument("--onset-guard", action="store_true",
                        help="inference-only onset protection: stay dry until a "
                             "talker has been heard for --guard-t-arm seconds")
    parser.add_argument("--guard-t-arm", type=float, default=1.0,
                        help="continuous speech that confirms an anchor (s)")
    parser.add_argument("--guard-t-forget", type=float, default=5.0,
                        help="silence that drops the anchor so the next onset is "
                             "protected again (s); 'inf' disables re-arming")
    parser.add_argument("--guard-tau-dn", type=float, default=2.0,
                        help="release time constant toward the model (s)")
    parser.add_argument("--guard-margin-db", type=float, default=8.0,
                        help="activity threshold above the tracked noise floor (dB)")
    return parser


def build_onset_guard(args):
    """The `OnsetGuard` the flags describe, or None."""
    if not bool(getattr(args, "onset_guard", False)):
        return None
    from puresound.system.onset_guard import OnsetGuard
    return OnsetGuard(t_arm_s=args.guard_t_arm, t_forget_s=args.guard_t_forget,
                      tau_dn_s=args.guard_tau_dn, margin_db=args.guard_margin_db)
