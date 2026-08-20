"""Shared `--presence-gate` plumbing for the eval scripts.

One place so every stage of run_full_benchmark.sh spells the operating
point the same way, and so a stage that forgets the flag is obviously
ungated rather than subtly differently gated.
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
