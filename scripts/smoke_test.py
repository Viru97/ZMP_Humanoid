"""Headless smoke test for ZMP humanoid phases.

Runs shortened phase durations without launching the interactive viewer.
Returns non-zero if any phase fails.
"""

import argparse
import sys

from main import load_g1_model
from phase_manager import VisualizedPhaseManager
from config import cfg


class DummyViewer:
    """Minimal viewer stub for headless testing."""

    def is_running(self):
        return True

    def sync(self):
        return None


def run_smoke_test(
    settle: float,
    balance: float,
    hold: float,
    sway: float,
    amplitude: float,
    frequency: float,
    phase4_mode: str,
) -> bool:
    # Speed up tests by disabling real-time pacing.
    cfg.PHASE.SETTLE_PACE = 0.0
    cfg.PHASE.BALANCE_PACE = 0.0
    cfg.PHASE.HOLD_PACE = 0.0
    cfg.PHASE.STATUS_INTERVAL = 0.5
    cfg.PHASE.PHASE4_MODE = phase4_mode

    model, data = load_g1_model()
    manager = VisualizedPhaseManager(model, data, DummyViewer())

    print("\\n[SMOKE] Running phases...")
    ok1 = manager.phase_settle(duration=settle)
    ok2 = ok1 and manager.phase_balance(duration=balance)
    ok3 = ok2 and manager.phase_stability_hold(duration=hold)
    ok4 = ok3 and manager.phase_zmp_sway(
        duration=sway,
        amplitude=amplitude,
        frequency=frequency,
    )

    print(f"[SMOKE] Results: settle={ok1}, balance={ok2}, hold={ok3}, phase4={ok4}")
    return bool(ok1 and ok2 and ok3 and ok4)


def main() -> int:
    parser = argparse.ArgumentParser(description="Headless smoke test for ZMP humanoid phases")
    parser.add_argument("--settle", type=float, default=1.0)
    parser.add_argument("--balance", type=float, default=2.0)
    parser.add_argument("--hold", type=float, default=1.0)
    parser.add_argument("--sway", type=float, default=8.0)
    parser.add_argument("--amplitude", type=float, default=0.02)
    parser.add_argument("--frequency", type=float, default=0.2)
    parser.add_argument("--phase4-mode", type=str, default="safe_sway", choices=["safe_sway", "preview_ik"])
    args = parser.parse_args()

    ok = run_smoke_test(
        settle=args.settle,
        balance=args.balance,
        hold=args.hold,
        sway=args.sway,
        amplitude=args.amplitude,
        frequency=args.frequency,
        phase4_mode=args.phase4_mode,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
