"""Scene31 local summary owner."""

import argparse

from kd_sensing.diagnostics.scene31_summary import (
    bc_next,
    subset_reliability,
    baseline_pack,
    funnel,
    next_round,
    p0_fresh_eval,
    patternfilm_d8,
    subset_reference,
)


PROFILES = {
    "baseline-pack": baseline_pack,
    "bc-next": bc_next,
    "funnel": funnel,
    "next-round": next_round,
    "p0-fresh-eval": p0_fresh_eval,
    "patternfilm-d8": patternfilm_d8,
    "subset-reference": subset_reference,
    "subset-reliability": subset_reliability,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Scene31 summary profile.")
    parser.add_argument("profile_name", nargs="?", choices=sorted(PROFILES))
    parser.add_argument("--profile", choices=sorted(PROFILES))
    args, rest = parser.parse_known_args(argv)
    profile = args.profile or args.profile_name or "bc-next"
    return int(PROFILES[profile].main(rest) or 0)


__all__ = ["PROFILES", "main"]
