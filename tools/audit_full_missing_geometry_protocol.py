#!/usr/bin/env python3
"""Run only the hard-fail Full-to-Missing geometry protocol audit."""

from run_full_missing_hard_sample_geometry import main


if __name__ == "__main__":
    raise SystemExit(main(stage_override="audit"))
