#!/usr/bin/env python3
"""Generate and audit the local clean inner-development protocol."""

from __future__ import annotations

import argparse

from kd_sensing.data.mmw.clean_protocol import (
    audit_clean_inner_protocol,
    build_clean_inner_protocol,
    write_clean_inner_protocol,
    write_clean_split_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--protocol-output", required=True)
    parser.add_argument("--audit-json", required=True)
    parser.add_argument("--audit-md", required=True)
    args = parser.parse_args()
    write_clean_inner_protocol(build_clean_inner_protocol(args.source_manifest), args.protocol_output)
    audit = audit_clean_inner_protocol(args.protocol_output, fail_closed=True)
    write_clean_split_audit(audit, args.audit_json, args.audit_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
