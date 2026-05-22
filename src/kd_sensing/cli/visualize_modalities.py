from __future__ import annotations

from kd_sensing.cli.export_viewer_manifest import main as export_viewer_manifest_main


def main(argv: list[str] | None = None) -> dict:
    return export_viewer_manifest_main(argv)


if __name__ == "__main__":
    main()
