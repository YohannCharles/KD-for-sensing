#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARGS=(--group eval_modular_lite_maskfix)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      ARGS+=(--baseline-root "$2")
      shift 2
      ;;
    --group)
      # Kept for the requested CLI shape; only modular_lite is meaningful here.
      if [[ "$2" != "modular_lite" && "$2" != "eval_modular_lite_maskfix" ]]; then
        echo "[ERROR] maskfix eval supports only --group modular_lite" >&2
        exit 2
      fi
      shift 2
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

exec bash "${SCRIPT_DIR}/run_scene31_subset_reliability.sh" "${ARGS[@]}"
