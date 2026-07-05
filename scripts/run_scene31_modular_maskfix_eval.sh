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
    --baseline-root|--gpus|--gpu|--max-parallel)
      ARGS+=("$1" "$2")
      shift 2
      ;;
    --overwrite-eval|-h|--help)
      ARGS+=("$1")
      shift
      ;;
    --group)
      if [[ "$2" != "modular_lite" && "$2" != "eval_modular_lite_maskfix" ]]; then
        echo "[ERROR] modular maskfix eval supports only --group modular_lite" >&2
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
