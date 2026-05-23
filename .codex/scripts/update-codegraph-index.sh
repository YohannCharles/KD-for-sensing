#!/usr/bin/env bash
set -u

project_root="${1:-.}"

if ! command -v codegraph >/dev/null 2>&1; then
  exit 0
fi

(
  cd "$project_root" 2>/dev/null || exit 0

  if [ -d ".codegraph" ]; then
    codegraph sync . --quiet || codegraph index . --quiet || true
  else
    codegraph init -i . >/dev/null 2>&1 || true
  fi
) >/dev/null 2>&1

exit 0
