#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_mmw_sunny_modal15_l5p3_h123.sh

默认后台运行 15 个 sunny MMW 实验，同一时刻最多 4 个进程。默认 `GPU_IDS=0,1,2,3`，适合 4 张 3090 每卡 1 个任务。

任务定义：
  SEQ_LEN=5      使用历史 5 帧
  NUM_PRED=3     预测未来 3 帧
  SPLIT_TAG=l5p3_group_safe
                 默认使用 group-safe strict split
  METRIC_HORIZONS=[1,2,3]
                 val_acc/val_atop3/val_atop5/val_adba 和 val_top*_avg 汇总第 1/2/3 帧

输出目录：
  outputs/mmw_sunny_modal15/l5p3_group_safe_h123/<scene>/sunny_MMW_<scene>_l5p3_group_safe_h123_<kind>_supervised

常用环境变量：
  GPU_IDS=0,1,2,3
  EPOCHS=100
  TRAIN_BATCH_SIZE=64
  TEST_BATCH_SIZE=128
  NUM_WORKERS=8
  PREFETCH_FACTOR=2
  PREPARE_SPLITS=1
  PREPARE_RADAR_MAPS=1
  PREWARM_CACHE=0
  RUN_IN_BACKGROUND=0
  POST_MODAL15_CMD='bash scripts/<current-follow-up>.sh'
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

export LOG_ROOT="${LOG_ROOT:-logs/mmw_sunny_modal15_l5p3_h123_train}"
export SEQ_LEN="${SEQ_LEN:-5}"
export NUM_PRED="${NUM_PRED:-3}"
export SPLIT_TAG="${SPLIT_TAG:-l5p3_group_safe}"
export SPLIT_STRATEGY="${SPLIT_STRATEGY:-group_safe_time_block}"
export HORIZON_TAG="${HORIZON_TAG:-l5p3_group_safe_h123}"
export METRIC_HORIZONS="${METRIC_HORIZONS:-[1,2,3]}"
export SCHEDULER_LABEL="${SCHEDULER_LABEL:-l5p3/h123}"

if [[ -n "${POST_MODAL15_CMD:-}" ]]; then
  RUN_IN_BACKGROUND=0 bash "$SCRIPT_DIR/run_mmw_sunny_modal15_l5p6_h246.sh" "$@"
else
  bash "$SCRIPT_DIR/run_mmw_sunny_modal15_l5p6_h246.sh" "$@"
fi
status=$?

if (( status != 0 )); then
  exit "$status"
fi

if [[ -n "${POST_MODAL15_CMD:-}" ]]; then
  printf '[%s] POST_MODAL15_CMD start: %s\n' "$(date '+%F %T')" "$POST_MODAL15_CMD"
  bash -lc "$POST_MODAL15_CMD"
fi
