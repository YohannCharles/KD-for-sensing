## 1. 配置矩阵

- [x] 1.1 新增 `scripts/generate_scene31_magic_overnight.py`，生成 17 个 magic overnight run 配置与 manifest。
- [x] 1.2 确认核心 11 个 run、全量 17 个 run 的 seed、epoch、group、method_tags 和默认 output root 正确。

## 2. 训练扩展

- [x] 2.1 在 U-MaskBeamJEPA training extension 中实现 opt-in `training.mpdro` EMA group loss、softmax group weight 和 sample weight 应用。
- [x] 2.2 输出 `mpdro_group_log.csv` 并在 epoch 结束打印 `[MPDRO]` 权重摘要。

## 3. Runner 与汇总

- [x] 3.1 新增 `scripts/run_scene31_magic_overnight.sh`，支持 group、GPU worker 队列、断点续跑、失败不中断、逐 run 日志和 failed list。
- [x] 3.2 runner 复用 `scripts/reevaluate_apples_to_apples.py` 做 fresh eval，并在结束后调用 summary。

## 4. 验证与启动

- [x] 4.1 运行 `openspec validate add-scene31-magic-overnight-workflow --strict`。
- [x] 4.2 运行 `conda run -n kd_mm_beam pytest tests/test_scene31_next_round.py tests/test_u_mask_beam_jepa.py -q` 或更窄 focused checks。
- [x] 4.3 用 GPU 4-7 后台启动 `overnight_all --auto-eval`，并确认 runner 进程和每 GPU worker 已开始。
