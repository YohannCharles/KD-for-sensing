## 1. 证据与汇报口径

- [x] 1.1 逐页解析 `进度汇报.pdf` 的文字、字体颜色和页面渲染，定位第 4、5、7、9、14、15 页全部红字并记录上下文。
- [x] 1.2 核对 T2/S1 seed1 固定 mask 结果、final C2 prototype 消融和当前 circular 实现，并只读审计 18,479 个 DeepSense Scene31-34 有限 beam-power sweep 的端点/相邻关系。
- [x] 1.3 新增中文 Markdown，写入逐页替换文案、文献建议、叙事/版式修改、当前证据边界、详细实验矩阵和停止条件。

## 2. 最小实验入口

- [x] 2.1 扩展现有 H5/P1 launcher 的显式方法表，增加 `S1-LG`、`T2-LG`、`S1-CLS`、`T2-CLS`，保持两个 profile 默认方法列表不变。
- [x] 2.2 为 eval script 增加兼容默认的 `config|circular|linear` distance override，并在 pattern/mask/training diagnostics 中记录 geometry、head 与 metric provenance。
- [x] 2.3 增加 focused tests，覆盖 linear-Gaussian 不退化 one-hot、classifier 关闭 prototype 依赖、T2 KL 保留、默认 profile 不变、router oracle 的 linear/circular 端点行为和 eval distance override。
- [x] 2.4 为 eval rows 增加稳定 mask identity/cache provenance，并让 summary 输出逐 seed、严格配对后 digest 去重的 delta 与 candidate/final gate decision；缺数据或 provenance 冲突必须 unavailable。

## 3. 验证与 Dry-run

- [x] 3.1 运行 `conda run -n kd_mm_beam pytest tests/test_h5_p1_temporal_matrix_v1.py tests/test_s1_temporal_superset_training.py -q`。
- [x] 3.2 运行 `openspec validate validate-t2-beam-geometry-and-head --strict`、`openspec validate --all --strict` 和 `make verify-quick`。
- [x] 3.3 用两个独立 ignored root dry-run，确认 GPU0-3 为 current S1/T2 seeds2/3、GPU4-7 为四候选 seed1，且 config、log、output 与 manifest 互不覆盖。

## 4. 第一轮 GPU0-7 筛选

- [x] 4.1 完成 LG/CLS 四候选 seed1 训练；current S1/T2 seeds2/3 后续按用户的 MMW 资源切换决策在 epoch 5/6 停止，并保留独立 root、stale `running` status 和取消边界，禁止当作完成 checkpoint。
- [x] 4.2 保留四候选 seed1 missing-pattern 评估；由于 current seeds2/3 未完成，不继续旧 first-round evaluator，也不为不完整 checkpoint 复算或聚合正式指标。
- [x] 4.3 将候选晋级门禁记录为 `unavailable/cancelled`：缺少完整 current 对照和五档固定 temporal matrix，不生成 pass，不消费后续多 seed 算力。

## 5. 条件式多 Seed 收口

- [x] 5.1 第一轮门禁不可用且用户已停止该 DeepSense 分支，因此不启动 LG/CLS seeds2/3，并记录 skipped reason 为 `cancelled_by_later_resource_switch`。
- [x] 5.2 没有晋级 checkpoint，固定 mask advancement evaluation 按门禁契约跳过；未跨 circular/linear distance mode 聚合距离指标。
- [x] 5.3 三 seed mean/std、逐 seed delta、paired delta 和主线 pass/fail 记为 `unavailable/cancelled`，中文 Markdown 回填实际执行边界；不更新正式 claim。

## 6. 治理收口

- [x] 6.1 收口旧 change：补记已完成的 6.1-6.3 与 7.1；记录 J1 未通过、J2 ineligible、T1/T1+T2 不继续，并将旧 7.2-7.3 按后继 change 的实际取消/不可用结果补记完成。
- [x] 6.2 复核 `git status --short`，确认没有纳入 PDF、dataset、outputs、logs、cache 或 checkpoint，并再次运行目标 OpenSpec strict validation。

## 收口结果

- `candidate_screen_clean` 中 S1-LG、T2-LG、S1-CLS、T2-CLS seed1 均完成；旧 `candidate_screen` 已作废。
- `current_multiseed` 四个 run 只到 epoch 5/6，`run_status.json` 仍为 `running`，且进程已经不存在；这些产物明确视为 stopped/ineligible，不修写为 complete。
- 后续 MMW change 已记录用户停止 DeepSense seeds2/3 与旧评估编排器的资源切换决定，因此本 change 以 cancelled experimental branch 收口，不形成三 seed T2、geometry 或 head claim。
- 收口前已将 `temporal-window-missing` delta 重基为 current H5/P1 requirement 与 LG/CLS 场景并集，并显式继承 group-safe split、跨 split identity audit 和 final test evidence 契约，避免归档覆盖后续数据完整性约束。
