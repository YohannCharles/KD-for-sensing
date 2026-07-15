## 1. 基线、split 与筛选配置

- [ ] 1.1 新增 scoped T2 hyperparameter screening launcher，读取本地 T2 resolved config、记录 SHA256，并复用现有 MMW T2 builder 生成 H0 基线。
- [ ] 1.2 从 15-domain outer train 侧复用共享 RSU time-axis 的 `group_safe_time_block` owner 生成固定 10% development validation，写出 ignored split artifacts、inner train/validation 身份不相交审计与 legacy outer-test 资源重叠诊断。
- [ ] 1.3 实现 H0-H7 的 matched-control overlays、canonical allowlist diff、mirrored alias 同步、固定 40-epoch/no-selection/每 5 epoch validation 和 development provenance。
- [ ] 1.4 增加 launcher/split/config tests，覆盖缺少基线、非 15 domains、身份交集、未登记 resolved diff、独立 validation、固定 architecture 与 ignored output 边界。

## 2. 显存探测与并行运行面

- [ ] 2.1 实现 fresh-process 真实 AMP train-step probe，记录每 GPU 的 candidate、设备、版本、退出状态、peak allocated/reserved 与总显存。
- [ ] 2.2 实现预注册 16 倍数候选和 90% 门槛的全卡共同 batch 解析；验证 probe 只写共同 batch，不改变学习率、optimizer/scheduler、梯度累积、epoch、split、loss、AMP、mask 或 checkpoint policy。
- [ ] 2.3 实现 H0-H7 到 GPU0-7 的确定性单进程映射、互不覆盖 run dirs、原子 manifest/status 更新和无共同 batch 时 fail-closed。
- [ ] 2.4 使用 `conda run -n kd_mm_beam` 增加并运行 probe/orchestration tests，覆盖 OOM、非零退出、阈值超限、异卡结果、无共同候选和 probe state 不复用。

## 3. Development 筛选执行

- [ ] 3.1 使用 `conda run -n kd_mm_beam` 完成 15-domain preflight、inner split dry-run、八行 config dry-run 和单 optimizer-step smoke，审计 resolved diffs 与 fingerprints。
- [ ] 3.2 在 GPU0-7 运行预注册 batch probe，冻结共同 batch 与 probe manifest；不得根据 variant 结果改写候选集合或其它 protocol。
- [ ] 3.3 在 GPU0-7 启动 H0-H7 八个 seed-1、40-epoch 任务，并记录每 5 epoch 独立 validation 观测与 run status。
- [ ] 3.4 校验八行均完成 epoch 40、`last.pth` 可加载且 config/split fingerprint 匹配；失败行不得以私有 batch、较早 checkpoint 或不同 split 混入矩阵。

## 4. 固定 checkpoint 评估与证据边界

- [ ] 4.1 使用 `conda run -n kd_mm_beam` 和现有 MMW evaluator，只对 epoch-40 `last.pth` 运行共享 Clean、Drop1/2/3、temporal AUC 与 temporal Drop80 masks。
- [ ] 4.2 在 sample/mask/config/split identity 一致后按预注册公式、0.005 保护门槛和确定性 tie-break 生成 summary；H0 胜出时写 `no_change`。
- [ ] 4.3 将 summary/candidate 标记为 `development_only=true`、`claim_eligible=false`、`screening_consumed_test=true`，确认未更新 reviewed claim、论文主表或 active BPA/CMA formal artifacts。
- [ ] 4.4 使用 `conda run -n kd_mm_beam` 增加并运行 summary tests，覆盖保护门槛、tie-break、缺失/非有限指标、未满 40 epoch、fingerprint 与 paired identity mismatch。

## 5. 回归与收尾

- [ ] 5.1 运行与 launcher、MMW runtime、training selection 和 summary 相关的 `conda run -n kd_mm_beam pytest ... -q` focused tests。
- [ ] 5.2 运行 `make verify-quick`、`make verify-compile`、`openspec validate tune-t2-mmw-hyperparameters --strict` 和 `openspec validate --all --strict`。
- [ ] 5.3 检查 `openspec status --change tune-t2-mmw-hyperparameters` 与 `git status --short`，确认 generated YAML、split、probe、日志、checkpoint、metrics 和图表只留在 ignored `outputs/`，且所有实现任务仍由真实完成状态决定是否勾选。
