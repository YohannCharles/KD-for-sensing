## 1. 审计与数据契约

- [x] 1.1 审计 MMW 四传感器 shape、单位、T/M/N、split、metadata、复制帧与跨天气匹配，并写入 ignored `outputs/pgcd_quick_search/implementation_notes.md`
- [x] 1.2 实现 PGCD 配置/batch 信道泄漏 fail-fast 检查，证明训练不加载 channel、CSI、path 或 beam-power tensor
- [x] 1.3 实现确定性 `SensorDegradationGenerator`、L0-L4、训练采样、stale frame 和 source-frame 分组

## 2. 模型与损失

- [x] 2.1 实现 opt-in PGCD block quality Router、learned prior、非负 beta、masked reliability fusion 和 cached reroute
- [x] 2.2 实现 topology transport/debiased drift、task degradation、batch-only normalization、regression、ranking 和 consistency loss
- [x] 2.3 将 clean/corrupted 双 view 与 C0-C7 固定模式接入 U-Mask training extension，并记录 loss/gradient 诊断

## 3. 配置与评测

- [x] 3.1 实现 C0-C7 resolved config/manifest/PID/status 生成器和 `scripts/run_pgcd_quick_search_gpu4_5.sh`
- [x] 3.2 实现 E0-E5、weather/severity/sensor/unseen/original-missing 指标与 D0-D3 dynamic replacement 评测
- [x] 3.3 实现质量相关性、单调性、robustness AUC、梯度一致性、计算开销、主表和 quick-gate 汇总
- [x] 3.4 按用户运行时指示恢复 batch 32，并将八任务调度为 GPU4/5 两条串行队列

## 4. 验证与本地快筛

- [x] 4.1 使用 `conda run -n kd_mm_beam pytest` 运行 corruption、quality、leakage 与 launcher 定向测试，并保存 `preflight_tests.txt`
- [x] 4.2 使用 `openspec validate add-pgcd-continuous-degradation --strict`、`make verify-quick` 和 compile 检查验证项目边界
- [ ] 4.3（运行中）在确认 GPU 占用后显式启动 C0-C7 seed1 inner 快筛，评测完成任务并如实记录失败；不得启动 multi-seed、outer test 或下一轮实验
