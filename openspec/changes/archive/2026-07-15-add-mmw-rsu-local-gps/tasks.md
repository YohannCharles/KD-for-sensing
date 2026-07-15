## 1. GPS 数据契约与实现

- [x] 1.1 在 GPS contract 中注册 `rsu_local_relative_polar`，保持默认 `relative_polar`、三维 shape 和旧模式数值不变。
- [x] 1.2 实现 MMW BS YAML `rsu_pose.rotation.yaw` 的逐帧读取、有限值/静态窗口一致性校验与帧级 cache，并据此构造 RSU 局部相对极坐标特征。
- [x] 1.3 将 GPS feature mode、angle frame、yaw source 和校验状态接入 dataset/runtime/scaler provenance，拒绝不匹配的持久化 artifact。

## 2. Focused 验证

- [x] 2.1 增加 synthetic focused tests，覆盖已知旋转、旧模式回归、非 YAML、缺失/非法 yaw、窗口 yaw 不一致和 cache 复用。
- [x] 2.2 使用 `conda run -n kd_mm_beam` 在 Town03 五场景真实 YAML 上运行 world/local 特征级 paired 诊断，记录 Exact、±1、±3 和平均 circular beam error。
- [x] 2.3 运行 `openspec validate add-mmw-rsu-local-gps --strict`、相关 GPS/MMW pytest、config/architecture 检查与 compile verification。

## 3. Matched 实验与结论

- [x] 3.1 生成 ignored world/local GPS-only 与 T2 seed1 配置，确认 15-domain inventory、H5/P1 split、seq_len=5、scaler、sample cache、训练预算和输出目录严格隔离，并完成 one-batch smoke。
- [x] 3.2 在空闲 GPU 上并行完成 world/local GPS-only 快速对照和 T2 40-epoch seed1 matched 训练，统一冻结 `last.pth`，不得干扰已有 GPU 任务。
- [x] 3.3 使用共享 v2 missing mask cache 评估 world/local T2 的 clean、whole-modality 与 temporal missing，输出 paired delta、天气/场景/domain macro、worst-domain 和 GPS branch 指标。
- [x] 3.4 根据 matched 结果判断 RSU 局部化是否修复 GPS 负贡献，并明确与 router pattern bias、seed1 local-validation 和多 seed claim 的边界。

## 收口结果

- world/local GPS-only 与 T2 seed1 均完成 40 epoch，正式 T2 评估各有 1,860 行、15 个 domain，并使用 fixed-epoch `last.pth` 与共享 v2 mask identity。
- RSU local 相对 world 的 full domain-macro Top1 提升 `+0.01065`，GPS-only Top1 提升 `+0.07432`，但 full Top3、ADBA 和 MAE 未同步改善，且 GPS gate 均值仍约为 `0.00168`。
- 结论仅为 seed1 local validation：局部坐标修复明显改善 GPS 表征和 Top1，但没有隔离 router pattern bias，也不能替代多 seed claim。
