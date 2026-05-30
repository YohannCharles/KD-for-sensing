## 1. KD 表面积盘点与分类

- [x] 1.1 盘点 `src/kd_sensing/distillation/`、`src/kd_sensing/engine/`、`src/kd_sensing/config/`、`configs/**/{logits_kd,rkd}.yaml`、README 和 tests 中的 KD/teacher/student/runtime 引用
- [x] 1.2 将每个 KD 入口分类为 active mainline 依赖、legacy KD baseline、纯算法 helper、历史兼容模型名或文档/测试引用
- [x] 1.3 记录不应物理删除的历史复现对象，包括已跟踪配置、模型注册名、历史 checkpoint 兼容和现有 KD baseline 测试
- [x] 1.4 明确第一阶段迁移清单：默认主线去依赖、legacy baseline 保留、后续可删除或重命名项

## 2. Legacy KD metadata 与生命周期

- [x] 2.1 新增或扩展 run metadata helper，写出 `distillation_enabled`、`method_family`、`teacher_checkpoint`、`teacher_source`、`distillation_type` 和 `main_conclusion_eligible`
- [x] 2.2 为显式 `logits_kd`、`rkd` 或等价 KD baseline 配置写出 `method_family=legacy_kd` 或等价字段
- [x] 2.3 保证 no-KD mainline、HiST-Beam、history-anchored residual、adapter/prototype/calibration run 写出 `distillation_enabled=false`
- [x] 2.4 为 KD 配置、script 或 virtual recipe 增加 lifecycle 分类来源，至少覆盖 legacy KD、optional baseline、historical reproduction 三类
- [x] 2.5 增加配置/metadata 单元测试，验证 legacy KD 与 no-KD mainline 的字段可区分

## 3. 训练运行时去默认 KD 依赖

- [x] 3.1 调整训练对象构建路径，使当前推荐 no-KD 配置不要求 `teacher_model_name`、KD temperature、alpha 或 RKD 权重字段
- [x] 3.2 将 teacher model 构建、teacher checkpoint 加载和 teacher/student forward 限定到显式 legacy KD baseline 或 optional KD extension
- [x] 3.3 保证 no-KD supervised/adaptation batch step 不调用 KD loss，也不生成新的 `loss/distillation` 字段
- [x] 3.4 保留 `kd_sensing.distillation` 中纯张量级 loss/schedule helper 的轻量导入边界，避免导入 dataset/model/checkpoint runtime
- [x] 3.5 新增或更新架构边界测试，拒绝 active mainline 代码重新依赖 legacy KD runtime 聚合入口

## 4. HiST-Beam 与跨场景主线边界

- [x] 4.1 调整 HiST-Beam LOSO source training、source-only eval、target adaptation 和 adapted eval metadata，默认标记 no-KD mainline
- [x] 4.2 确认 sensor-assisted quick validation、history-anchored residual quick validation 和默认 HiST-Beam plan 不自动生成 KD baseline variant
- [x] 4.3 若保留 HiST-Beam KD baseline，新增显式 profile 或配置标记，并将其 summary 分组为 supplemental/legacy
- [x] 4.4 确保 shared/private、prototype、path/radio prototype、residual calibration diagnostics 在无 teacher 输出时仍完整可用
- [x] 4.5 增加 focused tests，覆盖默认 HiST-Beam/adaptation 配置不构建 frozen teacher、不读取 teacher checkpoint

## 5. Soft beam label 命名与 loss 字段

- [x] 5.1 盘点代码和配置中将 beam-aware soft label 误命名为 KD 的字段、日志和文档
- [x] 5.2 将新写出的 no-KD soft target loss 字段命名为 `loss/beam_soft_target`、`loss/beam_smoothing` 或等价非 KD 名称
- [x] 5.3 为历史 `kd_soft_label` 或等价旧字段提供兼容读取路径，并在新 artifact 中写出 beam soft target 命名
- [x] 5.4 在 legacy KD baseline 同时启用 beam soft target 时，分离 supervised beam soft-target loss 与 teacher-student distillation loss
- [x] 5.5 增加 soft label 测试，验证 validation/evaluation 仍使用 hard `target_beam` 计算 Top-K、DBA、NRP 和 beam power 指标

## 6. Summary、eligibility 与实验入口

- [x] 6.1 扩展训练、评估、LOSO summary 和 quick validation conclusion，使 artifact 包含 method family 与 distillation enabled 状态
- [x] 6.2 默认 mainline ranking 和胜负判断排除 `method_family=legacy_kd` 或 `distillation_enabled=true` 的 run
- [x] 6.3 支持用户显式请求 KD comparison 时，将 KD 指标输出为 supplemental comparison，而不是 mainline conclusion
- [x] 6.4 增加 summary/eligibility tests，覆盖同一 fold/budget/seed 下 no-KD mainline 与 KD baseline 并存的分组行为
- [x] 6.5 检查默认 quickstart、推荐 CLI 和 quick validation 配置，确保首推路径不依赖 KD checkpoint

## 7. 文档与项目元数据

- [x] 7.1 更新 `pyproject.toml` description，使项目描述不再以 KD-first 工作流为唯一或首要定位
- [x] 7.2 更新 README 和相关 docs，将当前主线表述为少样本跨场景/多模态 beam prediction、HiST-Beam 或 history-anchored adaptation
- [x] 7.3 在文档中说明 KD 已从 active mainline 隔离，保留为 legacy/optional baseline，并给出历史 branch/tag 或配置保留策略
- [x] 7.4 确认文档不要求提交 `dataset/`、`outputs/`、`logs/`、cache 或新 checkpoint 等本地产物
- [x] 7.5 确认 README 推荐入口仍使用 `conda run -n kd_mm_beam` 环境示例

## 8. 验证与 OpenSpec 校验

- [x] 8.1 使用 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q` 验证架构边界
- [x] 8.2 使用 `conda run -n kd_mm_beam pytest <focused config/metadata tests> -q` 验证 no-KD 与 legacy KD 配置分类
- [x] 8.3 使用 `conda run -n kd_mm_beam pytest <focused soft-label tests> -q` 验证 beam soft target 命名和 hard-label evaluation
- [x] 8.4 使用 `conda run -n kd_mm_beam pytest <focused hist-beam/summary tests> -q` 验证 HiST-Beam 默认 no-KD、summary 分组和 eligibility
- [x] 8.5 使用 `conda run -n kd_mm_beam kd-sensing-train --help`、`conda run -n kd_mm_beam kd-sensing-evaluate --help` 和 `conda run -n kd_mm_beam kd-sensing-hist-beam-loso --help` 做入口 smoke
- [x] 8.6 使用 `openspec validate isolate-legacy-kd-mainline --strict` 校验 change
- [x] 8.7 实现完成后按风险决定是否运行 `conda run -n kd_mm_beam pytest -q` 作为最终回归
