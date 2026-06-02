## 1. 旧路线依赖审计与基线重定向

- [x] 1.1 梳理 `v2_shared_private`、`shared_private`、`v3_decoupled`、`decoupled` 在源码、配置、脚本、README、tests 和现行 OpenSpec 中的引用，按“必须删除 / 需要改 baseline / 归档保留”分类。
- [x] 1.2 更新 LOSO source checkpoint 选择逻辑，确保 adapter/prototype/target-prior/radio/path/image-only/V7 run 不再自动回退到 `v3_decoupled`。
- [x] 1.3 为旧 variant 配置解析增加拒绝路径或清晰迁移错误，错误信息指向当前合法 baseline。
- [x] 1.4 更新 summary comparison、quick conclusion 和缺失 run 判断，移除 `v3_decoupled` 默认主 baseline。

## 2. 模型与 loss 退役

- [x] 2.1 从 HiST-Beam variant 注册、配置 dataclass 默认值和 resolve 函数中删除旧 `v2/v3` 简单 shared/private 解耦 variant。
- [x] 2.2 删除仅服务旧简单解耦路线的 shared/private scene classifier、orthogonality loss、shared scene confusion loss、private scene preservation loss 和对应 diagnostics。
- [x] 2.3 保留 V7 residual、history residual、path/radio prototype、V8/V9 target prior/prototype 需要的现代 shared/private 或 residual/prototype 字段，并确认其不依赖旧 scene loss。
- [x] 2.4 更新 `HistBeamLossResult`、loss weight 解析和 metrics logging，避免继续暴露旧解耦 loss 字段作为默认公共契约。

## 3. 配置、脚本与文档清理

- [x] 3.1 删除或改写 `configs/hist_beam/` 中以 `v2_shared_private`、`v3_decoupled`、orthogonality、scene_confusion、scene_private 为核心的配置。
- [x] 3.2 更新 LOSO quick/matrix 配置中的 variant 列表，移除旧解耦 baseline，并写入当前合法 baseline。
- [x] 3.3 更新 `scripts/run_*` 中以旧解耦为默认 variant 或 source baseline 的脚本，必要时删除失败路线专用脚本。
- [x] 3.4 更新 README 示例和现行 OpenSpec 引用，停止推荐旧 shared/private 简单解耦路线；归档 OpenSpec 历史只保留失败原因上下文。

## 4. 测试更新

- [x] 4.1 更新 HiST-Beam 模型/loss 单元测试，删除旧 `v3_decoupled` 构建成功断言，新增旧 variant 被拒绝测试。
- [x] 4.2 更新 LOSO planner、source mapping、summary comparison 和 quick conclusion 测试，确认默认矩阵不包含旧解耦 variant。
- [x] 4.3 更新 path/radio/image-only/V7/V8/V9 相关 smoke tests，确认保留路线不依赖 `v3_decoupled` source checkpoint。
- [x] 4.4 运行相关测试：`conda run -n kd_mm_beam pytest tests/test_hist_beam_loso.py tests/test_history_anchored_residual_beam.py -q`。
- [x] 4.5 运行架构边界测试：`conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。

## 5. 本地输出日志和实验结果清理

- [x] 5.1 编写或执行 dry-run 清理命令，扫描 `outputs/`、`logs/` 中旧失败路线产物，生成 `legacy_knowledge_decoupling_cleanup_manifest.json`，记录 path、kind、matched_patterns、size 和保留/删除决策。
- [x] 5.2 审核清理 manifest，确认候选路径不包含 `image_only_legal*`、target-shot geometry residual、V7 residual、V8/V9 target prior/prototype 等活跃实验产物。
- [x] 5.3 按 manifest 删除匹配的旧失败实验日志、progress、metrics、summary、checkpoint/cache 等本地产物，不删除 `dataset/`、`All_models/`、源码、OpenSpec artifacts 或未匹配活跃产物。
- [x] 5.4 删除后重新扫描 `outputs/`、`logs/`，确认旧 `v2/v3` 简单解耦失败产物已清空或只剩 manifest 中标记保留的历史审计项。

## 6. 最终验证

- [x] 6.1 运行 OpenSpec 严格校验：`openspec validate remove-legacy-knowledge-decoupling --strict`。
- [x] 6.2 运行 OpenSpec 状态检查：`openspec status --change remove-legacy-knowledge-decoupling`，确认 artifacts complete 且 ready for apply。
- [x] 6.3 运行保留入口帮助检查：`conda run -n kd_mm_beam kd-sensing-export-viewer-manifest --help` 和 `conda run -n kd_mm_beam kd-sensing-visualize-modalities --help`。
- [x] 6.4 如改动影响公共训练/evaluate CLI，补充运行 `conda run -n kd_mm_beam python scripts/train.py --help` 与 `conda run -n kd_mm_beam python scripts/evaluate.py --help`。
