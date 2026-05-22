## Why

项目已经完成多轮功能扩展和架构拆分，但配置矩阵、历史 OpenSpec 文档、重复脚本入口和实验诊断工具仍在持续扩大维护表面积。现在需要把可生成、可归档或已迁移到包内入口的内容收敛掉，降低后续训练配置、数据集和诊断能力演进时的认知负担。

## What Changes

- 将可由 canonical/overlay recipe 稳定生成的 fusion 实体 YAML 收敛为 virtual config 入口，只保留真正需要人工维护的 seed/base/example 配置。
- 将 README 从长篇实验说明收缩为安装、快速健康检查、核心工作流和指向性文档；把细节迁移到更合适的 docs 或 OpenSpec。
- 清理已经迁移到包内 CLI 的重复脚本入口和文档推荐路径，保留明确需要的开发/研究脚本。
- 为配置矩阵、重复入口、OpenSpec/README 规模和本地产物泄漏增加轻量检查，防止后续重新膨胀。
- 将历史沉积的 OpenSpec specs 做语义整理：补齐 TBD purpose，归档或合并只记录历史迁移过程、不再定义当前行为的要求。
- **BREAKING**: 直接调用被删除 fallback 脚本路径的本地命令需要改用对应 console script 或 `python -m kd_sensing.cli.<name>`；训练、评估、预处理和已声明的 console script 行为不变。

## Capabilities

### New Capabilities

- 无。该变更收敛现有项目表面积，不引入新的运行时算法或用户工作流能力。

### Modified Capabilities

- `project-architecture`: 增加项目表面积预算、重复入口删除和源码/文档/产物膨胀回归检查要求。
- `canonical-config-resolution`: 扩展 virtual/overlay 配置作为高级 fusion 配置矩阵的首选表达，并约束实体 YAML 的保留条件。
- `experiment-workflow`: 收缩推荐实验文档和配置矩阵说明，要求训练/评估工作流继续接受 virtual config 且 final artifact 保存完整解析配置。
- `configurable-multimodal-fusion`: 调整高级 fusion 实体 YAML 兼容要求，允许等价 overlay 稳定后删除冗余实体 YAML。

## Impact

- 受影响代码和文件：`configs/fusion/`、`configs/csi/hardening_matrix/` 中可生成配置，`src/kd_sensing/config/canonical*.py`，`src/kd_sensing/cli/`，`scripts/`，`tools/`，`README.md`，`docs/`，`openspec/specs/`，`tests/test_architecture_boundaries.py` 和配置加载相关测试。
- 用户可见入口：推荐入口统一到 `kd-sensing-train`、`kd-sensing-evaluate`、`kd-sensing-preprocess`、`kd-sensing-export-viewer-manifest`、`kd-sensing-visualize-modalities` 或对应 `python -m kd_sensing.cli.*`。
- 运行行为：训练、评估、预处理、checkpoint registry、final/resolved config、existing console scripts 和快速健康检查应保持兼容。
- 验证要求：相关测试必须通过 `conda run -n kd_mm_beam ...` 运行；OpenSpec 需要通过 strict validation。
