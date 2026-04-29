## 1. 配置与解析

- [x] 1.1 在默认配置中新增 `data.cache` 全局和模态级 policy 字段，并保持旧配置兼容。
- [x] 1.2 实现 cache policy 解析 helper，将 `off/read_only/auto/rebuild` 转换为 image/LiDAR dataset 实际读写开关。
- [x] 1.3 将解析逻辑接入训练、评估和 profile 的 dataset 构建路径，并确保只作用于启用模态。

## 2. Cache 写入安全与运行记录

- [x] 2.1 为 image motion mask cache 写入增加原子写入或等价保护。
- [x] 2.2 为 LiDAR BEV cache 写入增加原子写入或等价保护，并支持 `rebuild` 覆盖语义。
- [x] 2.3 在训练、评估和 profile 输出中记录实际生效 cache policy、cache 目录和读写状态。

## 3. 文档与测试

- [x] 3.1 增加测试覆盖 policy 解析、按模态启用、非相关模态不访问 cache、cache miss 写入和低层覆盖。
- [x] 3.2 更新 README，说明自动 cache policy、各模态可复用范围、清理方式和推荐命令。
- [x] 3.3 运行 `conda run -n kd_mm_beam pytest -q tests` 或更精确目标测试。
- [x] 3.4 运行 `openspec validate --all` 并记录结果；如 CLI 异常，记录文件级校验结果。
