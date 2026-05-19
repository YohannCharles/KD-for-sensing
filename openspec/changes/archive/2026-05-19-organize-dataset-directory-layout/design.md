## Context

现有 DeepSense6G 场景由 `src/kd_sensing/data/scenes.py` 注册，默认路径是 `dataset/scenario9`、`dataset/scenario31`、`dataset/scenario32`。训练配置通过 `data.dataset.type: deepsense6g` 和 `data.dataset.scene` 选择场景，dataset 主类再用 `data_root + CSV 内相对路径` 加载 image、radar、GPS、LiDAR、mmWave 和 beam label。

问题不在单个 loader，而在目录约定没有数据集家族层级。后续 MMW 数据还会按天气和数据类型分层，如果继续把路径分散写在场景注册、预处理配置、README 和测试里，新增数据集时会重复路径规则并提高误用成本。

## Goals / Non-Goals

**Goals:**
- 将 DeepSense6G 的规范目录调整为 `dataset/DeepSense6G/scenario*`。
- 集中维护数据集家族、场景/条件和默认目录描述符。
- 让训练、评估、预处理和文档使用同一套默认根目录。
- 显式保留旧 `dataset/scenario*` 作为用户自定义 `data_root`，避免已经存在的本地数据立即失效。
- 给未来 MMW 的 `sunny/rainy/foggy` + `Sensor_Data/Channel_Data` 布局留下规范入口。

**Non-Goals:**
- 不移动、复制或下载真实数据文件。
- 不改 DeepSense6G CSV 列协议。
- 不改 CSV 内相对路径的含义，所有模态文件仍相对对应 scene root 解析。
- 不在本变更中实现 MMW dataset loader 或训练配置矩阵。

## Decisions

1. DeepSense6G 使用数据集家族目录作为新默认路径。

   新默认路径为 `dataset/DeepSense6G/scenario31`、`dataset/DeepSense6G/scenario32`、`dataset/DeepSense6G/scenario9`。这样 DeepSense6G 与未来 MMW、DeepVerse 等数据集家族在 `dataset/` 下平级，场景只作为家族内部的子目录。

   备选方案是继续使用 `dataset/scenario*` 并只新增 MMW 目录。这个方案短期改动小，但会让 DeepSense6G 成为唯一没有家族层的例外，后续文档和工具很难形成统一规则。

2. 不做默认路径的隐式旧目录回退。

   未显式配置 `data_root` 时，系统必须解析到新规范目录。旧目录继续通过显式 `data.dataset.data_root: dataset/scenario31` 使用。这样默认行为清晰，测试也能稳定验证新规范；需要渐进迁移的用户可以通过配置覆盖或创建软链接过渡。

   备选方案是“如果新目录不存在就自动回退旧目录”。该方案看似方便，但会让同一配置在不同机器上解析到不同物理目录，不利于复现实验。

3. 抽出 layout/descriptor 层，而不是继续扩展 `scenes.py` 的硬编码字符串。

   可以新增 `src/kd_sensing/data/layouts.py`，定义数据集家族布局和 helper，例如 DeepSense6G scene root、legacy root、MMW condition root 和 required subdirs。`scenes.py` 仍负责 DeepSense6G 场景语义，但默认路径从 layout helper 获取。

   备选方案是只修改 `DeepSenseScene.default_data_root` 字符串。这个方案能满足当前路径调整，但无法约束未来 MMW 的天气/数据类型目录，也无法减少路径规则散落。

4. 预处理配置继续以 scene root 为 `data_root`。

   对 DeepSense6G，`csv_path` 和生成的 split CSV 均放在 `dataset/DeepSense6G/scenarioXX` 内。CSV 内 `camera*`、`radar*`、`gps*`、`lidar*`、`mmwave*`、`beam*` 等列仍保存相对 scene root 的路径，因此现有 transform ops 的 `joined_resource(data_root, rel_path)` 语义不变。

5. MMW 先定义目录规范，不绑定 loader。

   未来 MMW 规范目录为 `dataset/MMW/<condition>/Sensor_Data` 与 `dataset/MMW/<condition>/Channel_Data`，其中 `<condition>` 首批为 `sunny`、`rainy`、`foggy`。本变更只提供目录命名和 layout helper 的预留，不假设文件格式、CSV 列名或标签构造方式。

## Risks / Trade-offs

- [Risk] 用户本地数据仍在旧 `dataset/scenario*`，默认训练命令会找不到新路径 → Mitigation: 文档给出 `mv`/软链接/显式 `data_root` 三种迁移方式，代码保持显式旧路径可用。
- [Risk] 预处理 scene override 只替换最后一层目录，无法正确处理 `dataset/DeepSense6G/scenario31` → Mitigation: 改为基于 DeepSense6G layout descriptor 重建 scene root 和 `scenarioXX_RA.csv`。
- [Risk] 部分测试硬编码旧路径 → Mitigation: 更新路径断言，补充旧 `data_root` 显式覆盖仍可用的测试。
- [Risk] 过早抽象影响简单性 → Mitigation: layout 层只管理目录和别名，不介入样本加载、CSV 解析或模型逻辑。

## Migration Plan

1. 新增或调整 layout descriptor，生成 DeepSense6G 新默认 scene root 和旧 legacy root。
2. 更新 DeepSense6G 场景注册默认路径，并让配置 normalize、retarget、metadata 使用新路径。
3. 更新预处理配置和 scene override 逻辑，使默认 `csv_path` 与 `data_root` 指向 `dataset/DeepSense6G/scenarioXX`。
4. 更新 README，说明推荐目录结构和旧目录迁移方式。
5. 更新测试覆盖默认路径、显式旧路径覆盖、预处理 scene override 和 CSV 相对路径语义。

Rollback 策略：如果新目录规范需要撤回，可将 DeepSense6G layout helper 的 canonical root 改回 `dataset/scenario*`，不需要改 dataset loader 或 transform ops。

## Open Questions

- MMW 的具体文件格式、索引 CSV 或标签协议尚未确定，本变更只固定物理目录约定。
