## MODIFIED Requirements

### Requirement: 统一新脚本入口并移除旧入口
项目 MUST 使用 `pyproject.toml` 声明的 `kd-sensing-*` package console scripts 或包内 CLI module 作为当前支持的运行入口。`scripts/*.py` 中只转发到包内 CLI 的 Python thin alias MUST 从当前支持面删除，不得作为 README、AGENTS、docs、OpenSpec 或维护索引推荐入口。项目 MUST 继续删除现有顶层旧脚本入口，包括 `train_image.py`、`train_both.py`、`test_model_image.py`、`test_model_both.py`、`CSV_process.py` 和 `gen_data_seq.py`，不得保留兼容包装脚本。

#### Scenario: 运行训练 console script 帮助信息
- **WHEN** 开发者执行 `conda run -n kd_mm_beam kd-sensing-train --help`
- **THEN** 命令 MUST 正常退出
- **AND** 帮助信息 MUST 展示配置文件、训练任务和命令行覆盖相关的参数说明

#### Scenario: 运行评估和预处理 console script 帮助信息
- **WHEN** 开发者执行 `conda run -n kd_mm_beam kd-sensing-evaluate --help` 或 `conda run -n kd_mm_beam kd-sensing-preprocess --help`
- **THEN** 命令 MUST 正常退出
- **AND** 帮助信息 MUST 展示对应任务的参数说明

#### Scenario: 旧脚本入口和 thin alias 已删除
- **WHEN** 结构收敛完成后检查仓库根目录和 `scripts/`
- **THEN** 根目录 MUST 不存在 `train_image.py`、`train_both.py`、`test_model_image.py`、`test_model_both.py`、`CSV_process.py` 或 `gen_data_seq.py`
- **AND** `scripts/` MUST 不保留 `train.py`、`evaluate.py`、`preprocess.py`、`check_dataset.py`、`eval_baseline.py`、`train_baseline.py`、`train_beambench_image_ae_gps.py` 或 `run_beambench_image_ae_gps_tableiii.py` 这类只转发到 package CLI 的 Python thin alias

#### Scenario: 文档不推荐 thin alias
- **WHEN** 开发者阅读 README、AGENTS、`docs/agent_navigation.md`、维护索引或当前 OpenSpec specs 中的运行入口说明
- **THEN** 当前训练、评估、预处理和 BeamBench workflow MUST 指向 `kd-sensing-*` console scripts 或明确包内 CLI
- **AND** 文档 MUST 不把已删除的 `scripts/*.py` thin alias 写成当前推荐命令
