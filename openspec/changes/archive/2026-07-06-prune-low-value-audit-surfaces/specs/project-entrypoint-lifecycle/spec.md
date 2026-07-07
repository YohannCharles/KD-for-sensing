## ADDED Requirements

### Requirement: Scene31 重复 wrapper 必须收敛到 canonical command
项目 MUST 删除本轮审计确认的 Scene31 thin wrapper，并将 docs、tests、inventory 和推荐命令指向已有 canonical script。删除旧 wrapper 时 MUST 不新增 alias、compat wrapper、deprecation trampoline 或同职责 shell 包装。

#### Scenario: beamsoft weak summary wrapper 删除
- **WHEN** 协作者需要汇总 Scene31 beamsoft weak 结果
- **THEN** 当前推荐入口 MUST 是 `scripts/summarize_scene31_bc_next.py --root ...` 或等价 canonical owner 命令
- **AND** 项目 MUST 不保留 `scripts/summarize_scene31_beamsoft_weak.py` 作为只设置默认参数的 wrapper

#### Scenario: maskfix eval shell wrapper 删除
- **WHEN** 协作者需要运行 Scene31 maskfix reliability 评估
- **THEN** 当前推荐入口 MUST 是 `scripts/run_scene31_subset_reliability.sh --group eval_modular_lite_maskfix`
- **AND** 项目 MUST 不保留 `scripts/run_scene31_modular_maskfix_eval.sh` 或 `scripts/run_scene31_baseline_pack_maskfix_eval.sh` 作为只转发 group 的 wrapper

#### Scenario: 删除 wrapper 后文档不推荐旧路径
- **WHEN** README、docs、OpenSpec current specs、tests 或 `docs/project_surface_inventory.md` 提到 Scene31 maskfix 或 beamsoft weak 汇总
- **THEN** 它们 MUST 指向 canonical command 或将旧 path 标记为 retired/historical
- **AND** current surface guardrail MUST 不要求已删除 wrapper 存在

### Requirement: Wrapper 删除需要 focused guardrail
删除 Scene31 wrapper 后，项目 MUST 通过脚本 inventory、architecture boundary 或 focused tests 防止同职责 wrapper 回流。保留的 Scene31 脚本 MUST 有明确 owner、推荐关系、输入输出边界和 focused validation。

#### Scenario: scripts surface 检查拒绝 wrapper 回流
- **WHEN** 开发者运行 scripts surface doctor 或 architecture boundary check
- **THEN** 检查 MUST 不把已删除 Scene31 wrapper 列为 current allowlist
- **AND** 若同名文件或等价 forwarding wrapper 回流，检查 MUST 报告需要删除或登记新的 OpenSpec reason
