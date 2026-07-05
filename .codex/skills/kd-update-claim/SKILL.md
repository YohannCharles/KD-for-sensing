---
name: kd-update-claim
description: Update research claims, paper tables, provenance notes, and claim-facing docs from local evidence without promoting draft or ignored artifacts.
license: MIT
---

# kd-update-claim

Use this skill when updating `docs/result_claims_registry.md`, paper table notes, experiment protocols, claim provenance, or claim-facing summaries.

## Required context

1. Read `AGENTS.md`, `docs/agent_navigation.md`, `docs/maintainer_context_index.yaml`, and `docs/agent_context/claims.md`.
2. Read `docs/result_claims_registry.md`, `docs/experiment_protocols.md`, and the relevant OpenSpec specs such as `openspec/specs/mainline-experiment-documentation/spec.md` and `openspec/specs/research-claim-harvester/spec.md`.
3. If the claim requires a new workflow, config family, metric definition, or data contract, open or continue an OpenSpec change before updating code or docs.

## Workflow

1. Check provenance: run id, config path, split/protocol, seed, checkpoint, metric definition, status, caveat, and blocked reason.
2. Keep candidate-only, mock, smoke, historical, upper-bound, pending, and blocked rows out of reviewed main claims.
3. Update claim docs and protocol docs together when the statement depends on parameter settings or evaluation scope.
4. Do not infer current facts from ignored `outputs/`, `logs/`, dashboard JSON, or local ledger without explicit provenance.

## Commands and artifacts

- Run project Python commands through `conda run -n kd_mm_beam ...`.
- Draft ledgers, paper exports, figures, reports, and analysis tables belong under ignored `outputs/` or `logs/`; real `dataset/` contents and checkpoint files are not source changes.
- Useful validation:

```bash
conda run -n kd_mm_beam kd-sensing-paper-export --input docs/result_claims_registry.md --output-dir outputs/paper_artifacts/current
conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q
```
