## 验证记录

- `conda run -n kd_mm_beam python -m compileall -q src scripts tests`：通过。
- `conda run -n kd_mm_beam pytest -q tests/test_training_io_workflow.py`：`27 passed`。
- `conda run -n kd_mm_beam pytest -q tests`：`274 passed`。
- `openspec validate --all`：`17 passed, 0 failed`。
