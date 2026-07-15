PYTHON ?= conda run -n kd_mm_beam python
PYTEST ?= conda run -n kd_mm_beam pytest
OPENSPEC ?= openspec

.PHONY: verify verify-quick verify-cli-config verify-docs verify-compile verify-full

verify: verify-quick

verify-quick:
	$(OPENSPEC) validate --all --strict
	$(PYTEST) tests/test_architecture_boundaries.py -q

verify-cli-config:
	$(PYTEST) tests/test_cli_help.py tests/test_config_load_characterization.py -q

verify-docs:
	$(OPENSPEC) validate --all --strict
	$(PYTEST) tests/test_architecture_boundaries.py -q

verify-compile:
	$(PYTHON) scripts/verify_compile.py

verify-full: verify-quick verify-cli-config verify-compile
	$(PYTEST) -q
