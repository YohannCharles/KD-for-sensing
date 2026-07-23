from pathlib import Path

import pytest

from kd_sensing.cli import evaluate, preprocess, train
from kd_sensing.cli.common import load_cli_config, parse_cli_args


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/mmw/u0.yaml"


@pytest.mark.parametrize("runner", (train.run, evaluate.run))
def test_train_and_evaluate_reject_unknown_option_before_runtime(runner) -> None:
    with pytest.raises(SystemExit) as exc_info:
        runner(["--config", str(CONFIG), "--num-wokers", "1"])

    assert exc_info.value.code == 2


@pytest.mark.parametrize("runner", (train.run, evaluate.run))
def test_train_and_evaluate_reject_unknown_dotted_override(runner) -> None:
    with pytest.raises(SystemExit) as exc_info:
        runner(["--config", str(CONFIG), "training.lrr=0.001"])

    assert exc_info.value.code == 2


def test_cli_keeps_known_bare_and_explicit_overrides() -> None:
    parser = train.build_parser()
    args, unknown = parse_cli_args(
        parser,
        ["--config", str(CONFIG), "training.lr=0.001", "--override", "data.dataloader.num_workers=0"],
    )

    cfg = load_cli_config(args, unknown, parser=parser)

    assert cfg["training"]["lr"] == 0.001
    assert cfg["data"]["dataloader"]["num_workers"] == 0


def test_preprocess_rejects_unknown_option_without_starting_action() -> None:
    with pytest.raises(SystemExit) as exc_info:
        preprocess.main(["--action", "mmw_sequence_splits_from_manifest", "--num-wokers", "1"])

    assert exc_info.value.code == 2
