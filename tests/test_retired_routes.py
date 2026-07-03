import importlib.util
from pathlib import Path

import pytest

from kd_sensing.config import load_config
from kd_sensing.engine.objectives.metadata import objective_spec, resolve_prediction_objective
from kd_sensing.registries import DATASETS, LOSSES, MODELS, PREPROCESSORS, RegistryError, import_default_components


ROOT = Path(__file__).resolve().parents[1]


RETIRED_ROUTES = (
    {
        "name": "hist_beam",
        "config_paths": ("configs/hist_beam/quick_smoke.yaml",),
        "console_fragments": ("kd-sensing-hist-beam-loso",),
        "module_paths": (
            "kd_sensing.models.fusion.hist_beam",
            "kd_sensing.engine.hist_beam_loso",
        ),
        "overrides": (
            "hist_beam.enabled=true",
            "model.primary.type=hist_beam_fusion",
        ),
        "message": "HiST-Beam|Hist",
        "objectives": (),
        "model_types": ("hist_beam_fusion",),
        "loss_types": (),
        "dataset_types": (),
        "preprocessor_types": (),
    },
    {
        "name": "bgam_top8_viewer",
        "config_paths": (
            "configs/deepsense6g_gps_lidar_bgam.yaml",
            "configs/mmw_town_gps_lidar_bgam.yaml",
            "configs/deepsense6g_top8_selector.yaml",
            "configs/diagnostics/modality_visualization.yaml",
        ),
        "console_fragments": (
            "kd-sensing-export-viewer-manifest",
            "kd-sensing-visualize-modalities",
            "gps-lidar-bgam",
            "top8",
        ),
        "module_paths": (
            "kd_sensing.data.deepsense6g_topk_candidate_manifest",
            "kd_sensing.data.mmw_town_topk_candidate_manifest",
            "kd_sensing.cli.export_viewer_manifest",
            "kd_sensing.cli.visualize_modalities",
        ),
        "overrides": (
            "experiment.name=deepsense6g_gps_lidar_bgam_reranker",
            "experiment.name=mmw_town_gps_lidar_bgam_reranker",
            "model.primary.type=gps_lidar_bgam_beam_predictor",
            "diagnostics.visualization.enabled=true",
        ),
        "message": "BGAM|Viewer",
        "objectives": (),
        "model_types": ("gps_lidar_bgam_beam_predictor",),
        "loss_types": (),
        "dataset_types": (),
        "preprocessor_types": (),
    },
    {
        "name": "raymobtime_s008",
        "config_paths": (
            "configs/raymobtime/s008_multitask_selection.yaml",
            "configs/preprocess/raymobtime_s008_cache.yaml",
        ),
        "console_fragments": ("raymobtime",),
        "module_paths": (
            "kd_sensing.data.datasets.raymobtime_s008",
            "kd_sensing.data.deepverse",
            "kd_sensing.preprocessing.raymobtime_s008",
        ),
        "overrides": (
            "data.dataset.type=raymobtime_s008",
            "preprocessing.type=raymobtime_s008_cache",
            "model.primary.type=simple_concat_multitask_selection",
        ),
        "message": "Raymobtime s008",
        "objectives": (),
        "model_types": ("simple_concat_multitask_selection",),
        "loss_types": (),
        "dataset_types": ("raymobtime_s008", "multimodal_nf"),
        "preprocessor_types": ("raymobtime_s008_cache", "multimodal_nf_index"),
    },
    {
        "name": "legacy_kd",
        "config_paths": (
            "configs/radar/logits_kd.yaml",
            "configs/radar/rkd.yaml",
            "configs/fusion/image_radar_logits_kd.yaml",
        ),
        "console_fragments": (),
        "module_paths": (
            "kd_sensing.losses.logits_kd",
            "kd_sensing.losses.rkd",
        ),
        "overrides": (
            "distillation.type=logits_kd",
            "kd_mode=logits_kd",
        ),
        "message": "KD support has been removed|Unknown loss",
        "objectives": (),
        "model_types": (),
        "loss_types": ("logits_kd", "rkd"),
        "dataset_types": (),
        "preprocessor_types": (),
    },
    {
        "name": "jepa_msac",
        "config_paths": (
            "configs/pretraining/jepa_msac_s32_smoke.yaml",
            "configs/pretraining/jepa_msac_s32_paper.yaml",
        ),
        "console_fragments": (
            "kd-sensing-run-jepa-msac",
            "kd_sensing.cli.run_jepa_msac",
        ),
        "module_paths": (
            "kd_sensing.cli.run_jepa_msac",
            "kd_sensing.baselines.jepa_msac",
            "kd_sensing.models.jepa_msac",
            "kd_sensing.losses.jepa_msac",
        ),
        "overrides": (
            "experiment.objective=jepa_msac_pretraining",
            "model.primary.type=jepa_msac",
            "workflow.jepa_msac.enabled=true",
            "loss.jepa_msac.weight=1.0",
        ),
        "message": "JEPA-MSAC|priority legacy workflow",
        "objectives": ("jepa_msac_pretraining",),
        "model_types": ("jepa_msac",),
        "loss_types": (),
        "dataset_types": (),
        "preprocessor_types": (),
    },
    {
        "name": "amr_net_gps_image",
        "config_paths": ("configs/baselines/amr_net_gps_image.yaml",),
        "console_fragments": (
            "kd-sensing-run-amr-net-gps-image",
            "kd_sensing.cli.run_amr_net_gps_image",
        ),
        "module_paths": (
            "kd_sensing.cli.run_amr_net_gps_image",
            "kd_sensing.baselines.amr_net_gps_image",
        ),
        "overrides": (
            "experiment.baseline_preset=amr_net_gps_image",
            "experiment.paper=amr_net_gps_image",
            "model.primary.model_name=AMR-Net_gps_image",
            "model.primary.baseline_preset=amr_net_gps_image",
            "amr_net_gps_image.enabled=true",
        ),
        "message": "AMR-Net_gps_image|priority legacy workflows",
        "objectives": (),
        "model_types": ("amr_net_gps_image",),
        "loss_types": (),
        "dataset_types": (),
        "preprocessor_types": (),
    },
)


def _cases(field: str):
    return [
        pytest.param(route["name"], item, route, id=f"{route['name']}:{item}")
        for route in RETIRED_ROUTES
        for item in route[field]
    ]


@pytest.mark.parametrize(("route_name", "rel_path", "route"), _cases("config_paths"))
def test_retired_config_paths_fail_fast(route_name: str, rel_path: str, route: dict):
    del route_name
    config_path = ROOT / rel_path

    assert not config_path.exists()
    with pytest.raises(ValueError, match=str(route["message"])):
        load_config(config_path)


@pytest.mark.parametrize(("route_name", "fragment", "route"), _cases("console_fragments"))
def test_retired_console_scripts_are_not_declared(route_name: str, fragment: str, route: dict):
    del route_name, route
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert fragment not in pyproject


@pytest.mark.parametrize(("route_name", "module_name", "route"), _cases("module_paths"))
def test_retired_module_paths_are_not_current(route_name: str, module_name: str, route: dict):
    del route_name, route
    assert importlib.util.find_spec(module_name) is None


@pytest.mark.parametrize(("route_name", "override", "route"), _cases("overrides"))
def test_retired_config_content_fails_fast(route_name: str, override: str, route: dict):
    del route_name
    with pytest.raises(ValueError, match=str(route["message"])):
        load_config(overrides=[override])


@pytest.mark.parametrize(("route_name", "objective", "route"), _cases("objectives"))
def test_retired_prediction_objectives_are_not_current(route_name: str, objective: str, route: dict):
    del route_name, route
    with pytest.raises(ValueError, match="experiment.objective must be one of"):
        resolve_prediction_objective({"experiment": {"objective": objective}})
    with pytest.raises(ValueError, match="Unknown prediction objective"):
        objective_spec(objective)


@pytest.mark.parametrize(("route_name", "model_type", "route"), _cases("model_types"))
def test_retired_model_registry_names_are_not_current(route_name: str, model_type: str, route: dict):
    del route_name, route
    import_default_components()

    assert model_type not in MODELS.list()
    with pytest.raises(RegistryError, match=model_type):
        MODELS.build({"type": model_type})


@pytest.mark.parametrize(("route_name", "loss_type", "route"), _cases("loss_types"))
def test_retired_loss_registry_names_are_not_current(route_name: str, loss_type: str, route: dict):
    del route_name, route
    import_default_components()

    assert loss_type not in LOSSES.list()
    with pytest.raises(RegistryError, match=loss_type):
        LOSSES.build({"type": loss_type})


@pytest.mark.parametrize(("route_name", "dataset_type", "route"), _cases("dataset_types"))
def test_retired_dataset_registry_names_are_not_current(route_name: str, dataset_type: str, route: dict):
    del route_name, route
    import_default_components()

    assert dataset_type not in DATASETS.list()
    with pytest.raises(RegistryError, match=dataset_type):
        DATASETS.build({"type": dataset_type})


@pytest.mark.parametrize(("route_name", "preprocessor_type", "route"), _cases("preprocessor_types"))
def test_retired_preprocessor_registry_names_are_not_current(route_name: str, preprocessor_type: str, route: dict):
    del route_name, route
    import_default_components()

    assert preprocessor_type not in PREPROCESSORS.list()
    with pytest.raises(RegistryError, match=preprocessor_type):
        PREPROCESSORS.build({"type": preprocessor_type})
