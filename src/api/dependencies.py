import copy
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from omegaconf import OmegaConf

from src.api.config import PROJECT_ROOT, get_settings
from src.models.target_predictor.model_factory import get_model
from src.pipelines.method.core.inference_manager import InferenceManager

settings = get_settings()


def _resolve_path(path_str: str) -> str:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


@lru_cache
def get_memorability_model():
    base_config = settings.memorability.model_dump()
    base_config["mlp_checkpoint_path"] = _resolve_path(
        base_config["mlp_checkpoint_path"]
    )
    config = {"model": base_config}
    return get_model(config)


def _load_feedback_config() -> Dict[str, Any]:
    config_path = _resolve_path(settings.feedback.hydra_config)
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)

    activation_cfg = cfg.get("activation_settings", {})
    if "negative_actvs_file" in activation_cfg:
        activation_cfg["negative_actvs_file"] = _resolve_path(
            activation_cfg["negative_actvs_file"]
        )
    if "positive_actvs_file" in activation_cfg:
        activation_cfg["positive_actvs_file"] = _resolve_path(
            activation_cfg["positive_actvs_file"]
        )

    if settings.feedback.coeff_override is not None:
        activation_cfg["coeff"] = settings.feedback.coeff_override
    if settings.feedback.target_layer_override is not None:
        activation_cfg["target_layer"] = settings.feedback.target_layer_override

    cfg["activation_settings"] = activation_cfg
    return cfg


@lru_cache
def _feedback_config_template() -> Dict[str, Any]:
    return _load_feedback_config()


def _build_feedback_config() -> Dict[str, Any]:
    return copy.deepcopy(_feedback_config_template())


@lru_cache
def get_feedback_manager():
    return InferenceManager(_build_feedback_config())


def get_feedback_prompt_version(default: str = "inference_prompt") -> str:
    prompt_cfg = _feedback_config_template().get("prompt", {})
    return prompt_cfg.get("version", default)
