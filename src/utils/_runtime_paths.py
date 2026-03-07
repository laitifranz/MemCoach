import os
from datetime import datetime
from pathlib import Path

import yaml


def _is_default_placeholder(raw_value: str) -> bool:
    return "<" in raw_value and ">" in raw_value


def load_yaml_config(config_path: str) -> dict:
    with open(config_path, "r") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(
            f"Expected a mapping config in {config_path}, got {type(loaded)}"
        )
    return loaded


def _resolve_env_path(var_name: str, project_root: Path) -> Path:
    raw_value = os.getenv(var_name)
    if not raw_value:
        raise EnvironmentError(f"Missing required environment variable: {var_name}")

    path = Path(raw_value)
    if not path.is_absolute():
        path = project_root / path
    return path


def resolve_project_root() -> Path:
    env_root = os.getenv("__PROJECT_ROOT__")
    if env_root and not _is_default_placeholder(env_root):
        return Path(env_root).resolve()
    return Path(__file__).resolve().parents[2]


def resolve_project_dataset_root() -> Path:
    project_root = resolve_project_root()
    raw_value = os.getenv("__PROJECT_DATASET_ROOT__")
    if not raw_value or _is_default_placeholder(raw_value):
        return project_root / "dataset"
    path = Path(raw_value)
    if not path.is_absolute():
        path = project_root / path
    return path


def resolve_project_experiments_root() -> Path:
    project_root = resolve_project_root()
    raw_value = os.getenv("__PROJECT_EXP_FOLDER__")
    if not raw_value or _is_default_placeholder(raw_value):
        return project_root / "experiments"
    path = Path(raw_value)
    if not path.is_absolute():
        path = project_root / path
    return path


def resolve_project_relative_path(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return resolve_project_root() / path


def get_model_config(config: dict) -> dict:
    if "model" in config and isinstance(config["model"], dict):
        return config["model"]
    return config


def get_model_name(config: dict) -> str:
    model_cfg = get_model_config(config)
    model_name = model_cfg.get("name")
    if not model_name:
        raise ValueError(
            "Missing model name in config. Expected 'name' or 'model.name'."
        )
    return model_name


def resolve_dataset_image_path(dataset_name: str, image_relative_path: str) -> str:
    dataset_root = resolve_project_dataset_root()
    return str(dataset_root / dataset_name / image_relative_path)


def build_stage_save_path(config: dict, include_datetime: bool = False) -> str:
    experiments_root = resolve_project_experiments_root()
    task_name = config["task_name"]
    if include_datetime:
        task_name = f"{task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    save_path = (
        experiments_root
        / task_name
        / config["data"]["dataset_name"]
        / get_model_name(config).split("/")[-1]
    )

    prompt_version = config.get("prompt", {}).get("version")
    if prompt_version:
        save_path = save_path / prompt_version

    return str(save_path)
