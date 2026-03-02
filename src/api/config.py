from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from omegaconf import OmegaConf
from pydantic import BaseModel

from src.utils._runtime_paths import resolve_project_root

load_dotenv()


PROJECT_ROOT = resolve_project_root()


class RuntimeConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class MemorabilityConfig(BaseModel):
    provider: str
    mlp_checkpoint_path: str


class FeedbackConfig(BaseModel):
    hydra_config: str
    coeff_override: Optional[float] = None
    target_layer_override: Optional[int] = None


class ApiSettings(BaseModel):
    runtime: RuntimeConfig
    memorability: MemorabilityConfig
    feedback: FeedbackConfig


@lru_cache
def get_settings() -> ApiSettings:
    config_path = PROJECT_ROOT / "config/api/server.yaml"
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    return ApiSettings(**cfg)
