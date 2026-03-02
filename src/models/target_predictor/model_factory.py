from src.models.target_predictor._base import BaseTargetPredictor


def get_model(config: dict) -> BaseTargetPredictor:
    model_cfg = config.get("model", config)
    if not isinstance(model_cfg, dict):
        raise ValueError(
            "Invalid target predictor config. Expected a dict for 'model'."
        )

    provider = model_cfg.get("provider")
    if not provider:
        raise ValueError(
            "Missing target predictor provider. Expected 'model.provider' or 'provider'."
        )

    normalized_config = config if "model" in config else {"model": model_cfg}

    if provider == "vitmem":
        from src.models.target_predictor.memorability._vitmem import ViTMem

        return ViTMem(normalized_config)
    elif provider == "ours":
        from src.models.target_predictor.memorability._ours import (
            MemBenchMemorabilityPredictor,
        )

        return MemBenchMemorabilityPredictor(normalized_config)
    else:
        raise ValueError(f"Unsupported model provider: {provider}")
