from src.models.image_text_to_text._base import ImageTextToTextModel


def get_model(config: dict) -> ImageTextToTextModel:
    model_cfg = config.get("model", config)
    if not isinstance(model_cfg, dict):
        raise ValueError(
            "Invalid image-text model config. Expected a dict for 'model'."
        )

    provider = model_cfg.get("provider")
    if not provider:
        raise ValueError(
            "Missing image-text model provider. Expected 'model.provider' or 'provider'."
        )

    if provider == "hf":
        from src.models.image_text_to_text._hf import HfModel

        return HfModel(model_cfg)
    elif provider == "litellm":
        from src.models.image_text_to_text._litellm import LiteLLMModel

        return LiteLLMModel(model_cfg)
    else:
        raise ValueError(f"Unsupported model provider: {provider}")
