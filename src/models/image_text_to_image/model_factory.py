import logging
from src.models.image_text_to_image._base import ImageTextToImageModel


def get_model(config: dict) -> ImageTextToImageModel:
    provider = config["provider"]

    if provider == "flux/kontext":
        logging.info(f"Using Flux/Kontext model: {config['name']}")
        from src.models.image_text_to_image._flux_kontext import FluxKontext

        return FluxKontext(config)
    elif provider == "flux/klein":
        logging.info(f"Using Flux/Klein model: {config['name']}")
        from src.models.image_text_to_image._flux_klein import Flux2Klein

        return Flux2Klein(config)
    elif provider == "qwen/image_edit":
        logging.info(f"Using Qwen/Image Edit model: {config['name']}")
        from src.models.image_text_to_image._qwen_image_edit import QwenImageEdit

        return QwenImageEdit(config)
    else:
        raise ValueError(f"Unsupported model provider: {provider}")


if __name__ == "__main__":
    import os

    model = get_model(
        {"provider": "flux/kontext", "name": "black-forest-labs/FLUX.1-Kontext-dev"}
    )

    # model = model_factory.get_model({
    #     "provider": "qwen/image_edit",
    #     "name": "Qwen/Qwen-Image-Edit"
    # })

    image = "/leonardo/home/userexternal/flaiti00/development/real-time-assistant/data/fast-area/datasets/scenes/ppr10k/632/632_1.jpg"
    prompt = "invert the position of the two individuals, the guy moves to the left and the girl to the right"

    generated_image = model.generate_edited_image(prompt, image)
    generated_image.save(
        os.path.join(
            "/leonardo/home/userexternal/flaiti00/development/real-time-assistant/.trash",
            "generated_image.jpg",
        )
    )
