import logging
import os

from src.utils._logging_utils import setup_logging
from src.pipelines.method.core.training_manager import TrainingManager
from src.utils._runtime_paths import build_stage_save_path, load_yaml_config

setup_logging()
logger = logging.getLogger(__name__)


def main(
    config_path: str,
    overwrite: bool = False,
    dry_run: bool = False,
    include_datetime: bool = False,
):
    config = load_yaml_config(config_path)
    config["statistics"] = {}

    save_path = build_stage_save_path(config, include_datetime=include_datetime)

    if os.path.exists(save_path):
        if overwrite:
            logger.info("Dataset already exists, overwriting")
        else:
            logger.info("Dataset already exists, skipping generation")
            return

    training_manager = TrainingManager(config)
    support_set_dataloader = training_manager.prepare_support_set_dataloader()

    config["statistics"]["support_set_size"] = len(support_set_dataloader) * config[
        "model"
    ].get("batch_size", 8)

    if not dry_run:
        support_set_activations = training_manager.extract_activations_from_support_set(
            support_set_dataloader
        )
        training_manager.save_activations(support_set_activations, save_path)
    else:
        logger.info("DRY RUN: Skipping activation extraction")
        logger.info(f"Save path: {save_path}")
        import pprint

        logger.info("Config:\n%s", pprint.pformat(config, indent=4, width=120))


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    import fire

    fire.Fire(main)
