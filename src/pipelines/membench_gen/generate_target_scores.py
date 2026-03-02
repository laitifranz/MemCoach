import os
from datetime import datetime

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import logging

from src.models.target_predictor.model_factory import get_model
from src.utils._logging_utils import setup_logging
from src.utils._runtime_paths import resolve_project_experiments_root
from src.utils._runtime_paths import resolve_project_dataset_root

setup_logging()
logger = logging.getLogger(__name__)


class ImageDataset(Dataset):
    def __init__(self, data_path, recursive: bool = False):
        if recursive:
            self.image_paths = list(self._list_images_recursive(data_path))
        else:
            self.image_paths = [
                os.path.join(data_path, f)
                for f in os.listdir(data_path)
                if self._is_an_image(f)
            ]

    def __getitem__(self, idx):
        return self.image_paths[idx]

    def __len__(self):
        return len(self.image_paths)

    def _is_an_image(self, path: str) -> bool:
        return path.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))

    def _list_images_recursive(self, root_dir: str):
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                if self._is_an_image(fname):
                    yield os.path.join(dirpath, fname)


class InferenceTargetPredictor:
    def __init__(self, config: dict):
        self.config = config
        self.model = get_model(config)
        self.batch_size = config.get("batch_size", 64)

    def process_scene(self, scene_path):
        dataset = ImageDataset(scene_path, recursive=False)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        scores = {}
        for image_paths in dataloader:
            score = self.model.predict(image_paths)
            for path, s in zip(image_paths, score):
                scores[os.path.basename(path)] = s.item()
        return scores

    def process_dataset(self):
        root_folder = os.path.join(
            resolve_project_dataset_root(), self.config["dataset"]
        )
        for subset in tqdm(os.listdir(root_folder)):
            output_path = os.path.join(
                resolve_project_experiments_root(),
                "target_predictor",
                self.config["model"]["provider"],
                self.config["dataset"],
                subset,
                "scores.json",
            )

            if os.path.exists(output_path):
                if self.config["overwrite"]:
                    logger.info(
                        f"Score file {output_path} already exists, overwriting..."
                    )
                else:
                    logger.info(f"Score file {output_path} already exists, skipping...")
                    continue

            if self.config["dry_run"]:
                logger.info(
                    f"Dry run, should process {subset} and save to {output_path}"
                )
                continue

            scores = self.process_scene(
                os.path.join(root_folder, subset, self.config["subfolder"])
            )

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump({"metadata": self.config, "scores": scores}, f, indent=4)
            logger.info(f"Saved scores for {subset} to {output_path}")


def main(
    dataset="ppr10k",
    subfolder="",
    provider="ours",
    mlp_checkpoint_path="",
    overwrite=False,
    dry_run=False,
):
    config = {
        "model": {
            "provider": provider,
            "mlp_checkpoint_path": mlp_checkpoint_path,
        },
        "dataset": dataset,
        "batch_size": 256,
        "overwrite": overwrite,
        "dry_run": dry_run,
        "subfolder": subfolder,
        "datetime": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    inference = InferenceTargetPredictor(config)
    inference.process_dataset()


if __name__ == "__main__":
    import fire
    from dotenv import load_dotenv

    load_dotenv()
    fire.Fire(main)
