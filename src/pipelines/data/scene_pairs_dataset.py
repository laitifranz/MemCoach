from torch.utils.data import Dataset
import os
import json
from tqdm import tqdm
import logging
from src.pipelines.data.criteria_order_pairs import apply_criteria_less_to_higher
from src.utils._runtime_paths import (
    resolve_project_dataset_root,
    resolve_project_relative_path,
)

logger = logging.getLogger(__name__)


class ScenePairsDataset(Dataset):
    def __init__(self, config: dict):
        self._config = config
        self.dataset = self._build_dataset()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def _build_dataset(self):
        dataset = []
        dataset_path = resolve_project_dataset_root() / self._config["dataset_name"]
        scene_ids = sorted(os.listdir(dataset_path))
        for scene_id in tqdm(scene_ids, desc="Building dataset"):
            target_scores = self._load_target_scores(
                str(
                    resolve_project_relative_path(self._config["target_score_path"])
                    / scene_id
                    / "scores.json"
                )
            )
            lowest_score_keys, highest_score_key = apply_criteria_less_to_higher(
                target_scores
            )
            scene_entries = []
            for lowest_score_key in lowest_score_keys:
                scene_entries.append(
                    {
                        "scene_id": scene_id,
                        "source_image_path": os.path.join(
                            scene_id, lowest_score_key[0]
                        ),
                        "target_image_path": os.path.join(
                            scene_id, highest_score_key[0]
                        ),
                        "source_score": lowest_score_key[1],
                        "target_score": highest_score_key[1],
                    }
                )
            dataset.extend(scene_entries)
        logger.info(f"Built dataset with {len(dataset)} scene pairs")
        return dataset

    def _load_target_scores(self, json_path: str):
        with open(json_path, "r") as f:
            target_scores = json.load(f)["scores"]
        return target_scores


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    import fire

    fire.Fire(
        ScenePairsDataset(
            config={
                "dataset_name": "ppr10k",
                "target_score_path": "experiments/target_predictor/ours/ppr10k",
            }
        )
    )
