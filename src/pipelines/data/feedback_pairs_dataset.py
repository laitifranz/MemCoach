import logging
import os
import json
from typing import Sequence
from torch.utils.data import Dataset
from tqdm import tqdm

from src.pipelines.data.filter_feedback_scenes import max_delta_scene_pair
from src.utils._runtime_paths import resolve_project_relative_path

logger = logging.getLogger(__name__)


class FeedbackPairsDataset(Dataset):
    def __init__(self, config: dict, split: str = "all"):
        self.config = config
        self.split = split
        if split == "test" and not config.get("testset_scenes_path"):
            raise ValueError(
                "Test split requested but no testset_scenes_path provided in config."
            )

        self.dataset = self._build_dataset()
        self.testset_scene_ids = self._load_testset_scene_ids()
        self.train_indices, self.test_indices = self._build_split_indices()
        self.active_indices = self._resolve_active_indices()

    def __len__(self):
        if self.split == "all":
            return len(self.dataset)
        return len(self.active_indices)

    def __getitem__(self, idx):
        if self.split == "all":
            return self.dataset[idx]
        return self.dataset[self.active_indices[idx]]

    def _build_dataset(self):
        dataset = []
        data_path = str(
            resolve_project_relative_path(self.config["feedback_dataset_path"])
        )
        data_files = sorted(os.listdir(data_path))
        filtered_data_files = [
            file
            for file in data_files
            if file.endswith(".jsonl") and file.startswith("dataset")
        ]
        logger.info(
            f"Found {len(filtered_data_files)} data files in {self.config['feedback_dataset_path']}"
        )
        for data_file in tqdm(filtered_data_files, desc="Building dataset"):
            with open(os.path.join(data_path, data_file), "r") as f:
                for line in f:
                    data = json.loads(line)
                    dataset.append(data)
        logger.info(f"Built dataset with {len(dataset)} feedback pairs.")
        return dataset

    def _load_testset_scene_ids(self) -> set:
        testset_scenes_path = self.config.get("testset_scenes_path")
        if not testset_scenes_path:
            return set()

        full_path = str(resolve_project_relative_path(testset_scenes_path))
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"Test set scenes file not found: {full_path}")

        with open(full_path, "r") as handle:
            scenes = json.load(handle)
        logger.info(f"Loaded {len(scenes)} held-out scenes from {full_path}")
        return set(scenes)

    def _build_split_indices(self) -> tuple[list[int], list[int]]:
        train_indices: list[int] = []
        test_indices: list[int] = []
        for idx, record in enumerate(self.dataset):
            scene_id = record.get("scene_id")
            if scene_id in self.testset_scene_ids:
                test_indices.append(idx)
            else:
                train_indices.append(idx)
        logger.info(
            f"Computed split indices: {len(train_indices)} train, {len(test_indices)} test"
        )
        return train_indices, test_indices

    def _resolve_active_indices(self) -> Sequence[int]:
        if self.split == "train":
            return self.train_indices
        if self.split == "test":
            return self.test_indices
        if self.split == "all":
            return range(len(self.dataset))
        raise ValueError(
            f"Unknown split '{self.split}'. Expected one of ['all', 'train', 'test']."
        )

    def group_by_scene_id(
        self, dataset: Sequence[dict] | None = None
    ) -> dict[str, list[dict]]:
        if dataset is None:
            dataset = self
        grouped_dataset = {}
        for item in dataset:
            scene_id = item["scene_id"]
            if scene_id not in grouped_dataset:
                grouped_dataset[scene_id] = []
            grouped_dataset[scene_id].append(item)
        return grouped_dataset

    def apply_criteria(
        self, scene_list: list[dict], filter_criteria: str | None = None
    ) -> dict:
        if filter_criteria is None:
            filter_criteria = self.config.get("filter_criteria")
        if not filter_criteria:
            raise ValueError("Missing filter_criteria in dataset config")
        if filter_criteria == "less_higher_two_elements":
            return max_delta_scene_pair(scene_list)
        raise ValueError(f"Criteria {filter_criteria} not found")

    def group_by_scene_id_with_criteria(
        self, dataset: Sequence[dict] | None = None, filter_criteria: str | None = None
    ) -> list[dict]:
        grouped_dataset = self.group_by_scene_id(dataset)
        return [
            self.apply_criteria(scene_list, filter_criteria)
            for scene_list in grouped_dataset.values()
        ]


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    import fire

    fire.Fire(
        FeedbackPairsDataset(
            config={
                "feedback_dataset_path": "experiments/ppr10k_data_zero_shot/ppr10k/InternVL3_5-8B-HF/zero_shot_memorability_prompt",
                "testset_scenes_path": "config/test_set_scenes_seed42.json",
            },
            split="test",
        )
    )
