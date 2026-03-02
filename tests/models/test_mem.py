import os

import pytest
from dotenv import load_dotenv

from src.models.target_predictor.memorability._ours import MemBenchMemorabilityPredictor
from src.models.target_predictor.memorability._vitmem import ViTMem
from src.utils._runtime_paths import resolve_project_root

load_dotenv()


@pytest.fixture
def image_path():
    return os.path.join(resolve_project_root(), "tests/models/assets/plane.jpg")


def test_mem_vitmem(image_path):
    model = ViTMem(config={})
    memorability = model.predict(image_path)
    assert memorability is not None
    assert memorability >= 0
    assert memorability <= 1


def test_mem_ours(image_path):
    model = MemBenchMemorabilityPredictor(
        config={
            "model": {
                "mlp_checkpoint_path": "ckpt/target_predictor/memorability/ours/model_weights.pth"
            },
            "normalize_fts": False,
        }
    )
    memorability = model.predict(image_path)
    assert memorability is not None
    assert memorability >= 0
    assert memorability <= 1
