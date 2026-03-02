import os

import pytest
import torch
from dotenv import load_dotenv
from PIL import Image as PILImage

from src.models.image_text_to_image._flux_kontext import FluxKontext
from src.utils._runtime_paths import resolve_project_root

load_dotenv()


@pytest.fixture
def image_path():
    return os.path.join(resolve_project_root(), "tests/models/assets/plane.jpg")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_edit_flux_kontext(image_path):
    model = FluxKontext(config={"name": "black-forest-labs/FLUX.1-Kontext-dev"})
    image = model.generate(prompt="Change the airplane into a bird", image=image_path)
    assert image is not None
    assert isinstance(image, PILImage.Image)
