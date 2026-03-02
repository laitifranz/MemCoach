# code adapted from https://github.com/brainpriority/vitmem/blob/master/src/vitmem/__init__.py

import os
import ssl
import torch
from PIL import Image
from torchvision import transforms
import timm

from src.models.target_predictor._base import BaseTargetPredictor


NORMALIZE_MEAN = (0.0, 0.0, 0.0)
NORMALIZE_STD = (1.0, 1.0, 1.0)

transform = transforms.Compose(
    [
        transforms.Resize(
            size=(256, 256),
            interpolation=transforms.InterpolationMode.BILINEAR,
            max_size=None,
            antialias=None,
        ),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ]
)


class ViTMem_model(torch.nn.Module):
    def __init__(self, arch="vit_base_patch16_224_miil"):
        super().__init__()
        self.arch = arch
        self.vit = timm.create_model(self.arch, pretrained=False, num_classes=1)

    def forward(self, x):
        vitfeat = self.vit(x)
        out = torch.sigmoid(vitfeat)
        return out


class ViTMem(BaseTargetPredictor):
    def __init__(self, config: dict):
        super().__init__(config)
        model_url = "https://brainpriority.com/models/vitmem_base_patch16_224_miil_v1-f580d24a.pth"
        model_filename = os.path.basename(model_url)
        hub_path = torch.hub.get_dir()
        checkpoints_path = os.path.join(hub_path, "checkpoints")
        model_path = os.path.join(checkpoints_path, model_filename)

        if not os.path.exists(model_path):
            print("Seems like this is the first time you are using VitMem.")
            print("Will now download required model.")
            print("Model will be located in:", os.path.normpath(model_path))

        ssl._create_default_https_context = ssl._create_unverified_context
        model_state = torch.utils.model_zoo.load_url(
            model_url, map_location="cpu", check_hash=True
        )
        self.model = ViTMem_model()
        self.model.load_state_dict(model_state)
        self.model.eval()
        self.model.to(self._device)

    def get_transformed_image(self, image_object_or_path):
        if isinstance(image_object_or_path, str):
            if os.path.exists(image_object_or_path):
                image_object = Image.open(image_object_or_path).convert("RGB")
            else:
                raise Exception(f"Unable to locate image {image_object_or_path}")
        elif isinstance(image_object_or_path, Image.Image):
            image_object = image_object_or_path
        elif isinstance(image_object_or_path, torch.Tensor):
            image_tensor = image_object_or_path
            if len(image_tensor.shape) == 4:
                return image_tensor
            elif len(image_tensor) == 3:
                return image_tensor.unsqueeze(0)
        else:
            raise Exception(
                "Argument not understood, please supply path to image file or image object"
            )

        image = transform(image_object)
        return image

    def predict(self, image_path: str | list[str]) -> torch.Tensor:
        if isinstance(image_path, str):
            image_path = [image_path]
        transformed_image = torch.stack(
            [self.get_transformed_image(image) for image in image_path]
        )
        transformed_image = transformed_image.to(self._device)
        with torch.no_grad():
            output = self.model(transformed_image)
            return output
