import torch
import torch.nn as nn
import os
import pickle
import open_clip
from PIL import Image

from src.models.target_predictor._base import BaseTargetPredictor
from src.utils._runtime_paths import resolve_project_root


class OpenCLIPVisualExtractor:
    def __init__(self, model_name, pretrained_model_name):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained_model_name, device=self._device
        )
        self._model.eval()
        self._model.to(self._device)

    def extract_visual_fts(
        self, image_path: str | list[str] | Image.Image, enable_normalization=False
    ):
        if isinstance(image_path, str):
            image_path = [Image.open(image_path)]
        elif isinstance(image_path, list):
            image_path = [Image.open(image) for image in image_path]
        elif isinstance(image_path, Image.Image):
            image_path = [image_path]
        images = torch.stack([self._preprocess(image) for image in image_path])
        with (
            torch.no_grad(),
            torch.autocast(device_type=self._device, dtype=torch.float32),
        ):
            image_features = self._model.encode_image(images.to(self._device))
            if enable_normalization:
                image_features /= image_features.norm(dim=1, keepdim=True)

        return image_features

    def save_visual_fts(
        self, batch_number, image_paths, image_features, output_path, recursive
    ):
        os.makedirs(output_path, exist_ok=True)
        features = {}
        if recursive:
            for i, path in enumerate(image_paths):
                features[os.path.sep.join(path.split(os.path.sep)[-3:])] = (
                    image_features[i].cpu().numpy()
                )
        else:
            for i, path in enumerate(image_paths):
                features[os.path.basename(path)] = image_features[i].cpu().numpy()
        with open(os.path.join(output_path, f"{batch_number}.pkl"), "wb") as f:
            pickle.dump(features, f)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze()


class MemBenchMemorabilityPredictor(BaseTargetPredictor):
    def __init__(self, config: dict):
        super().__init__(config)
        ckpt = torch.load(
            os.path.join(
                resolve_project_root(), config["model"]["mlp_checkpoint_path"]
            ),
            map_location=self._device,
        )
        self._backbone = OpenCLIPVisualExtractor(
            ckpt.get("backbone_model"), ckpt.get("pretrained_backbone_model")
        )
        self._regressor = MLP(
            input_dim=ckpt.get("input_dim"),
            hidden_dim=ckpt.get("hidden_dim"),
            output_dim=1,
        )
        self._regressor.load_state_dict(ckpt["model"])
        self._regressor.eval()
        self._regressor.to(self._device)
        self._normalize_fts = config.get("normalize_fts", False)

    def predict(self, images) -> torch.Tensor:
        visual_fts = self._backbone.extract_visual_fts(
            images, enable_normalization=self._normalize_fts
        )
        output = self._regressor(visual_fts)
        if output.dim() == 0:
            return output.unsqueeze(0)
        return output.clamp(0, 1)
