import torch
from operator import attrgetter
from typing import Union, Sequence
import logging
from src.models.image_text_to_text._hf import HfModel

logger = logging.getLogger(__name__)

_MAPPING_MODEL_ID_TO_LAYERS_PATH = {
    "HuggingFaceM4/Idefics3-8B-Llama3": "model.text_model.layers"
}


class SteeringActivationsExtractor:
    def __init__(self, model: HfModel, target_layers: int, module: str):
        self._model = model
        self._model_id = self._model.model_id
        self._target_layers = list(range(target_layers))
        self._module = module
        self._layers = self._locate_layers()
        self._activations = {}
        self._hooks = []

    def _locate_layers(self):
        layers_path = _MAPPING_MODEL_ID_TO_LAYERS_PATH.get(
            self._model_id, "model.language_model.layers"
        )  # most of the models have the layers under model.language_model.layers
        return attrgetter(layers_path)(self._model.raw_model)

    # ---------- context manager ----------
    def _get_hook(self, layer_idx):
        def extract_hook(module, input, output):
            # The output of a decoder layer is often a tuple (hidden_state, caches, ...)
            # We are interested in the hidden_state, which is the first element.
            activation = output[0] if isinstance(output, tuple) else output
            self._activations[layer_idx] = activation

        return extract_hook

    def __enter__(self):
        for idx in self._target_layers:
            if self._module == "down_proj":
                self._hooks.append(
                    self._layers[idx].mlp.down_proj.register_forward_hook(
                        self._get_hook(idx)
                    )
                )
            if self._module == "residual":
                self._hooks.append(
                    self._layers[idx].register_forward_hook(self._get_hook(idx))
                )
            if self._module == "self_attn":
                self._hooks.append(
                    self._layers[idx].post_attention_layernorm.register_forward_hook(
                        self._get_hook(idx)
                    )
                )
        return self._activations

    def __exit__(self, exc_type, exc_value, traceback):
        for hook in self._hooks:
            hook.remove()


class SteeringActivationsInjector:
    def __init__(
        self,
        model: HfModel,
        steering_vectors: Union[torch.Tensor, Sequence[float]],
        layer_idx: int,
        module: str,
        coeff: float,
    ):
        self._model = model
        self._model_id = self._model.model_id
        self._layer_idx = layer_idx
        self._module = module
        self._steering_vectors = steering_vectors
        self._coeff = coeff
        self._layers = self._locate_layers()

        self._hooks = []

    def _locate_layers(self):
        layers_path = _MAPPING_MODEL_ID_TO_LAYERS_PATH.get(
            self._model_id, "model.language_model.layers"
        )  # most of the models have the layers under model.language_model.layers
        return attrgetter(layers_path)(self._model.raw_model)

    # ---------- context manager ----------
    def get_inject_hook(self, layer_idx):
        def inject_hook(module, input, output):
            activation = output[0] if isinstance(output, tuple) else output
            matched_ruv = self._steering_vectors[layer_idx].repeat(
                1, activation.shape[1], 1
            )
            activation = activation + matched_ruv.to(activation.device) * self._coeff

            if isinstance(output, tuple):
                output = (activation,) + output[1:]
            else:
                output = activation

            return output

        return inject_hook

    def __enter__(self):
        if self._module == "down_proj":
            self._hooks.append(
                self._layers[self._layer_idx].mlp.down_proj.register_forward_hook(
                    self.get_inject_hook(self._layer_idx)
                )
            )
        if self._module == "residual":
            self._hooks.append(
                self._layers[self._layer_idx].register_forward_hook(
                    self.get_inject_hook(self._layer_idx)
                )
            )
        if self._module == "self_attn":
            self._hooks.append(
                self._layers[
                    self._layer_idx
                ].post_attention_layernorm.register_forward_hook(
                    self.get_inject_hook(self._layer_idx)
                )
            )

    def __exit__(self, exc_type, exc_value, traceback):
        for hook in self._hooks:
            hook.remove()
