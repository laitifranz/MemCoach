import torch

from src.models.image_text_to_text.utils.prompt_builder import PromptBuilder

# Model-specific tokens for locating assistant reply span in input_ids.
ASSISTANT_TOKEN_MAPPING = {
    "HuggingFaceM4/Idefics3-8B-Llama3": "Assistant",
    "unsloth/gemma-3-12b-it": "model",
    "Anjoe/AesMMIT_LLaVA_v1.5_7b_240325-hf": "ass",
}
EOS_TOKEN_MAPPING = {
    "HuggingFaceM4/Idefics3-8B-Llama3": "<end_of_utterance>",
}


def _get_assistant_token_id(model_id: str, tokenizer) -> int:
    token = ASSISTANT_TOKEN_MAPPING.get(model_id, "assistant")
    return tokenizer.convert_tokens_to_ids(token)


def _get_eos_token_id(model_id: str, tokenizer) -> int:
    if model_id in EOS_TOKEN_MAPPING:
        return tokenizer.convert_tokens_to_ids(EOS_TOKEN_MAPPING[model_id])
    return tokenizer.eos_token_id


def _build_labels(
    input_ids: torch.Tensor,
    assistant_token_id: int,
    eos_token_id: int,
) -> torch.Tensor:
    bool_mask = input_ids == assistant_token_id
    int_mask = bool_mask.int()  # argmax is not working on bool
    idx = torch.argmax(int_mask, dim=-1)
    labels = input_ids.clone()
    labels[0, : idx + 1] = -100

    bool_mask = labels == eos_token_id
    int_mask = bool_mask.int()
    idx = torch.argmax(int_mask, dim=-1)
    labels[0, idx:] = -100
    return labels


def prepare_perplexity_batch(model, prompt):
    """
    Tokenize prompt and build labels for perplexity. Returns (batch, labels) ready
    to pass to raw_model(**(batch.to(device)), labels=labels.to(device)).
    """
    processor = model.raw_processor
    model_id = model.model_id
    messages = PromptBuilder.get_message_variable([prompt], unwrap_image=True)
    batch = processor.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        padding=True,
        return_tensors="pt",
    )
    input_ids = batch["input_ids"]
    assistant_token_id = _get_assistant_token_id(model_id, processor.tokenizer)
    eos_token_id = _get_eos_token_id(model_id, processor.tokenizer)
    labels = _build_labels(input_ids, assistant_token_id, eos_token_id)
    return batch, labels


def compute_perplexity(model, prompt) -> float:
    raw_model = model.raw_model
    device = next(raw_model.parameters()).device
    dtype = next(raw_model.parameters()).dtype

    batch, labels = prepare_perplexity_batch(model, prompt)
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }
    labels = labels.to(device)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
        out = raw_model(**batch, labels=labels)
    return out.loss.exp().item()
