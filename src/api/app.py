import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from tempfile import NamedTemporaryFile
from typing import Iterable, Tuple
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from src.api.config import PROJECT_ROOT, get_settings
from src.api.dependencies import (
    get_feedback_manager,
    get_feedback_prompt_version,
    get_memorability_model,
)
from src.api.schema import ScoreFeedbackResponse, ScoreResponse
from src.models.image_text_to_text.utils.prompt_builder import PromptBuilder
from src.pipelines.method.input.prompts import SYSTEM_PROMPT, USER_PROMPT

settings = get_settings()
LOG_ROOT = PROJECT_ROOT / "outputs/api_requests"
LOG_ROOT.mkdir(parents=True, exist_ok=True)
STATIC_UI_ROOT = PROJECT_ROOT / "web/camera"


@asynccontextmanager
async def _lifespan(app: FastAPI):
    get_memorability_model()
    get_feedback_manager()
    yield


app = FastAPI(title="MemCoach API", version="0.1.0", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_UI_ROOT.exists():
    app.mount(
        "/camera",
        StaticFiles(directory=str(STATIC_UI_ROOT), html=True),
        name="camera",
    )

ALLOWED_CONTENT_TYPES: Tuple[str, ...] = (
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
)


@app.get("/health")
async def healthcheck() -> dict:
    return {"status": "ok"}


async def _persist_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "").suffix
    if not suffix:
        suffix = ".jpg"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(await upload.read())
        return Path(tmp_file.name)


def _cleanup_file(path: Path) -> None:
    if path.exists():
        path.unlink(missing_ok=True)


def _ensure_supported(upload: UploadFile) -> None:
    if upload.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported image content type '{upload.content_type}'",
        )


def _to_float(value) -> float:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().reshape(-1)[0].item())
    except ImportError:
        pass

    if hasattr(value, "item"):
        return float(value.item())
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        first = next(iter(value))
        return _to_float(first)
    return float(value)


def _build_feedback_prompt(image_path: str):
    prompt_version = get_feedback_prompt_version()
    if prompt_version not in USER_PROMPT:
        raise HTTPException(
            status_code=500,
            detail=f"Prompt version '{prompt_version}' is not defined.",
        )
    builder = PromptBuilder(
        {
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": USER_PROMPT[prompt_version],
        }
    )
    return builder.get_prompt([("Image:", image_path)], image_before_text=True)


def _save_request_artifacts(
    image_path: Path,
    endpoint: str,
    score: float,
    original_filename: str | None = None,
    feedback: str | None = None,
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    request_id = uuid.uuid4().hex
    base_name = f"{timestamp.replace(':', '-')}_{request_id}"
    image_dest = LOG_ROOT / f"{base_name}.jpg"
    metadata_dest = LOG_ROOT / f"{base_name}.json"

    with Image.open(image_path) as img:
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        img.convert("RGB").save(image_dest, format="JPEG")

    metadata = {
        "request_id": request_id,
        "timestamp": timestamp,
        "endpoint": endpoint,
        "score": score,
        "feedback": feedback,
        "original_filename": original_filename,
        "saved_image": str(image_dest),
    }

    with open(metadata_dest, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


async def run_memorability(image_path: str) -> float:
    model = get_memorability_model()

    def _predict() -> float:
        score = model.predict(image_path)
        return _to_float(score)

    return await asyncio.to_thread(_predict)


async def run_feedback(image_path: str) -> str:
    manager = get_feedback_manager()
    prompt = _build_feedback_prompt(image_path)

    def _generate() -> str:
        response = manager.generate(prompt)
        if isinstance(response, list):
            return str(response[0])
        return str(response)

    return await asyncio.to_thread(_generate)


@app.post("/score", response_model=ScoreResponse)
async def score(image: UploadFile = File(...)) -> ScoreResponse:
    _ensure_supported(image)
    file_path = await _persist_upload(image)
    try:
        start = time.perf_counter()
        score_value = await run_memorability(str(file_path))
        latency_ms = (time.perf_counter() - start) * 1000
        _save_request_artifacts(
            file_path,
            endpoint="/score",
            score=score_value,
            original_filename=image.filename,
            feedback=None,
        )
        return ScoreResponse(score=score_value, latency_ms=latency_ms)
    finally:
        _cleanup_file(file_path)


@app.post("/score-feedback", response_model=ScoreFeedbackResponse)
async def score_with_feedback(image: UploadFile = File(...)) -> ScoreFeedbackResponse:
    _ensure_supported(image)
    file_path = await _persist_upload(image)
    try:
        score_start = time.perf_counter()
        score_value = await run_memorability(str(file_path))
        score_latency = (time.perf_counter() - score_start) * 1000

        feedback_start = time.perf_counter()
        feedback_text = await run_feedback(str(file_path))
        feedback_latency = (time.perf_counter() - feedback_start) * 1000

        response = ScoreFeedbackResponse(
            score=score_value,
            latency_ms=score_latency,
            feedback=feedback_text,
            feedback_latency_ms=feedback_latency,
        )
        _save_request_artifacts(
            file_path,
            endpoint="/score-feedback",
            score=score_value,
            original_filename=image.filename,
            feedback=feedback_text,
        )
        return response
    finally:
        _cleanup_file(file_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.app:app",
        host=settings.runtime.host,
        port=settings.runtime.port,
        reload=False,
    )
