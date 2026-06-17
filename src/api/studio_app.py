import asyncio
import base64
import copy
import io
import json
import logging
import sqlite3
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional, Tuple

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from omegaconf import OmegaConf
from PIL import Image
from pydantic import BaseModel

from src.api.config import PROJECT_ROOT
from src.models.image_text_to_image._flux_klein import Flux2Klein
from src.models.target_predictor.model_factory import get_model
from src.models.image_text_to_text.utils.prompt_builder import PromptBuilder
from src.pipelines.method.core.inference_manager import InferenceManager
from src.pipelines.method.input.prompts import SYSTEM_PROMPT, USER_PROMPT


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class _RuntimeConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8001


class _FluxConfig(BaseModel):
    name: str = "black-forest-labs/FLUX.2-klein-9B"


class _MemorabilityConfig(BaseModel):
    provider: str
    mlp_checkpoint_path: str


class _FeedbackConfig(BaseModel):
    hydra_config: str
    coeff_override: Optional[float] = None
    target_layer_override: Optional[int] = None


class _DemoSettings(BaseModel):
    runtime: _RuntimeConfig
    flux: _FluxConfig
    memorability: _MemorabilityConfig
    feedback: _FeedbackConfig


@lru_cache
def _get_settings() -> _DemoSettings:
    config_path = PROJECT_ROOT / "config/api/studio_server.yaml"
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    return _DemoSettings(**cfg)


def _resolve(path_str: str) -> str:
    p = Path(path_str)
    return str(p if p.is_absolute() else PROJECT_ROOT / p)


# ---------------------------------------------------------------------------
# Model singletons
# ---------------------------------------------------------------------------

@lru_cache
def _get_memorability_model():
    settings = _get_settings()
    base = settings.memorability.model_dump()
    base["mlp_checkpoint_path"] = _resolve(base["mlp_checkpoint_path"])
    return get_model({"model": base})


@lru_cache
def _get_feedback_manager() -> InferenceManager:
    settings = _get_settings()
    cfg_path = _resolve(settings.feedback.hydra_config)
    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)

    act = cfg.get("activation_settings", {})
    for key in ("negative_actvs_file", "positive_actvs_file"):
        if key in act:
            act[key] = _resolve(act[key])

    if settings.feedback.coeff_override is not None:
        act["coeff"] = settings.feedback.coeff_override
    if settings.feedback.target_layer_override is not None:
        act["target_layer"] = settings.feedback.target_layer_override

    cfg["activation_settings"] = act
    return InferenceManager(copy.deepcopy(cfg))


@lru_cache
def _get_flux_model() -> Flux2Klein:
    settings = _get_settings()
    return Flux2Klein({"name": settings.flux.name})


# ---------------------------------------------------------------------------
# Leaderboard (SQLite)
# ---------------------------------------------------------------------------

LOG_ROOT = PROJECT_ROOT / "outputs" / "studio_requests"
_DB_PATH = PROJECT_ROOT / "outputs" / "studio_leaderboard.db"
_db_lock = threading.Lock()
_db_conn: Optional[sqlite3.Connection] = None

# ---------------------------------------------------------------------------
# Session tracker (in-memory, for "connected users" count)
# ---------------------------------------------------------------------------

_SESSION_TIMEOUT = 45.0  # seconds — 3 missed heartbeats at 15 s interval
_sessions: dict = {}
_sessions_lock = threading.Lock()


def _update_session(session_id: str) -> int:
    now = time.time()
    with _sessions_lock:
        _sessions[session_id] = now
        stale = [sid for sid, ts in _sessions.items() if now - ts > _SESSION_TIMEOUT]
        for sid in stale:
            del _sessions[sid]
        return len(_sessions)


def _count_active_sessions() -> int:
    now = time.time()
    with _sessions_lock:
        return sum(1 for ts in _sessions.values() if now - ts <= _SESSION_TIMEOUT)


def _init_db() -> None:
    global _db_conn
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    _db_conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    _db_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS leaderboard (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            score_before REAL NOT NULL,
            score_after REAL NOT NULL,
            improvement_pct REAL NOT NULL
        )
        """
    )
    _db_conn.commit()


def _insert_leaderboard(score_before: float, score_after: float) -> None:
    if _db_conn is None:
        return
    improvement_pct = ((score_after - score_before) / max(score_before, 1e-6)) * 100
    ts = datetime.now(timezone.utc).isoformat()
    with _db_lock:
        _db_conn.execute(
            "INSERT INTO leaderboard (timestamp, score_before, score_after, improvement_pct) VALUES (?, ?, ?, ?)",
            (ts, score_before, score_after, improvement_pct),
        )
        _db_conn.commit()


def _query_leaderboard(limit: int = 20) -> Tuple[list, int]:
    if _db_conn is None:
        return [], 0
    with _db_lock:
        rows = _db_conn.execute(
            "SELECT timestamp, score_before, score_after, improvement_pct "
            "FROM leaderboard ORDER BY improvement_pct DESC LIMIT ?",
            (limit,),
        ).fetchall()
        total = _db_conn.execute("SELECT COUNT(*) FROM leaderboard").fetchone()[0]
    return rows, total


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI):
    logging.info("Demo: loading memorability model...")
    _get_memorability_model()
    logging.info("Demo: loading feedback manager...")
    _get_feedback_manager()
    logging.info("Demo: loading FLUX.2-klein model...")
    _get_flux_model()
    logging.info("Demo: initialising leaderboard DB...")
    _init_db()
    logging.info("Demo: all models ready.")
    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="MemCoach Before/After Studio", version="1.0.0", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_UI_ROOT = PROJECT_ROOT / "web/studio"
if STATIC_UI_ROOT.exists():
    app.mount("/studio", StaticFiles(directory=str(STATIC_UI_ROOT), html=True), name="studio")

ALLOWED_CONTENT_TYPES: Tuple[str, ...] = (
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class EditResponse(BaseModel):
    edited_image: str
    feedback: str
    score_before: float
    score_after: float
    latency_ms: float


class LeaderboardEntry(BaseModel):
    rank: int
    score_before: float
    score_after: float
    improvement_pct: float
    timestamp: str


class LeaderboardResponse(BaseModel):
    entries: list[LeaderboardEntry]
    total_edits: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_supported(upload: UploadFile) -> None:
    if upload.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported image content type '{upload.content_type}'",
        )


async def _persist_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "").suffix or ".jpg"
    with NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(await upload.read())
        return Path(f.name)


def _cleanup(*paths: Path) -> None:
    for p in paths:
        p.unlink(missing_ok=True)


def _to_float(value) -> float:
    try:
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().reshape(-1)[0].item())
    except Exception:
        pass
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _build_feedback_prompt(image_path: str):
    version = "inference_prompt"
    builder = PromptBuilder(
        {"system_prompt": SYSTEM_PROMPT, "user_prompt": USER_PROMPT[version]}
    )
    return builder.get_prompt([("Image:", image_path)], image_before_text=True)


def _pil_to_b64_jpeg(img: Image.Image, quality: int = 75) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _save_pil_to_tmp(img: Image.Image) -> Path:
    with NamedTemporaryFile(delete=False, suffix=".jpg") as f:
        path = Path(f.name)
    img.convert("RGB").save(str(path), format="JPEG", quality=95)
    return path


def _save_request_artifacts(
    src_path: Path,
    edited_pil: Image.Image,
    feedback: str,
    score_before: float,
    score_after: float,
    latency_ms: float,
    original_filename: Optional[str] = None,
) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    request_id = uuid.uuid4().hex
    base_name = f"{timestamp.replace(':', '-')}_{request_id}"

    src_dest    = LOG_ROOT / f"{base_name}_source.jpg"
    edited_dest = LOG_ROOT / f"{base_name}_edited.jpg"
    meta_dest   = LOG_ROOT / f"{base_name}.json"

    with Image.open(src_path) as img:
        img.convert("RGB").save(src_dest, format="JPEG")
    edited_pil.convert("RGB").save(edited_dest, format="JPEG")

    metadata = {
        "request_id": request_id,
        "timestamp": timestamp,
        "score_before": score_before,
        "score_after": score_after,
        "improvement_pct": round(((score_after - score_before) / max(score_before, 1e-6)) * 100, 2),
        "latency_ms": latency_ms,
        "feedback": feedback,
        "original_filename": original_filename,
        "source_image": str(src_dest),
        "edited_image": str(edited_dest),
    }
    with open(meta_dest, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def healthcheck(session: Optional[str] = None) -> dict:
    if session:
        connected = await asyncio.to_thread(_update_session, session)
    else:
        connected = _count_active_sessions()
    return {"status": "ok", "connected": connected}


@app.post("/edit", response_model=EditResponse)
async def edit(image: UploadFile = File(...)) -> EditResponse:
    _ensure_supported(image)
    src_path = await _persist_upload(image)
    edited_path: Optional[Path] = None

    try:
        wall_start = time.perf_counter()

        # Score source image
        mem_model = _get_memorability_model()
        score_before = _to_float(
            await asyncio.to_thread(mem_model.predict, str(src_path))
        )

        # Generate memorability-improving feedback
        manager = _get_feedback_manager()
        prompt = _build_feedback_prompt(str(src_path))

        def _generate_feedback() -> str:
            resp = manager.generate(prompt)
            return str(resp[0]) if isinstance(resp, list) else str(resp)

        feedback_text = await asyncio.to_thread(_generate_feedback)

        # Edit image with FLUX.2-klein
        flux = _get_flux_model()

        def _run_flux() -> Image.Image:
            return flux.generate(feedback_text, str(src_path))

        edited_pil = await asyncio.to_thread(_run_flux)

        # Score edited image
        edited_path = await asyncio.to_thread(_save_pil_to_tmp, edited_pil)
        score_after = _to_float(
            await asyncio.to_thread(mem_model.predict, str(edited_path))
        )

        latency_ms = (time.perf_counter() - wall_start) * 1000

        # Persist to leaderboard
        await asyncio.to_thread(_insert_leaderboard, score_before, score_after)

        # Log source + edited images and metadata to disk
        await asyncio.to_thread(
            _save_request_artifacts,
            src_path, edited_pil, feedback_text,
            score_before, score_after, round(latency_ms, 1),
            image.filename,
        )

        edited_b64 = await asyncio.to_thread(_pil_to_b64_jpeg, edited_pil)

        return EditResponse(
            edited_image=edited_b64,
            feedback=feedback_text,
            score_before=round(score_before, 4),
            score_after=round(score_after, 4),
            latency_ms=round(latency_ms, 1),
        )

    finally:
        _cleanup(src_path)
        if edited_path is not None:
            _cleanup(edited_path)


@app.get("/leaderboard", response_model=LeaderboardResponse)
async def leaderboard() -> LeaderboardResponse:
    rows, total = await asyncio.to_thread(_query_leaderboard, 20)
    entries = [
        LeaderboardEntry(
            rank=i + 1,
            score_before=round(r[1], 4),
            score_after=round(r[2], 4),
            improvement_pct=round(r[3], 1),
            timestamp=r[0],
        )
        for i, r in enumerate(rows)
    ]
    return LeaderboardResponse(entries=entries, total_edits=total)


if __name__ == "__main__":
    import uvicorn

    s = _get_settings()
    uvicorn.run(
        "src.api.studio_app:app",
        host=s.runtime.host,
        port=s.runtime.port,
        reload=False,
    )
