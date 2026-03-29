"""OpenAI-compatible compatibility routes for external clients.

These endpoints provide a thin facade over Voicebox's native API so
existing OpenAI-style clients can work without code changes.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from sqlalchemy.orm import Session
from starlette.datastructures import FormData

from .. import models
from ..backends import (
    WHISPER_HF_REPOS,
    check_model_loaded,
    engine_needs_trim,
    ensure_model_cached_or_raise,
    get_model_config,
    get_model_load_func,
    get_tts_backend_for_engine,
    get_tts_model_configs,
    load_engine_model,
)
from ..database import VoiceProfile as DBVoiceProfile, get_db
from ..services import profiles as profile_service
from ..services.task_queue import create_background_task
from ..utils.audio import load_audio, normalize_audio, save_audio, trim_tts_output
from ..utils.chunked_tts import generate_chunked
from ..utils.platform_detect import get_backend_type
from ..utils.tasks import get_task_manager

logger = logging.getLogger(__name__)

router = APIRouter()

OPENAI_VOICE_ALIASES = {"alloy", "echo", "fable", "onyx", "nova", "shimmer"}
LANGUAGE_NAME_TO_CODE = {
    "auto": None,
    "english": "en",
    "chinese": "zh",
    "japanese": "ja",
    "korean": "ko",
    "german": "de",
    "french": "fr",
    "russian": "ru",
    "portuguese": "pt",
    "spanish": "es",
    "italian": "it",
    "hebrew": "he",
    "arabic": "ar",
    "danish": "da",
    "greek": "el",
    "finnish": "fi",
    "hindi": "hi",
    "malay": "ms",
    "dutch": "nl",
    "norwegian": "no",
    "polish": "pl",
    "swedish": "sv",
    "swahili": "sw",
    "turkish": "tr",
}
LANGUAGE_CODE_TO_NAME = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
    "he": "Hebrew",
    "ar": "Arabic",
    "da": "Danish",
    "el": "Greek",
    "fi": "Finnish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "sv": "Swedish",
    "sw": "Swahili",
    "tr": "Turkish",
}
SUPPORTED_AUDIO_FORMATS = {"wav", "mp3", "flac", "opus", "aac"}
FORMAT_TO_MIME = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "opus": "audio/ogg",
    "aac": "audio/aac",
}
MODEL_ALIASES = {
    "tts-1": "qwen-tts-1.7B",
    "tts-1-hd": "qwen-tts-1.7B",
    "gpt-4o-mini-tts": "qwen-tts-1.7B",
    "xtts-v2": "qwen-tts-1.7B",
}
MODEL_LOAD_STATE: dict[str, dict[str, str]] = {"tts_base": {"status": "idle"}}


def _is_upload_file(value: Any) -> bool:
    return hasattr(value, "filename") and hasattr(value, "read")


def _get_form_upload(form: FormData, *field_names: str) -> UploadFile | None:
    for key, value in form.multi_items():
        if key in field_names and _is_upload_file(value):
            return value

    for field_name in field_names:
        value = form.get(field_name)
        if _is_upload_file(value):
            return value

    return None


def _normalise_language(value: str | None, fallback: str | None = None) -> str | None:
    if value is None:
        return fallback
    candidate = value.strip()
    if not candidate:
        return fallback
    lowered = candidate.lower()
    if lowered in LANGUAGE_NAME_TO_CODE:
        return LANGUAGE_NAME_TO_CODE[lowered] or fallback
    if lowered in LANGUAGE_CODE_TO_NAME:
        return lowered
    return fallback


def _language_display_name(code: str | None) -> str:
    if not code:
        return "Auto"
    return LANGUAGE_CODE_TO_NAME.get(code, code)


def _normalise_response_format(value: Any) -> str:
    candidate = str(value or "mp3").strip().lower()
    if candidate not in SUPPORTED_AUDIO_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported response_format '{candidate}'. Must be one of: {', '.join(sorted(SUPPORTED_AUDIO_FORMATS))}",
        )
    return candidate


def _ffmpeg_binary() -> str:
    binary = os.environ.get("VOICEBOX_FFMPEG_BINARY", "ffmpeg")
    if shutil.which(binary):
        return binary
    if Path(binary).exists():
        return binary
    raise HTTPException(status_code=500, detail="ffmpeg binary not found; Opus/MP3/AAC output is unavailable")


def _encode_audio_bytes(audio, sample_rate: int, response_format: str) -> tuple[bytes, str, str]:
    if response_format in {"wav", "flac"}:
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format=response_format.upper())
        buffer.seek(0)
        ext = "ogg" if response_format == "opus" else response_format
        return buffer.read(), FORMAT_TO_MIME[response_format], f"speech.{ext}"

    ffmpeg = _ffmpeg_binary()
    output_suffix = {"mp3": ".mp3", "aac": ".aac", "opus": ".ogg"}[response_format]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as input_file:
        input_path = Path(input_file.name)
    with tempfile.NamedTemporaryFile(suffix=output_suffix, delete=False) as output_file:
        output_path = Path(output_file.name)

    try:
        sf.write(str(input_path), audio, sample_rate, format="WAV")
        command = [ffmpeg, "-y", "-loglevel", "error", "-i", str(input_path)]
        if response_format == "mp3":
            command += ["-c:a", "libmp3lame", "-b:a", "192k", str(output_path)]
        elif response_format == "aac":
            command += ["-c:a", "aac", "-b:a", "192k", "-f", "adts", str(output_path)]
        else:
            command += [
                "-c:a",
                "libopus",
                "-b:a",
                "48k",
                "-vbr",
                "on",
                "-application",
                "voip",
                str(output_path),
            ]

        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=result.stderr.strip() or f"ffmpeg failed for {response_format}")

        filename = "speech.ogg" if response_format == "opus" else f"speech.{response_format}"
        return output_path.read_bytes(), FORMAT_TO_MIME[response_format], filename
    finally:
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)


def _voice_payload(profile: DBVoiceProfile) -> dict[str, Any]:
    voice_type = getattr(profile, "voice_type", None) or "cloned"
    return {
        "id": profile.id,
        "object": "voice",
        "name": profile.name,
        "language": _language_display_name(profile.language),
        "voice_type": voice_type,
        "engine": getattr(profile, "default_engine", None) or getattr(profile, "preset_engine", None) or "qwen",
        "created_at": profile.created_at.isoformat() if getattr(profile, "created_at", None) else None,
    }


def _resolve_profile(db: Session, voice: str | None = None, voice_id: str | None = None) -> DBVoiceProfile | None:
    if voice_id:
        profile = db.query(DBVoiceProfile).filter_by(id=voice_id).first()
        if profile:
            return profile

    if voice:
        profile = db.query(DBVoiceProfile).filter_by(id=voice).first()
        if profile:
            return profile

        lowered = voice.strip().lower()
        for profile in db.query(DBVoiceProfile).order_by(DBVoiceProfile.created_at.desc()).all():
            if profile.name.lower() == lowered:
                return profile

        if lowered in OPENAI_VOICE_ALIASES:
            fallback_profiles = db.query(DBVoiceProfile).order_by(DBVoiceProfile.created_at.desc()).all()
            if fallback_profiles:
                return fallback_profiles[0]

    profiles = db.query(DBVoiceProfile).order_by(DBVoiceProfile.created_at.desc()).all()
    if len(profiles) == 1:
        return profiles[0]
    return None


def _resolve_model_alias(model: str | None) -> str:
    if not model:
        return MODEL_ALIASES["tts-1"]
    return MODEL_ALIASES.get(model, model)


def _resolve_engine_and_size(profile: DBVoiceProfile | None, requested_model: str | None) -> tuple[str, str]:
    resolved = _resolve_model_alias(requested_model)
    config = get_model_config(resolved)
    if config:
        return config.engine, config.model_size or "default"

    if profile is not None:
        engine = getattr(profile, "default_engine", None) or getattr(profile, "preset_engine", None) or "qwen"
    else:
        engine = "qwen"
    return engine, "1.7B" if engine == "qwen" else "default"


async def _synthesise_from_profile(
    db: Session,
    profile: DBVoiceProfile,
    text: str,
    language: str | None,
    response_format: str,
    requested_model: str | None,
    instructions: str | None,
) -> tuple[bytes, str, str]:
    engine, model_size = _resolve_engine_and_size(profile, requested_model)
    await ensure_model_cached_or_raise(engine, model_size)
    await load_engine_model(engine, model_size)

    voice_prompt = await profile_service.create_voice_prompt_for_profile(profile.id, db, engine=engine)
    tts_model = get_tts_backend_for_engine(engine)
    trim_fn = trim_tts_output if engine_needs_trim(engine) else None

    audio, sample_rate = await generate_chunked(
        tts_model,
        text,
        voice_prompt,
        language=language or profile.language or "en",
        instruct=instructions,
        trim_fn=trim_fn,
    )
    return _encode_audio_bytes(audio, sample_rate, response_format)


async def _synthesise_from_upload(
    audio_upload: UploadFile,
    reference_text: str,
    text: str,
    language: str | None,
    response_format: str,
    requested_model: str | None,
    instructions: str | None,
) -> tuple[bytes, str, str]:
    engine, model_size = _resolve_engine_and_size(None, requested_model)
    if engine == "kokoro":
        engine = "qwen"
        model_size = "1.7B"

    await ensure_model_cached_or_raise(engine, model_size)
    await load_engine_model(engine, model_size)

    suffix = Path(audio_upload.filename or "reference.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_path = Path(temp_file.name)
        while chunk := await audio_upload.read(1024 * 1024):
            temp_file.write(chunk)

    try:
        tts_model = get_tts_backend_for_engine(engine)
        voice_prompt, _ = await tts_model.create_voice_prompt(str(temp_path), reference_text, use_cache=False)
        trim_fn = trim_tts_output if engine_needs_trim(engine) else None
        audio, sample_rate = await generate_chunked(
            tts_model,
            text,
            voice_prompt,
            language=language or "en",
            instruct=instructions,
            trim_fn=trim_fn,
        )
        return _encode_audio_bytes(audio, sample_rate, response_format)
    finally:
        temp_path.unlink(missing_ok=True)


async def _parse_speech_request(request: Request) -> tuple[dict[str, Any], UploadFile | None]:
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type:
        form = await request.form()
        payload: dict[str, Any] = {}
        upload = _get_form_upload(form, "audio_sample")
        for key, value in form.multi_items():
            if _is_upload_file(value):
                continue
            payload[key] = value
        return payload, upload

    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")
    return payload, None


def _tts_base_model_name() -> str:
    return MODEL_ALIASES["tts-1"]


def _compat_catalog_entry(slug: str, status: str) -> dict[str, Any]:
    return {
        "id": "tts_models/multilingual/multi-dataset/xtts_v2",
        "slug": slug,
        "type": "Voicebox compatibility model",
        "size": "managed",
        "device": "CPU/GPU",
        "description": "Compatibility alias for OpenAI-style clients backed by Voicebox.",
        "api_type": "tts_base",
        "container_path": f"voicebox:{_tts_base_model_name()}",
        "state": {"status": status},
    }


async def _prepare_reference_audio_path(upload: UploadFile) -> tuple[Path, list[Path]]:
    suffix = Path(upload.filename or "reference.wav").suffix or ".wav"
    cleanup_paths: list[Path] = []

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_path = Path(temp_file.name)
        cleanup_paths.append(temp_path)
        while chunk := await upload.read(1024 * 1024):
            temp_file.write(chunk)

    try:
        audio, sample_rate = await asyncio.to_thread(load_audio, str(temp_path))
    except Exception as exc:
        logger.warning("Skipping pre-normalisation for reference audio: %s", exc)
        return temp_path, cleanup_paths

    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak <= 0.99:
        return temp_path, cleanup_paths

    normalized_audio = normalize_audio(audio, peak_limit=0.9)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as normalized_file:
        normalized_path = Path(normalized_file.name)
    cleanup_paths.append(normalized_path)
    await asyncio.to_thread(save_audio, normalized_audio, str(normalized_path), sample_rate)
    logger.info("Normalised clipped reference audio for OpenAI-compatible voice save")
    return normalized_path, cleanup_paths


@router.get("/v1/voices")
async def list_openai_voices(db: Session = Depends(get_db)):
    profiles = db.query(DBVoiceProfile).order_by(DBVoiceProfile.created_at.desc()).all()
    return {"object": "list", "data": [_voice_payload(profile) for profile in profiles]}


@router.get("/v1/audio/voices")
async def list_openai_audio_voices(db: Session = Depends(get_db)):
    return await list_openai_voices(db)


@router.post("/v1/voices")
async def create_openai_voice(request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    upload = _get_form_upload(form, "audio_sample", "file")
    name = str(form.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Field 'name' is required")
    if upload is None:
        raise HTTPException(status_code=400, detail="Field 'audio_sample' is required")

    reference_text = str(form.get("audio_sample_text") or "").strip()
    if not reference_text:
        raise HTTPException(status_code=400, detail="Field 'audio_sample_text' is required for Voicebox cloning")

    language = _normalise_language(str(form.get("language") or "Auto"), fallback="en") or "en"
    profile_data = models.VoiceProfileCreate(
        name=name,
        description="Imported from OpenAI-compatible voice endpoint",
        language=language,
        default_engine="qwen",
    )
    try:
        profile = await profile_service.create_profile(profile_data, db)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    temp_path, cleanup_paths = await _prepare_reference_audio_path(upload)

    try:
        await profile_service.add_profile_sample(profile.id, str(temp_path), reference_text, db)
    except ValueError as exc:
        await profile_service.delete_profile(profile.id, db)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        await profile_service.delete_profile(profile.id, db)
        raise
    finally:
        for path in cleanup_paths:
            path.unlink(missing_ok=True)

    created = db.query(DBVoiceProfile).filter_by(id=profile.id).first() or profile
    return JSONResponse(_voice_payload(created), status_code=200)


@router.delete("/v1/voices/{voice_id}")
async def delete_openai_voice(voice_id: str, db: Session = Depends(get_db)):
    deleted = await profile_service.delete_profile(voice_id, db)
    if not deleted:
        raise HTTPException(status_code=404, detail="Voice not found")
    return {"id": voice_id, "deleted": True}


@router.post("/v1/audio/speech")
async def openai_audio_speech(request: Request, db: Session = Depends(get_db)):
    payload, upload = await _parse_speech_request(request)
    text = str(payload.get("input") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Field 'input' is required")

    response_format = _normalise_response_format(payload.get("response_format"))
    language = _normalise_language(payload.get("language"))
    instructions = str(payload.get("instructions") or payload.get("instruct") or "").strip() or None
    requested_model = str(payload.get("model") or "").strip() or None

    if upload is not None:
        reference_text = str(payload.get("audio_sample_text") or "").strip()
        if not reference_text:
            raise HTTPException(status_code=400, detail="Field 'audio_sample_text' is required when sending audio_sample")
        audio_bytes, mime_type, filename = await _synthesise_from_upload(
            upload,
            reference_text,
            text,
            language,
            response_format,
            requested_model,
            instructions,
        )
    else:
        voice = str(payload.get("voice") or "").strip() or None
        voice_id = str(payload.get("voice_id") or "").strip() or None
        profile = _resolve_profile(db, voice=voice, voice_id=voice_id)
        if profile is None:
            raise HTTPException(status_code=404, detail="No compatible voice/profile found for this request")
        audio_bytes, mime_type, filename = await _synthesise_from_profile(
            db,
            profile,
            text,
            language,
            response_format,
            requested_model,
            instructions,
        )

    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(content=audio_bytes, media_type=mime_type, headers=headers)


@router.post("/v1/audio/transcriptions")
async def openai_audio_transcriptions(request: Request):
    form = await request.form()
    upload = form.get("file")
    if not isinstance(upload, UploadFile):
        raise HTTPException(status_code=400, detail="Field 'file' is required")

    language = _normalise_language(str(form.get("language") or ""))
    requested_model = str(form.get("model") or "").strip().lower()
    model_size = requested_model if requested_model in WHISPER_HF_REPOS else None

    with tempfile.NamedTemporaryFile(suffix=Path(upload.filename or "audio.wav").suffix or ".wav", delete=False) as temp_file:
        temp_path = Path(temp_file.name)
        while chunk := await upload.read(1024 * 1024):
            temp_file.write(chunk)

    try:
        from ..services import transcribe as transcribe_service

        audio, sample_rate = await asyncio.to_thread(load_audio, str(temp_path))
        duration = len(audio) / sample_rate

        whisper_model = transcribe_service.get_whisper_model()
        current_model_size = getattr(whisper_model, "model_size", None)
        effective_model_size = model_size or current_model_size
        if effective_model_size not in WHISPER_HF_REPOS:
            effective_model_size = "base"

        is_model_cached = getattr(whisper_model, "_is_model_cached", None)
        already_loaded = whisper_model.is_loaded() and current_model_size == effective_model_size
        if not already_loaded and callable(is_model_cached) and not is_model_cached(effective_model_size):
            raise HTTPException(
                status_code=202,
                detail={
                    "message": f"Whisper model {effective_model_size} is being downloaded. Please retry shortly.",
                    "model_name": f"whisper-{effective_model_size}",
                    "downloading": True,
                },
            )

        text = await whisper_model.transcribe(str(temp_path), language, effective_model_size)
        return {"text": text, "duration": duration}
    finally:
        temp_path.unlink(missing_ok=True)


@router.get("/v1/models")
@router.get("/v1/audio/models")
async def list_openai_models():
    data = []
    for config in get_tts_model_configs():
        data.append(
            {
                "id": config.model_name,
                "object": "model",
                "created": 0,
                "owned_by": "voicebox",
                "engine": config.engine,
                "display_name": config.display_name,
            }
        )
    return {"object": "list", "data": data}


@router.get("/v1/models/catalog")
async def openai_model_catalog():
    catalog = []
    base_model_name = _tts_base_model_name()
    base_config = get_model_config(base_model_name)
    alias_status = "unknown"
    if base_config:
        alias_status = "active" if check_model_loaded(base_config) else "downloaded"
    catalog.append(_compat_catalog_entry("xtts-v2", alias_status))

    for config in get_tts_model_configs():
        status = "active" if check_model_loaded(config) else "downloaded"
        catalog.append(
            {
                "id": config.model_name,
                "slug": config.model_name,
                "type": config.display_name,
                "size": config.model_size,
                "device": "CPU/GPU",
                "description": f"Voicebox model for engine {config.engine}",
                "api_type": "tts_base",
                "container_path": f"voicebox:{config.model_name}",
                "state": {"status": status},
            }
        )
    return {"object": "list", "data": catalog}


@router.get("/v1/models/download/status")
async def openai_model_download_status():
    task_manager = get_task_manager()
    active = {task.model_name: task.status for task in task_manager.get_active_downloads()}
    data = {slug: {"status": status} for slug, status in active.items()}
    if _tts_base_model_name() in active:
        data["xtts-v2"] = {"status": active[_tts_base_model_name()]}
    return {"data": data}


@router.get("/v1/models/runtime/status")
async def openai_model_runtime_status():
    from ..app import _get_gpu_status

    loaded = [config.model_name for config in get_tts_model_configs() if check_model_loaded(config)]
    return {
        "data": {
            "backend": get_backend_type(),
            "gpu": _get_gpu_status(),
            "loaded_models": loaded,
            "status": "loaded" if loaded else "idle",
        }
    }


@router.post("/v1/models/download")
async def openai_model_download(request: Request):
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be an object")

    slug = str(payload.get("slug") or "").strip()
    if not slug:
        raise HTTPException(status_code=400, detail="Field 'slug' is required")

    model_name = _resolve_model_alias(slug)
    config = get_model_config(model_name)
    if not config:
        raise HTTPException(status_code=404, detail=f"Unknown model slug: {slug}")

    task_manager = get_task_manager()
    task_manager.start_download(model_name)
    load_func = get_model_load_func(config)

    async def _download_task():
        try:
            result = load_func()
            if asyncio.iscoroutine(result):
                await result
            task_manager.complete_download(model_name)
        except Exception as exc:
            logger.exception("Model download failed for %s: %s", model_name, exc)
            task_manager.error_download(model_name, str(exc))

    create_background_task(_download_task())
    return {"status": "downloading", "slug": slug, "model_name": model_name}


@router.post("/v1/models/load")
async def openai_model_load(request: Request):
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be an object")

    path_hint = str(payload.get("path") or "").strip()
    model_name = path_hint.split(":", 1)[1] if path_hint.startswith("voicebox:") else _tts_base_model_name()
    model_name = _resolve_model_alias(model_name)
    config = get_model_config(model_name)
    if not config:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}")

    MODEL_LOAD_STATE["tts_base"] = {"status": "loading"}
    load_func = get_model_load_func(config)

    async def _load_task():
        try:
            result = load_func()
            if asyncio.iscoroutine(result):
                await result
            MODEL_LOAD_STATE["tts_base"] = {"status": "loaded", "model_name": model_name}
        except Exception as exc:
            logger.exception("Model load failed for %s: %s", model_name, exc)
            MODEL_LOAD_STATE["tts_base"] = {"status": "error", "detail": str(exc)}

    create_background_task(_load_task())
    return {"status": "loading", "model_name": model_name}


@router.get("/v1/models/load/status")
async def openai_model_load_status():
    return MODEL_LOAD_STATE