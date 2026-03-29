from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker

from backend.database import Base, get_db
from backend.database.models import VoiceProfile
from backend.routes import openai_compat


@pytest.fixture()
def client(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    app = FastAPI()
    app.include_router(openai_compat.router)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    async def fake_synthesise_from_profile(db, profile, text, language, response_format, requested_model, instructions):
        return (
            b"opus-bytes",
            "audio/ogg",
            "speech.ogg",
        )

    monkeypatch.setattr(
        openai_compat,
        "_synthesise_from_profile",
        fake_synthesise_from_profile,
    )
    return TestClient(app), TestingSessionLocal


def test_list_openai_voices_returns_profiles(client):
    test_client, session_factory = client
    session = session_factory()
    session.add(
        VoiceProfile(
            id="voice-1",
            name="Alice",
            language="es",
            voice_type="cloned",
            default_engine="qwen",
        )
    )
    session.commit()
    session.close()

    response = test_client.get("/v1/voices")

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert payload["data"][0]["id"] == "voice-1"
    assert payload["data"][0]["name"] == "Alice"


def test_openai_audio_speech_uses_voice_name_and_returns_opus(client):
    test_client, session_factory = client
    session = session_factory()
    session.add(
        VoiceProfile(
            id="voice-1",
            name="Alice",
            language="es",
            voice_type="cloned",
            default_engine="qwen",
        )
    )
    session.commit()
    session.close()

    response = test_client.post(
        "/v1/audio/speech",
        json={
            "model": "tts-1",
            "input": "Hola mundo",
            "voice": "Alice",
            "response_format": "opus",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/ogg")
    assert response.content == b"opus-bytes"


def test_openai_model_catalog_exposes_xtts_alias(client):
    test_client, _ = client

    response = test_client.get("/v1/models/catalog")

    assert response.status_code == 200
    payload = response.json()
    assert payload["data"][0]["slug"] == "xtts-v2"


def test_create_openai_voice_accepts_multipart_audio_sample(client, monkeypatch):
    test_client, _ = client

    async def fake_create_profile(profile_data, db):
        return type(
            "CreatedProfile",
            (),
            {
                "id": "voice-new",
                "name": profile_data.name,
                "language": profile_data.language,
                "default_engine": profile_data.default_engine,
                "created_at": datetime.utcnow(),
            },
        )()

    async def fake_add_profile_sample(profile_id, file_path, reference_text, db):
        return None

    monkeypatch.setattr(openai_compat.profile_service, "create_profile", fake_create_profile)
    monkeypatch.setattr(openai_compat.profile_service, "add_profile_sample", fake_add_profile_sample)

    response = test_client.post(
        "/v1/voices",
        data={
            "name": "Carlos",
            "language": "Spanish",
            "audio_sample_text": "hola mundo",
        },
        files={
            "audio_sample": ("sample.wav", b"fake-wav", "audio/wav"),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == "voice-new"
    assert payload["name"] == "Carlos"


def test_create_openai_voice_returns_400_for_invalid_reference_audio(client, monkeypatch):
    test_client, _ = client

    async def fake_create_profile(profile_data, db):
        return type(
            "CreatedProfile",
            (),
            {
                "id": "voice-new",
                "name": profile_data.name,
                "language": profile_data.language,
                "default_engine": profile_data.default_engine,
                "created_at": datetime.utcnow(),
            },
        )()

    async def fake_add_profile_sample(profile_id, file_path, reference_text, db):
        raise ValueError("Invalid reference audio: Audio is clipping (reduce input gain)")

    async def fake_delete_profile(profile_id, db):
        return True

    monkeypatch.setattr(openai_compat.profile_service, "create_profile", fake_create_profile)
    monkeypatch.setattr(openai_compat.profile_service, "add_profile_sample", fake_add_profile_sample)
    monkeypatch.setattr(openai_compat.profile_service, "delete_profile", fake_delete_profile)

    response = test_client.post(
        "/v1/voices",
        data={
            "name": "Carlos",
            "language": "Spanish",
            "audio_sample_text": "hola mundo",
        },
        files={
            "audio_sample": ("sample.wav", b"fake-wav", "audio/wav"),
        },
    )

    assert response.status_code == 400
    assert "Audio is clipping" in response.json()["detail"]


def test_create_openai_voice_normalizes_clipped_audio_before_saving(client, monkeypatch):
    test_client, _ = client
    captured = {}

    async def fake_create_profile(profile_data, db):
        return type(
            "CreatedProfile",
            (),
            {
                "id": "voice-new",
                "name": profile_data.name,
                "language": profile_data.language,
                "default_engine": profile_data.default_engine,
                "created_at": datetime.utcnow(),
            },
        )()

    async def fake_add_profile_sample(profile_id, file_path, reference_text, db):
        captured["file_path"] = file_path
        return None

    def fake_load_audio(path, sample_rate=24000, mono=True):
        return np.array([0.0, 1.0, -1.0], dtype=np.float32), 24000

    def fake_save_audio(audio, path, sample_rate=24000):
        Path(path).write_bytes(b"normalized")

    monkeypatch.setattr(openai_compat.profile_service, "create_profile", fake_create_profile)
    monkeypatch.setattr(openai_compat.profile_service, "add_profile_sample", fake_add_profile_sample)
    monkeypatch.setattr(openai_compat, "load_audio", fake_load_audio)
    monkeypatch.setattr(openai_compat, "save_audio", fake_save_audio)

    response = test_client.post(
        "/v1/voices",
        data={
            "name": "Carlos",
            "language": "Spanish",
            "audio_sample_text": "hola mundo",
        },
        files={
            "audio_sample": ("sample.wav", b"fake-wav", "audio/wav"),
        },
    )

    assert response.status_code == 200
    assert captured["file_path"].endswith(".wav")