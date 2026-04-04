"""Tests for the API server — uses FastAPI's TestClient."""
from pathlib import Path
from unittest.mock import patch

import pytest

try:
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")


def _patch_paths(tmp_path):
    db = tmp_path / "registry.db"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(exist_ok=True)
    return (
        patch("mcq.api.server.REGISTRY_DB", db),
        patch("mcq.api.server.ARTIFACTS_DIR", artifacts),
    )


def test_health(tmp_path):
    from mcq.api.server import create_app
    p1, p2 = _patch_paths(tmp_path)
    with p1, p2:
        client = TestClient(create_app())
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


def test_list_corpora_empty(tmp_path):
    from mcq.api.server import create_app
    p1, p2 = _patch_paths(tmp_path)
    with p1, p2:
        client = TestClient(create_app())
        resp = client.get("/corpora")
        assert resp.status_code == 200
        assert resp.json() == []


def test_info_not_found(tmp_path):
    from mcq.api.server import create_app
    p1, p2 = _patch_paths(tmp_path)
    with p1, p2:
        client = TestClient(create_app())
        resp = client.get("/info/nonexistent")
        assert resp.status_code == 404
