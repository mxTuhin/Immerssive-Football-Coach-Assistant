import io

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FMI_DATABASE_URL", f"sqlite:///{tmp_path}/fmi.db")
    monkeypatch.setenv("FMI_STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("FMI_CELERY_EAGER", "1")
    from fmi_api import db, service
    from fmi_api.settings import get_settings

    get_settings.cache_clear()
    db.reset_engine()
    service.invalidate_package_cache()
    import importlib

    import fmi_api.tasks as tasks_mod

    importlib.reload(tasks_mod)
    import fmi_api.main as main_mod

    importlib.reload(main_mod)
    from fastapi.testclient import TestClient

    return TestClient(main_mod.app)


def test_health(client):
    assert client.get("/health").json() == {"ok": True}


def test_synthetic_match_end_to_end(client):
    r = client.post("/matches", json={"match_id": "syn1", "home_name": "Reds", "away_name": "Blues",
                                      "video_url": "synthetic://seed=5&seconds=8"})
    assert r.status_code == 200, r.text
    job = client.post("/matches/syn1/process", json={}).json()
    job = client.get(f"/jobs/{job['id']}").json()
    assert job["status"] == "done", job
    assert client.get("/matches/syn1").json()["has_package"]
    s = client.get("/matches/syn1/summary").json()
    assert s["meta"]["home"]["name"] == "Reds"
    assert isinstance(client.get("/matches/syn1/events").json(), list)
    assert client.get("/matches/syn1/metrics/physical").status_code == 200
    win = client.get("/matches/syn1/tracking", params={"period": 1, "t0": 0, "t1": 5}).json()
    assert win["format"] == "scene.v1" and len(win["players"]) >= 20
    assert client.get("/matches/syn1/frame", params={"period": 1, "t": 2.0}).status_code == 200
    review = client.get("/matches/syn1/review").json()
    assert isinstance(review, list)
    c = client.post("/matches/syn1/corrections", json={"kind": "team", "value": {"L2": "away"}}).json()
    assert c["stage"] == "team" and "team" in c["will_rerun"] and "detect" not in c["will_rerun"]
    j2 = client.post("/matches/syn1/rerun").json()
    assert client.get(f"/jobs/{j2['id']}").json()["status"] == "done"
    csv = client.get("/matches/syn1/export/events.csv")
    assert csv.status_code == 200 and csv.text.startswith("event_id")


def test_squad_upload_validation(client):
    client.post("/matches", json={"match_id": "sq"})
    bad = client.post("/matches/sq/squad", files={"file": ("s.csv", io.BytesIO(b"foo,bar\n1,2\n"), "text/csv")})
    assert bad.status_code == 422
    ok = client.post("/matches/sq/squad", files={"file": ("s.csv", io.BytesIO(b"team,number,name\nhome,1,A\naway,1,B\n"), "text/csv")})
    assert ok.json() == {"players": 2, "home": 1, "away": 1}


def test_process_requires_video(client):
    client.post("/matches", json={"match_id": "novid"})
    assert client.post("/matches/novid/process", json={}).status_code == 409
