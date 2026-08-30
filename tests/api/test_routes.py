from pathlib import Path

from fastapi.testclient import TestClient

from bazi_api.core.config import Settings
from bazi_api.main import create_app

BACKEND_ROOT = Path(__file__).resolve().parents[2]


def test_health_and_chart_routes(tmp_path: Path) -> None:
    settings = Settings(
        knowledge_path=BACKEND_ROOT / "knowledge",
        database_path=tmp_path / "app.db",
        qdrant_path=tmp_path / "qdrant",
        vector_backend="memory",
    )

    with TestClient(create_app(settings)) as client:
        health = client.get("/api/v1/health")
        legacy_health = client.get("/api/health")
        chart = client.post(
            "/api/v1/chart",
            json={
                "date": "1990-01-01",
                "time": "12:00:00",
                "timezone": "Asia/Shanghai",
                "name": "接口测试",
            },
        )

    assert health.status_code == 200
    assert health.json()["cards"] == 50
    assert legacy_health.status_code == 200
    assert chart.status_code == 200
    assert chart.json()["day_master"] == "丙"
