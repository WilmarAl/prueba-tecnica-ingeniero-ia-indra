"""Pruebas de integración y contratos API para el Servicio de Inferencia de Visión (Reto 2 & Reto 4).

Valida:
1. `/health` y `/api/v1/model-info` (carga del artefacto `v1.0.0` y Model Card).
2. `/api/v1/predict` en caso nominal: identificación correcta de `SKU-003` a `25°`, error angular <= 10°,
   comando `PICK_AND_ROTATE` y recuperación de Top-K vistas similares.
3. `/api/v1/predict` en escenarios adversos:
   - Objeto desconocido (`ood_unknown`) -> `recognized=False`, `DIVERT_TO_HUMAN_INSPECTION`.
   - Imagen borrosa (`blurry_degraded`) -> `quality_report.is_valid=False`, `REJECT_IMAGE_QUALITY`.
   - Incertidumbre angular alta (`ambiguous_angle`) -> bloqueado por guardrail `±10°`.
4. `/api/v1/predict/batch` para múltiples imágenes simultáneas.
5. `/api/v1/sandbox/evaluate-sku` con material suficiente vs. vistas insuficientes.
6. Simulación de fallos 503 y 504 (`X-Simulate-Fault`) para resiliencia del Reto 3.5.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.vision_service.app import app

client = TestClient(app)


def test_vision_health_and_model_card() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["artifact_loaded"] is True
    assert data["model_version"] == "v1.0.0"

    mc_resp = client.get("/api/v1/model-info")
    assert mc_resp.status_code == 200
    mc = mc_resp.json()
    assert "model_card" in mc
    assert mc["model_card"]["model_name"] == "COIL100-Classical-Hybrid-PickPack"


def test_predict_nominal_known_object() -> None:
    resp = client.post(
        "/api/v1/predict",
        json={"sample_id": "obj003_ang025", "top_k": 5, "station_id": "STATION-01"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["recognized"] is True
    assert body["object_id"] == 3
    assert body["sku_code"] == "SKU-003"
    assert body["safe_for_robot_grasp"] is True
    assert body["robot_payload"]["action"] == "PICK_AND_ROTATE"
    assert len(body["top_k_similar"]) == 5
    # Verificar que el ángulo estimado esté cerca de 25° (dentro de ±10°)
    assert abs(body["angle_deg"] - 25.0) <= 10.0


def test_predict_adverse_ood_and_blurry_and_ambiguous() -> None:
    # 1. Objeto desconocido fuera de dominio (OOD)
    resp_ood = client.post("/api/v1/predict", json={"synthetic_scenario": "ood_unknown"})
    assert resp_ood.status_code == 200
    ood_data = resp_ood.json()
    assert ood_data["safe_for_robot_grasp"] is False
    assert ood_data["robot_payload"]["action"] == "DIVERT_TO_HUMAN_INSPECTION"
    assert any("UNKNOWN_OBJECT_OOD" in r for r in ood_data["rejection_reasons"])

    # 2. Imagen degradada por desenfoque
    resp_blur = client.post("/api/v1/predict", json={"synthetic_scenario": "blurry_degraded"})
    assert resp_blur.status_code == 200
    blur_data = resp_blur.json()
    assert blur_data["quality_report"]["is_valid"] is False
    assert blur_data["safe_for_robot_grasp"] is False
    assert blur_data["robot_payload"]["action"] == "REJECT_IMAGE_QUALITY"

    # 3. Caso con incertidumbre angular > 10°
    resp_amb = client.post("/api/v1/predict", json={"synthetic_scenario": "ambiguous_angle"})
    assert resp_amb.status_code == 200
    amb_data = resp_amb.json()
    assert amb_data["safe_for_robot_grasp"] is False
    assert amb_data["angle_uncertainty_deg"] > 10.0


def test_predict_batch_and_sandbox_sku() -> None:
    batch_resp = client.post(
        "/api/v1/predict/batch",
        json={
            "items": [
                {"sample_id": "obj003_ang025"},
                {"synthetic_scenario": "ood_unknown"},
            ]
        },
    )
    assert batch_resp.status_code == 200
    b_data = batch_resp.json()
    assert b_data["total_items"] == 2
    assert b_data["accepted_count"] == 1
    assert b_data["rejected_count"] == 1

    # Sandbox SKU con vistas suficientes (18 vistas cada 20°)
    sbx_ok = client.post(
        "/api/v1/sandbox/evaluate-sku",
        json={"sku_code": "SKU-105", "product_name": "Envase Industrial Nuevo", "reference_object_id": 19},
    )
    assert sbx_ok.status_code == 200
    assert sbx_ok.json()["material_sufficient"] is True

    # Sandbox SKU con vistas insuficientes (solo 4 vistas)
    sbx_bad = client.post(
        "/api/v1/sandbox/evaluate-sku",
        json={
            "sku_code": "SKU-106",
            "product_name": "Envase Incompleto",
            "reference_object_id": 22,
            "angles_provided_deg": [0, 90, 180, 270],
        },
    )
    assert sbx_bad.status_code == 200
    assert sbx_bad.json()["material_sufficient"] is False


def test_fault_simulation_503_and_504() -> None:
    r503 = client.post(
        "/api/v1/predict",
        json={"sample_id": "obj001_ang015"},
        headers={"X-Simulate-Fault": "error_503"},
    )
    assert r503.status_code == 503

    r504 = client.post(
        "/api/v1/predict",
        json={"sample_id": "obj001_ang015"},
        headers={"X-Simulate-Fault": "timeout"},
    )
    assert r504.status_code == 504
