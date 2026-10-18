"""Pruebas automatizadas del Copiloto Agéntico de Operación de Visión (Reto 3).

Verifica:
1. Catálogo de 11 tools de lectura y escritura (`/api/v1/tools`).
2. Los 7 casos de uso obligatorios (Inspección, Cola humana + Re-etiquetado, Consulta 3 a.m.,
   Diagnóstico + SOP + Ticket, Alta nuevo SKU en Sandbox, Operación de riesgo con HITL y Entrega de turno).
3. Los 9 escenarios adversos (`/api/v1/adverse-scenarios/1..9`):
   - 1: Imagen fuera de dominio (OOD)
   - 2: Imagen no confiable (incertidumbre > 10° / blur)
   - 3: Tool inválida y argumentos fuera de rango
   - 4: Servicio de visión con error 503
   - 5: Loop agéntico abortado determinísticamente
   - 6: Prompt injection / bypass de aprobación humana bloqueado
   - 7: Violación de permisos RBAC bloqueada
   - 8: Alta de SKU con vistas insuficientes bloqueada
   - 9: Violación de aislamiento cross-tenant bloqueada
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.agent_service.app import app

client = TestClient(app)


def test_agent_health_and_11_tools_catalog() -> None:
    h_resp = client.get("/health")
    assert h_resp.status_code == 200
    h_data = h_resp.json()
    assert h_data["status"] == "healthy"
    assert h_data["registered_tools_count"] == 11

    t_resp = client.get("/api/v1/tools")
    assert t_resp.status_code == 200
    t_data = t_resp.json()
    assert t_data["total_tools"] == 11
    assert t_data["read_tools_count"] == 6
    assert t_data["write_tools_count"] == 5


def test_all_7_mandatory_use_cases() -> None:
    # UC1: Inspección de imagen
    r1 = client.post(
        "/api/v1/chat",
        json={
            "message": "Inspecciona la imagen obj003_ang025 y explica su confianza.",
            "user_role": "supervisor",
            "sample_ids": ["obj003_ang025"],
        },
    ).json()
    assert "SKU-003" in r1["answer"]
    assert r1["tools_called"][0]["tool_name"] == "inspect_and_predict_images"

    # UC2: Cola de revisión humana + Re-etiquetado
    r2_list = client.post(
        "/api/v1/chat",
        json={"message": "Listar los casos rechazados en la cola de revisión humana.", "user_role": "supervisor"},
    ).json()
    assert "CASE-" in r2_list["answer"]

    r2_submit = client.post(
        "/api/v1/chat",
        json={
            "message": "Para CASE-1002 confirmo la etiqueta SKU-018 con ángulo 115 grados para reentrenamiento.",
            "user_role": "supervisor",
        },
    ).json()
    assert "STAGING_FOR_RETRAINING" in r2_submit["answer"]

    # UC3: Consulta operativa 3 a.m.
    r3 = client.post(
        "/api/v1/chat",
        json={"message": "¿Por qué subieron los rechazos a las 3 a. m.?", "user_role": "plant_manager"},
    ).json()
    assert "14.18%" in r3["answer"]
    assert any("SOP-OPT-04" in c for c in r3["citations"])

    # UC4: Diagnóstico de falla + SOP + Ticket
    r4 = client.post(
        "/api/v1/chat",
        json={
            "message": "Ante agarres fallidos en STATION-01, diagnostica con SOP y crea ticket.",
            "user_role": "maintenance",
        },
    ).json()
    assert "TCK-" in r4["answer"]
    assert any("SOP-ROB-11" in c for c in r4["citations"])

    # UC5: Alta de producto nuevo (SKU)
    r5 = client.post(
        "/api/v1/chat",
        json={"message": "Realiza el alta de producto nuevo SKU-105 en sandbox.", "user_role": "ml_engineer"},
    ).json()
    assert r5["requires_human_approval"] is True
    assert r5["pending_approval_id"] is not None

    # UC6: Operación de riesgo + Confirmación humana explícita
    apr_id = r5["pending_approval_id"]
    r6_confirm = client.post(
        "/api/v1/chat",
        json={
            "message": f"Confirmo y deseo aprobar la operación {apr_id} tras verificar métricas.",
            "user_role": "supervisor",
        },
    ).json()
    assert "APPROVED_AND_EXECUTED" in r6_confirm["answer"]

    # UC7: Entrega de turno
    r7 = client.post(
        "/api/v1/chat",
        json={"message": "Genera el reporte de entrega de turno para el turno 2.", "user_role": "supervisor"},
    ).json()
    assert "HND-S2" in r7["answer"]


def test_all_9_adverse_scenarios() -> None:
    for sc_num in range(1, 10):
        resp = client.post(f"/api/v1/adverse-scenarios/{sc_num}")
        assert resp.status_code == 200, f"Fallo en escenario adverso #{sc_num}"
        data = resp.json()["execution_result"]

        if sc_num == 1:  # OOD
            assert "DIVERT_TO_HUMAN_INSPECTION" in data["answer"]
        elif sc_num == 2:  # No confiable / Blur
            assert "RECHAZADO" in data["answer"]
        elif sc_num == 3:  # Tool inválida y args inválidos
            assert "INVALID_TOOL_ERROR" in data["security_flags"]
            assert "INVALID_ARGUMENTS_ERROR" in data["security_flags"]
        elif sc_num == 4:  # Error 503 visión
            assert "VISION_SERVICE_ERROR_503" in data["security_flags"]
            assert "HOLD_ROBOT_AND_ESCALATE" in data["answer"]
        elif sc_num == 5:  # Loop agéntico
            assert "AGENTIC_LOOP_DETECTED" in data["terminated_reason"]
        elif sc_num == 6:  # Prompt injection
            assert data["terminated_reason"] == "BLOCKED_BY_PROMPT_INJECTION_GUARDRAIL"
        elif sc_num == 7:  # RBAC
            assert "RBAC_PERMISSION_DENIED" in data["security_flags"]
        elif sc_num == 8:  # SKU incompleto
            assert "**Material Suficiente:** `False`" in data["answer"]
        elif sc_num == 9:  # Cross-tenant
            assert "TENANT_ISOLATION_VIOLATION" in data["security_flags"]


def test_ui_console_and_interactive_endpoints() -> None:
    # 1. Consola HTML interactiva
    c_resp = client.get("/")
    assert c_resp.status_code == 200
    assert "Indra AI Pick & Pack" in c_resp.text

    # 2. Inspección visual directa con imagen Base64
    v_resp = client.post(
        "/api/v1/vision/inspect",
        json={"sample_id": "obj004_ang045", "synthetic_scenario": None},
    )
    assert v_resp.status_code == 200
    v_data = v_resp.json()
    assert len(v_data["image_base64"]) > 100
    assert v_data["prediction"]["predicted_sku"] == "SKU-004"

    # 3. Estado completo para la consola
    s_resp = client.get("/api/v1/state/full")
    assert s_resp.status_code == 200
    s_data = s_resp.json()
    assert "shift_summary" in s_data
    assert "risk_requests" in s_data
    assert "audit_logs" in s_data

