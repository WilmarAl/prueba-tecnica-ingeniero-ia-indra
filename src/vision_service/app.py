"""API FastAPI Independiente del Servicio de Inferencia de Visión (Reto 2 & Reto 4).

Cumple con los requisitos de arquitectura:
- Servicio de inferencia de visión con API propia e independiente del agente.
- Expone las 3 capacidades del Reto 2 (Identificación con confianza/rechazo, Ángulo [0°, 360°)
  con incertidumbre, Top-K vistas similares) más la compuerta determinística hacia el robot (±10°).
- Soporta inyección controlada de fallos (`503 Service Unavailable` / `504 Gateway Timeout`) mediante
  encabezado `X-Simulate-Fault` o estado de prueba para demostrar resiliencia agéntica (Reto 3.5).
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.vision_service.engine import vision_engine
from src.vision_service.schemas import (
    BatchPredictRequest,
    BatchPredictionResponse,
    HealthResponse,
    NewSkuEvaluationRequest,
    NewSkuEvaluationResponse,
    PredictRequest,
    SimilarViewSchema,
    ThresholdUpdateRequest,
    VisionPredictionResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el artefacto versionado del modelo al iniciar el microservicio."""
    vision_engine.ensure_loaded()
    yield


app = FastAPI(
    title="Indra AI Pick & Pack — Vision Inference Service API",
    description=(
        "Servicio determinístico independiente de Visión por Computador y ML Clásico (COIL-100). "
        "Identifica SKUs, estima orientación circular en [0°, 360°) con incertidumbre σ_θ, "
        "recupera vistas Top-K similares y gobierna el comando seguro hacia el brazo robótico (±10°)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _check_fault_simulation(x_simulate_fault: Optional[str]) -> None:
    """Verifica si se solicitó simular una falla 5xx o timeout del servicio de visión (Reto 3.5 #4)."""
    fault = x_simulate_fault or vision_engine.simulated_fault_mode
    if not fault:
        return
    fault_norm = fault.strip().lower()
    if fault_norm in ("503", "error_503", "5xx", "service_unavailable"):
        raise HTTPException(
            status_code=503,
            detail={
                "error_code": "VISION_SENSOR_OR_ENGINE_UNAVAILABLE_503",
                "message": "Fallo simulado 503 en servicio de visión: pérdida temporal de bus de cámara industrial.",
                "safe_fallback_action": "HOLD_ROBOT_AND_ESCALATE",
            },
        )
    if fault_norm in ("timeout", "504", "gateway_timeout"):
        await asyncio.sleep(0.15)
        raise HTTPException(
            status_code=504,
            detail={
                "error_code": "VISION_INFERENCE_TIMEOUT_504",
                "message": "Timeout simulado (504) superando el SLA determinístico de línea.",
                "safe_fallback_action": "DIVERT_TO_HUMAN_INSPECTION",
            },
        )


@app.get("/health", response_model=HealthResponse, tags=["Operación y Salud"])
async def health_check() -> HealthResponse:
    """Devuelve el estado de salud del servicio, versión del modelo y telemetría de cámara/PLC."""
    return HealthResponse(**vision_engine.get_health())


@app.get("/api/v1/model-info", tags=["Gobierno y Model Card"])
async def get_model_info() -> Dict[str, Any]:
    """Retorna el Model Card oficial (`v1.0.0`), los umbrales activos y las métricas por protocolo."""
    vision_engine.ensure_loaded()
    return {
        "model_version": vision_engine.model.model_version,
        "line_status": vision_engine.line_status,
        "active_thresholds": vision_engine.get_health()["active_thresholds"],
        "model_card": vision_engine.model_card,
    }


@app.post("/api/v1/predict", response_model=VisionPredictionResponse, tags=["Inferencia de Visión"])
async def predict_image(
    req: PredictRequest,
    x_simulate_fault: Optional[str] = Header(default=None, alias="X-Simulate-Fault"),
) -> VisionPredictionResponse:
    """Ejecuta identificación de objeto, estimación angular circular, Top-K similares y decisión para el robot."""
    await _check_fault_simulation(x_simulate_fault)
    try:
        return vision_engine.predict(req)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/v1/predict/upload", response_model=VisionPredictionResponse, tags=["Inferencia de Visión"])
async def predict_uploaded_file(
    file: UploadFile = File(..., description="Archivo de imagen RGB (PNG/JPG/PPM)."),
    top_k: int = Form(default=5),
    station_id: str = Form(default="STATION-01"),
    frame_id: Optional[str] = Form(default=None),
    x_simulate_fault: Optional[str] = Header(default=None, alias="X-Simulate-Fault"),
) -> VisionPredictionResponse:
    """Recibe una imagen cargada vía multipart/form-data y devuelve la predicción completa."""
    await _check_fault_simulation(x_simulate_fault)
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="El archivo de imagen cargado está vacío.")
    req = PredictRequest(
        top_k=top_k,
        station_id=station_id,
        frame_id=frame_id or file.filename,
    )
    try:
        return vision_engine.predict(req, raw_bytes=raw_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Archivo de imagen inválido: {exc}") from exc


@app.post("/api/v1/predict/batch", response_model=BatchPredictionResponse, tags=["Inferencia de Visión"])
async def predict_batch(
    batch_req: BatchPredictRequest,
    x_simulate_fault: Optional[str] = Header(default=None, alias="X-Simulate-Fault"),
) -> BatchPredictionResponse:
    """Inspecciona un lote de imágenes en una sola operación (Caso de Uso 1: una o varias imágenes)."""
    await _check_fault_simulation(x_simulate_fault)
    t0 = time.perf_counter()
    results: List[VisionPredictionResponse] = []
    for item in batch_req.items:
        try:
            results.append(vision_engine.predict(item))
        except (KeyError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    accepted = sum(1 for r in results if r.safe_for_robot_grasp)
    total_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return BatchPredictionResponse(
        total_items=len(results),
        accepted_count=accepted,
        rejected_count=len(results) - accepted,
        results=results,
        batch_latency_ms=total_ms,
    )


@app.post("/api/v1/similar", response_model=List[SimilarViewSchema], tags=["Inferencia de Visión"])
async def retrieve_similar_views(
    req: PredictRequest,
    x_simulate_fault: Optional[str] = Header(default=None, alias="X-Simulate-Fault"),
) -> List[SimilarViewSchema]:
    """Capacidad 3 dedicada: Recupera las Top-K vistas más parecidas con su información asociada."""
    await _check_fault_simulation(x_simulate_fault)
    try:
        res = vision_engine.predict(req)
        return res.top_k_similar
    except (KeyError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/v1/catalog/samples", tags=["Catálogo y Pruebas"])
async def list_catalog_samples(
    object_id: Optional[int] = Query(default=None, ge=1, le=100),
    limit: int = Query(default=20, ge=1, le=100),
) -> Dict[str, Any]:
    """Devuelve muestras reales del catálogo COIL-100 y presets listos para pruebas de inspección."""
    vision_engine.ensure_loaded()
    df = vision_engine.catalog_df
    if object_id is not None:
        df = df[df["object_id"] == object_id]

    samples = df.head(limit)[["sample_id", "object_id", "sku_code", "angle_deg"]].to_dict(orient="records")
    return {
        "total_catalog_images": len(vision_engine.catalog_df),
        "returned_count": len(samples),
        "samples": samples,
        "recommended_test_presets": {
            "nominal_test_view": "obj003_ang025",
            "symmetric_180_object": "obj001_ang045",
            "high_angle_uncertainty_view": "obj007_ang335",
            "synthetic_scenarios": [
                "normal_known",
                "ood_unknown",
                "blurry_degraded",
                "dark_underexposed",
                "ambiguous_angle",
            ],
        },
    }


@app.post(
    "/api/v1/sandbox/evaluate-sku",
    response_model=NewSkuEvaluationResponse,
    tags=["Gobierno y Nuevos SKUs"],
)
async def evaluate_new_sku_in_sandbox(req: NewSkuEvaluationRequest) -> NewSkuEvaluationResponse:
    """Evalúa un conjunto de vistas de un nuevo SKU en entorno controlado antes de pasar a producción."""
    return vision_engine.evaluate_new_sku_sandbox(req)


@app.post("/api/v1/config/thresholds", tags=["Gobierno y Nuevos SKUs"])
async def update_operational_thresholds(req: ThresholdUpdateRequest) -> Dict[str, Any]:
    """Aplica cambios gobernados sobre umbrales de confianza o estado de línea con token de aprobación humana."""
    return vision_engine.update_thresholds(req)


@app.post("/api/v1/diagnostics/fault-injection", tags=["Operación y Salud"])
async def configure_fault_injection(
    mode: Optional[str] = Query(
        default=None,
        description="Modo de fallo a inyectar: 'error_503', 'timeout' o None para operación normal.",
    )
) -> Dict[str, Any]:
    """Permite activar o limpiar la simulación de errores 5xx/timeout para demos del Reto 3.5."""
    vision_engine.simulated_fault_mode = mode
    return {
        "simulated_fault_mode": vision_engine.simulated_fault_mode,
        "status": "FAULT_ACTIVE" if mode else "NORMAL_OPERATION",
    }
