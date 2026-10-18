"""Contratos de datos y esquemas OpenAPI para el Servicio de Inferencia de Visión (Reto 2 & Reto 4).

Define entradas y salidas tipadas estrictamente con Pydantic v2 para:
- Inferencia individual y por lotes (`POST /api/v1/predict`, `POST /api/v1/predict/batch`)
- Búsqueda de vistas similares (`POST /api/v1/similar`)
- Comando determinístico hacia el brazo robótico (`RobotControlPayload`)
- Evaluación en sandbox de nuevos productos/SKUs (`POST /api/v1/sandbox/evaluate-sku`)
- Salud del servicio, cámara y enlace PLC (`GET /health`, `GET /api/v1/diagnostics`)
- Gobernanza de umbrales y estado operativo (`POST /api/v1/config/thresholds`)
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class ImageQualitySchema(BaseModel):
    """Reporte determinístico de calidad óptica de la imagen capturada."""

    is_valid: bool = Field(..., description="Indica si la imagen supera los controles ópticos mínimos.")
    laplacian_variance: float = Field(..., description="Varianza del operador Laplaciano (nitidez/enfoque).")
    mean_brightness: float = Field(..., description="Brillo promedio en escala [0, 255].")
    foreground_ratio: float = Field(..., description="Proporción de píxeles activos del objeto frente al fondo.")
    saturation_mean: float = Field(..., description="Saturación cromática media del objeto en espacio HSV.")
    issues: List[str] = Field(default_factory=list, description="Lista de anomalías ópticas detectadas.")


class SimilarViewSchema(BaseModel):
    """Información asociada a cada vista recuperada en el Top-K de imágenes más similares."""

    rank: int = Field(..., ge=1, description="Posición en el ranking de similitud (1 = más parecida).")
    sample_id: str = Field(..., description="Identificador único de la vista de referencia (ej. obj001_ang040).")
    object_id: int = Field(..., ge=1, description="ID numérico de clase en COIL-100.")
    sku_code: str = Field(..., description="Código de producto industrial (ej. SKU-001).")
    angle_deg: int = Field(..., ge=0, lt=360, description="Ángulo de rotación de referencia en grados [0, 360).")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Similitud coseno en el espacio de características.")
    filepath: str = Field(..., description="Ruta local del archivo de referencia para trazabilidad visual.")


class RobotControlPayload(BaseModel):
    """Contrato determinístico que recibe el controlador PLC / Brazo Robótico de Pick & Pack."""

    action: Literal[
        "PICK_AND_ROTATE",
        "DIVERT_TO_HUMAN_INSPECTION",
        "REJECT_IMAGE_QUALITY",
        "LINE_PAUSED_HOLD",
    ] = Field(..., description="Acción determinística ordenada al brazo robótico.")
    safe_to_act: bool = Field(..., description="True únicamente si es seguro que el robot ejecute el agarre.")
    target_sku: str = Field(..., description="SKU identificado o UNKNOWN_SKU si no es confiable.")
    estimated_object_angle_deg: Optional[float] = Field(
        None, description="Ángulo estimado del objeto en la banda [0°, 360°)."
    )
    gripper_Compensation_angle_deg: Optional[float] = Field(
        None,
        description="Rotación compensatoria (-θ mod 360°) que debe aplicar la muñeca del robot al agarrar.",
    )
    angle_uncertainty_deg: Optional[float] = Field(
        None, description="Incertidumbre angular estimada σ_θ en grados."
    )
    robot_tolerance_limit_deg: float = Field(
        default=10.0, description="Tolerancia mecánica máxima permitida por el robot (±10°)."
    )
    is_rotational_symmetric_sku: bool = Field(
        default=False,
        description="Indica si el SKU presenta simetría rotacional de 180° (agarre equivalente en θ y θ+180°).",
    )
    station_id: str = Field(default="STATION-01", description="Identificador de estación de empaque.")
    frame_id: str = Field(..., description="Identificador trazable de captura.")
    timestamp_utc: str = Field(..., description="Marca de tiempo ISO-8601 UTC.")


class VisionPredictionResponse(BaseModel):
    """Respuesta integral del Servicio de Inferencia de Visión (Capacidades 1, 2, 3 + Guardrail)."""

    request_id: str = Field(..., description="ID único de petición para auditoría.")
    frame_id: str = Field(..., description="Identificador de la imagen analizada.")
    plant_id: str = Field(..., description="Identificador de planta.")
    line_id: str = Field(..., description="Identificador de línea de empaque.")
    model_version: str = Field(..., description="Versión del artefacto de ML clásico utilizado.")

    # Capacidad 1: Identificación del objeto
    recognized: bool = Field(..., description="True si el objeto pertenece al catálogo conocido con confianza suficiente.")
    object_id: Optional[int] = Field(None, description="ID del objeto (1..100) o None si es rechazado/desconocido.")
    sku_code: str = Field(..., description="Código de SKU o UNKNOWN_SKU.")
    class_confidence: float = Field(..., ge=0.0, le=1.0, description="Nivel de confianza calibrado [0, 1].")
    ood_similarity_score: float = Field(
        ..., ge=0.0, le=1.0, description="Similitud máxima al prototipo más cercano en el manifold de entrenamiento."
    )

    # Capacidad 2: Estimación de ángulo de rotación
    angle_deg: float = Field(..., ge=0.0, lt=360.0, description="Ángulo estimado en grados [0°, 360°).")
    angle_uncertainty_deg: float = Field(..., ge=0.0, description="Incertidumbre angular σ_θ en grados.")
    is_symmetric_object: bool = Field(
        default=False, description="Indica si el objeto tiene simetría rotacional axial a 180°."
    )

    # Capacidad 3: Recuperación de imágenes más similares
    top_k_similar: List[SimilarViewSchema] = Field(
        default_factory=list, description="Top-K vistas más parecidas con su información asociada."
    )

    # Calidad óptica, Guardrail de Seguridad y Comando al Robot
    quality_report: ImageQualitySchema = Field(..., description="Diagnóstico óptico determinístico de la imagen.")
    safe_for_robot_grasp: bool = Field(..., description="Indica si se cumplen todos los criterios para agarre automático.")
    rejection_reasons: List[str] = Field(
        default_factory=list, description="Causas objetivas de rechazo cuando safe_for_robot_grasp es False."
    )
    confidence_explanation: str = Field(
        ..., description="Explicación legible y trazable de la decisión tomada y de sus métricas."
    )
    robot_payload: RobotControlPayload = Field(..., description="Payload determinístico enviado al robot.")
    total_latency_ms: float = Field(..., description="Latencia total de extremo a extremo en milisegundos.")


class PredictRequest(BaseModel):
    """Petición JSON versátil para inferencia de visión (por sample_id, filepath, base64 o escenario sintético)."""

    sample_id: Optional[str] = Field(
        default=None,
        description="ID de muestra en el catálogo COIL-100 (ej. 'obj001_ang045', 'obj019_ang015').",
    )
    image_path: Optional[str] = Field(
        default=None,
        description="Ruta local a una imagen RGB para inspección.",
    )
    image_base64: Optional[str] = Field(
        default=None,
        description="Imagen codificada en Base64 (PNG/JPG).",
    )
    synthetic_scenario: Optional[
        Literal["normal_known", "ood_unknown", "blurry_degraded", "dark_underexposed", "ambiguous_angle"]
    ] = Field(
        default=None,
        description="Permite generar o seleccionar determinísticamente un caso de prueba o adverso.",
    )
    top_k: int = Field(default=5, ge=1, le=15, description="Número de vistas similares a recuperar.")
    station_id: str = Field(default="STATION-01", description="Estación de la línea que origina la captura.")
    frame_id: Optional[str] = Field(default=None, description="ID de frame opcional.")


class BatchPredictRequest(BaseModel):
    """Petición para inspeccionar múltiples imágenes en una sola llamada (Caso de uso 1)."""

    items: List[PredictRequest] = Field(..., min_length=1, max_length=25)


class BatchPredictionResponse(BaseModel):
    """Respuesta agregada de inspección por lotes."""

    total_items: int
    accepted_count: int
    rejected_count: int
    results: List[VisionPredictionResponse]
    batch_latency_ms: float


class NewSkuEvaluationRequest(BaseModel):
    """Petición para validar e incorporar en sandbox un conjunto de vistas de un producto nuevo (SKU)."""

    sku_code: str = Field(..., min_length=3, max_length=32, description="Código del nuevo SKU (ej. SKU-105).")
    product_name: str = Field(..., min_length=2, max_length=80, description="Nombre descriptivo del producto.")
    reference_object_id: Optional[int] = Field(
        default=19,
        description="ID de objeto en COIL-100 (ej. uno de los 10 excluidos en Protocolo D) para cargar sus vistas reales.",
    )
    angles_provided_deg: Optional[List[int]] = Field(
        default=None,
        description="Lista de ángulos suministrados en grados. Si no se indica, toma las vistas disponibles.",
    )
    requested_by: str = Field(default="ml_engineer", description="Usuario que solicita el alta en sandbox.")


class NewSkuEvaluationResponse(BaseModel):
    """Resultado de la validación y prueba de desempeño en entorno controlado (Sandbox) para un nuevo SKU."""

    sku_code: str
    product_name: str
    sandbox_id: str
    material_sufficient: bool
    views_evaluated: int
    angular_coverage_pct: float
    max_angular_gap_deg: float
    optical_quality_pass_rate: float
    sandbox_cv_classification_accuracy: float
    sandbox_cv_angle_mae_deg: float
    meets_production_criteria: bool
    approval_required: bool
    recommendation: str
    validation_issues: List[str]


class ThresholdUpdateRequest(BaseModel):
    """Petición gobernada para ajustar umbrales o estado de línea (requiere aprobación humana previa)."""

    class_confidence_threshold: Optional[float] = Field(default=None, ge=0.30, le=0.99)
    ood_similarity_threshold: Optional[float] = Field(default=None, ge=0.30, le=0.99)
    robot_angle_tolerance_deg: Optional[float] = Field(default=None, ge=2.0, le=30.0)
    line_status: Optional[Literal["ACTIVE", "PAUSED"]] = Field(default=None)
    approved_by: str = Field(..., min_length=2, description="Identificador del aprobador humano.")
    approval_token: str = Field(..., min_length=4, description="Token/ID de aprobación auditable.")


class HealthResponse(BaseModel):
    """Estado de salud del servicio de inferencia de visión y componentes acoplados."""

    status: Literal["healthy", "degraded", "paused"]
    service_name: str
    model_version: str
    artifact_loaded: bool
    line_status: str
    plant_id: str
    line_id: str
    active_thresholds: Dict[str, float]
    total_predictions_served: int
    avg_latency_ms: float
    camera_sensor_status: str
    plc_link_status: str
