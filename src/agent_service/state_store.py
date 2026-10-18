"""Gestión de Estado, Memoria Persistente, Contexto Conversacional y Auditoría (Reto 3, Sección 6.4).

Diferencia arquitectónicamente las 4 capas de información exigidas en la Sección 6.4:
1. `ExecutionState` (Estado de ejecución efímero por turno):
   Vive únicamente durante el ciclo de resolución de una petición (`request_id`). Controla
   el contador de pasos (`step_count <= max_steps`), los hashes de llamadas a tools para cortar
   loops agénticos, el presupuesto de tokens consumidos y las banderas de bloqueo HITL.
2. `ConversationHistory` (Historial conversacional por sesión con control de crecimiento):
   Asociado a `(plant_id, line_id, session_id)`. Aplica una estrategia de ventana deslizante
   (`max_recent_turns = 6`) más un resumen determinístico compacto de los turnos previos,
   garantizando que el contexto enviado al LLM nunca crezca sin límite ni incluya tensores/base64.
3. `PersistentOperationalMemory` (Memoria operativa transaccional en SQLite):
   Almacena con claves primarias, marcas de tiempo y políticas de retención (TTL):
   - `rejection_queue` (casos rechazados pendientes de revisión humana con su evidencia Top-K)
   - `retraining_dataset` (etiquetas y ángulos confirmados por humanos para reentrenamiento)
   - `shift_telemetry` y `hourly_telemetry` (KPIs por turno y hora, incluyendo anomalía de las 3 a.m.)
   - `maintenance_tickets` (tickets abiertos/escalados con clave de idempotencia)
   - `sku_sandbox_registry` (evaluaciones de nuevos productos antes de producción)
   - `risk_operation_approvals` (solicitudes y confirmaciones HITL de operaciones críticas)
   - `audit_log` (registro inmutable de cada invocación de tool, usuario, rol, argumentos y resultado)
4. `RetrievedKnowledge` (Conocimiento recuperado RAG con citas):
   Fragmentos de SOPs y manuales recuperados de `data/knowledge_base/plant_sops_manual.json`
   con su cita obligatoria `[DOC-ID, Sección]`.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import PROJECT_ROOT, settings


@dataclass
class ExecutionState:
    """1. Estado de ejecución efímero (vive durante una sola invocación del agente)."""

    request_id: str
    session_id: str
    plant_id: str
    line_id: str
    user_id: str
    user_role: str
    max_steps: int = 5
    token_budget: int = 4000
    step_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_call_signatures: List[str] = field(default_factory=list)
    executed_tools: List[Dict[str, Any]] = field(default_factory=list)
    loop_detected: bool = False
    terminated_reason: str = "IN_PROGRESS"
    requires_human_approval: bool = False
    pending_approval_id: Optional[str] = None


# Políticas explícitas de retención (Sección 6.4)
RETENTION_POLICIES_DAYS: Dict[str, Any] = {
    "execution_state": "Efímero en RAM (0 días — se destruye al finalizar el request_id)",
    "conversation_history": "7 días por session_id (ventana activa de últimos 6 turnos + resumen comprimido)",
    "uploaded_images_and_frames": "90 días en almacenamiento de objetos (referenciados por frame_id y sha256_hash)",
    "rejection_queue": "30 días en estado PENDING_REVIEW; histórico perpetuo al pasar a RELABELED",
    "retraining_dataset": "365 días por versión de dataset (vinculado a sample_id, frame_id y human_reviewer)",
    "maintenance_tickets": "365 días (trazabilidad de fallas mecánicas/ópticas por station_id)",
    "risk_operation_approvals_and_audit_log": "1825 días (5 años — cumplimiento normativo y auditoría industrial inmutable)",
}


class OperationalStateStore:
    """Gestor SQLite con aislamiento por planta/línea, semillas realistas y auditoría completa."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = Path(db_path) if db_path else (PROJECT_ROOT / settings.agent_state_db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema_and_seed()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema_and_seed(self) -> None:
        with self._connect() as conn:
            cur = conn.cursor()

            cur.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plant_id TEXT NOT NULL,
                    line_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    user_role TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    agent_response TEXT NOT NULL,
                    tools_used TEXT NOT NULL,
                    tokens_used INTEGER NOT NULL,
                    created_at_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS session_summaries (
                    session_id TEXT PRIMARY KEY,
                    plant_id TEXT NOT NULL,
                    line_id TEXT NOT NULL,
                    compressed_summary TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS rejection_queue (
                    case_id TEXT PRIMARY KEY,
                    plant_id TEXT NOT NULL,
                    line_id TEXT NOT NULL,
                    shift_number INTEGER NOT NULL,
                    frame_id TEXT NOT NULL,
                    sample_id TEXT NOT NULL,
                    candidate_sku TEXT NOT NULL,
                    class_confidence REAL NOT NULL,
                    ood_similarity REAL NOT NULL,
                    estimated_angle_deg REAL NOT NULL,
                    angle_uncertainty_deg REAL NOT NULL,
                    rejection_reason TEXT NOT NULL,
                    top_k_evidence_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING_HUMAN_REVIEW',
                    confirmed_sku TEXT,
                    confirmed_angle_deg REAL,
                    reviewed_by TEXT,
                    created_at_utc TEXT NOT NULL,
                    reviewed_at_utc TEXT
                );

                CREATE TABLE IF NOT EXISTS retraining_dataset (
                    record_id TEXT PRIMARY KEY,
                    case_id TEXT UNIQUE NOT NULL,
                    plant_id TEXT NOT NULL,
                    line_id TEXT NOT NULL,
                    frame_id TEXT NOT NULL,
                    confirmed_sku TEXT NOT NULL,
                    confirmed_angle_deg REAL NOT NULL,
                    reviewer_id TEXT NOT NULL,
                    reviewer_role TEXT NOT NULL,
                    notes TEXT,
                    dataset_stage TEXT NOT NULL DEFAULT 'STAGING_FOR_RETRAINING',
                    created_at_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS shift_telemetry (
                    plant_id TEXT NOT NULL,
                    line_id TEXT NOT NULL,
                    shift_number INTEGER NOT NULL,
                    shift_name TEXT NOT NULL,
                    total_processed INTEGER NOT NULL,
                    auto_picked INTEGER NOT NULL,
                    rejected_to_queue INTEGER NOT NULL,
                    rejection_rate_pct REAL NOT NULL,
                    avg_latency_ms REAL NOT NULL,
                    avg_angle_error_deg REAL NOT NULL,
                    failed_grasps_count INTEGER NOT NULL,
                    camera_health_status TEXT NOT NULL,
                    active_model_version TEXT NOT NULL,
                    summary_notes TEXT NOT NULL,
                    PRIMARY KEY (plant_id, line_id, shift_number)
                );

                CREATE TABLE IF NOT EXISTS hourly_telemetry (
                    plant_id TEXT NOT NULL,
                    line_id TEXT NOT NULL,
                    shift_number INTEGER NOT NULL,
                    hour_slot TEXT NOT NULL,
                    processed_count INTEGER NOT NULL,
                    rejected_count INTEGER NOT NULL,
                    rejection_rate_pct REAL NOT NULL,
                    camera_led_intensity_pct REAL NOT NULL,
                    mean_laplacian_sharpness REAL NOT NULL,
                    model_version TEXT NOT NULL,
                    anomaly_cause TEXT NOT NULL,
                    PRIMARY KEY (plant_id, line_id, shift_number, hour_slot)
                );

                CREATE TABLE IF NOT EXISTS maintenance_tickets (
                    ticket_id TEXT PRIMARY KEY,
                    idempotency_key TEXT UNIQUE NOT NULL,
                    plant_id TEXT NOT NULL,
                    line_id TEXT NOT NULL,
                    station_id TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    status TEXT NOT NULL,
                    issue_category TEXT NOT NULL,
                    diagnosis_summary TEXT NOT NULL,
                    sop_citations TEXT NOT NULL,
                    created_by_role TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sku_sandbox_registry (
                    sandbox_id TEXT PRIMARY KEY,
                    plant_id TEXT NOT NULL,
                    line_id TEXT NOT NULL,
                    sku_code TEXT NOT NULL,
                    product_name TEXT NOT NULL,
                    views_evaluated INTEGER NOT NULL,
                    angular_coverage_pct REAL NOT NULL,
                    max_angular_gap_deg REAL NOT NULL,
                    cv_accuracy REAL NOT NULL,
                    cv_angle_mae_deg REAL NOT NULL,
                    meets_criteria INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS risk_operation_approvals (
                    approval_id TEXT PRIMARY KEY,
                    idempotency_key TEXT UNIQUE NOT NULL,
                    plant_id TEXT NOT NULL,
                    line_id TEXT NOT NULL,
                    operation_type TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    requested_by_role TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING_HUMAN_APPROVAL',
                    approved_by TEXT,
                    approver_role TEXT,
                    decision_notes TEXT,
                    created_at_utc TEXT NOT NULL,
                    decided_at_utc TEXT
                );

                CREATE TABLE IF NOT EXISTS audit_log (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    plant_id TEXT NOT NULL,
                    line_id TEXT NOT NULL,
                    user_role TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    arguments_redacted_json TEXT NOT NULL,
                    outcome_status TEXT NOT NULL,
                    result_summary TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL
                );
                """
            )

            # Verificar si ya existen datos semilla
            cur.execute("SELECT COUNT(*) as c FROM shift_telemetry")
            if cur.fetchone()["c"] == 0:
                self._seed_initial_operational_data(cur)
            conn.commit()

    def _seed_initial_operational_data(self, cur: sqlite3.Cursor) -> None:
        """Puebla datos operativos realistas para los 3 turnos, la anomalía de las 3 a.m. y la cola humana."""
        now_iso = datetime.now(timezone.utc).isoformat()
        p_id = settings.plant_id
        l_id = settings.line_id

        # 1. Telemetría de los 3 turnos de 8 horas (24/5)
        shifts = [
            (
                p_id, l_id, 1, "Turno 1 (06:00 - 14:00)",
                4820, 4685, 135, 2.80, 12.8, 2.75, 1,
                "NOMINAL_OK (LED 99%, Laplaciano 64.2)", "v1.0.0",
                "Operación nominal sin paradas de línea; throughput al 100.4% del objetivo.",
            ),
            (
                p_id, l_id, 2, "Turno 2 (14:00 - 22:00)",
                4710, 4498, 212, 4.50, 13.1, 3.10, 4,
                "WARNING_VIBRATION_ST01 (LED 96%, Laplaciano 51.0)", "v1.0.0",
                "Incremento moderado de rechazos en Estación 1 por holgura de ventosa en SKU-007 y SKU-018.",
            ),
            (
                p_id, l_id, 3, "Turno 3 Nocturno (22:00 - 06:00)",
                4590, 4186, 404, 8.80, 13.6, 4.45, 7,
                "DEGRADED_LED_FLICKER_03AM (LED 78% @ 03:00, Laplaciano 31.4)", "v1.0.0",
                "Pico anómalo de rechazos a las 03:00 a.m. (14.2%) causado por caída de intensidad en el anillo LED cenital (SOP-OPT-04).",
            ),
        ]
        cur.executemany(
            """
            INSERT INTO shift_telemetry VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            shifts,
        )

        # 2. Telemetría horaria detallada (incluyendo la pregunta obligatoria: '¿por qué subieron los rechazos a las 3 a. m.?')
        hourly = [
            (p_id, l_id, 2, "14:00", 600, 18, 3.0, 98.0, 62.0, "v1.0.0", "Operación normal"),
            (p_id, l_id, 2, "18:00", 590, 32, 5.4, 96.5, 49.5, "v1.0.0", "Micro-vibración en pinza J6 con envases asimétricos"),
            (p_id, l_id, 2, "03:00", 550, 78, 14.18, 76.4, 29.8, "v1.0.0", "Deriva térmica en fuente PWM de anillo LED cenital (brillo cayó al 76.4% y nitidez Laplaciana a 29.8 < 35.0) tras micro-corte eléctrico; modelo v1.0.0 sin cambios"),
            (p_id, l_id, 3, "02:00", 580, 21, 3.62, 97.2, 60.1, "v1.0.0", "Operación normal previa al evento lumínico"),
            (p_id, l_id, 3, "03:00", 550, 78, 14.18, 76.4, 29.8, "v1.0.0", "Deriva térmica en fuente PWM de anillo LED cenital (brillo cayó al 76.4% y nitidez Laplaciana a 29.8 < 35.0) tras micro-corte eléctrico; modelo v1.0.0 sin cambios"),
            (p_id, l_id, 3, "04:00", 575, 49, 8.52, 84.0, 36.5, "v1.0.0", "Recuperación parcial tras reinicio automático del controlador de iluminación"),
        ]
        cur.executemany(
            """
            INSERT INTO hourly_telemetry VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            hourly,
        )

        # 3. Casos en la Cola de Revisión Humana (Caso de uso 2)
        queue_items = [
            (
                "CASE-1001", p_id, l_id, 2, "FRM-007-335", "obj007_ang335",
                "SKU-007", 0.8486, 0.8610, 346.0, 21.35,
                "ANGLE_UNCERTAINTY_EXCEEDS_ROBOT_TOLERANCE (sigma_theta=21.35° > 10.0°)",
                json.dumps([
                    {"rank": 1, "sample_id": "obj007_ang340", "sku_code": "SKU-007", "angle_deg": 340, "similarity": 0.861},
                    {"rank": 2, "sample_id": "obj007_ang330", "sku_code": "SKU-007", "angle_deg": 330, "similarity": 0.854},
                ]),
                "PENDING_HUMAN_REVIEW", None, None, None, now_iso, None,
            ),
            (
                "CASE-1002", p_id, l_id, 2, "FRM-018-115", "obj018_ang115",
                "SKU-018", 0.8631, 0.8720, 125.14, 13.36,
                "ANGLE_UNCERTAINTY_EXCEEDS_ROBOT_TOLERANCE (sigma_theta=13.36° > 10.0°)",
                json.dumps([
                    {"rank": 1, "sample_id": "obj018_ang120", "sku_code": "SKU-018", "angle_deg": 120, "similarity": 0.872},
                    {"rank": 2, "sample_id": "obj018_ang110", "sku_code": "SKU-018", "angle_deg": 110, "similarity": 0.865},
                ]),
                "PENDING_HUMAN_REVIEW", None, None, None, now_iso, None,
            ),
            (
                "CASE-1003", p_id, l_id, 3, "FRM-OOD-019", "obj019_ang045",
                "UNKNOWN_SKU", 0.4820, 0.5410, 48.0, 19.40,
                "UNKNOWN_OBJECT_OOD (max_similarity=0.541 < threshold=0.6666); LOW_CLASS_CONFIDENCE",
                json.dumps([
                    {"rank": 1, "sample_id": "obj014_ang050", "sku_code": "SKU-014", "angle_deg": 50, "similarity": 0.541},
                ]),
                "PENDING_HUMAN_REVIEW", None, None, None, now_iso, None,
            ),
        ]
        cur.executemany(
            """
            INSERT INTO rejection_queue VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            queue_items,
        )

        # 4. Ticket inicial abierto para el reporte de entrega de turno
        cur.execute(
            """
            INSERT INTO maintenance_tickets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "TCK-401",
                "seed-ticket-401",
                p_id,
                l_id,
                "STATION-01",
                "HIGH",
                "OPEN",
                "OPTICAL_LED_DRIFT",
                "Caída de intensidad en anillo LED cenital a las 03:00 a.m. provocando baja nitidez Laplaciana (29.8).",
                "[SOP-OPT-04, Sección 3.2]",
                "supervisor",
                now_iso,
            ),
        )

    # =========================================================================
    # Métodos de Consulta y Escritura Gobernada con Aislamiento de Planta/Línea
    # =========================================================================

    def validate_tenant_isolation(self, plant_id: str, line_id: str) -> None:
        """Verifica que la petición no intente acceder a otra planta o línea no autorizada."""
        if plant_id != settings.plant_id or line_id != settings.line_id:
            raise PermissionError(
                f"TENANT_ISOLATION_VIOLATION: Sesión autorizada únicamente para "
                f"({settings.plant_id}, {settings.line_id}). Acceso denegado a ({plant_id}, {line_id})."
            )

    def enqueue_rejected_prediction(
        self,
        pred_dict: Dict[str, Any],
        shift_number: int = 2,
    ) -> str:
        """Inserta un caso rechazado en `rejection_queue` de forma idempotente."""
        frame_id = str(pred_dict.get("frame_id", f"FRM-{uuid.uuid4().hex[:6].upper()}"))
        case_id = f"CASE-{hashlib.sha1(frame_id.encode('utf-8')).hexdigest()[:6].upper()}"
        now_iso = datetime.now(timezone.utc).isoformat()
        reasons = "; ".join(pred_dict.get("rejection_reasons", ["LOW_CONFIDENCE"]))
        top_k_json = json.dumps(pred_dict.get("top_k_similar", [])[:3], ensure_ascii=False)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO rejection_queue VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    case_id,
                    settings.plant_id,
                    settings.line_id,
                    shift_number,
                    frame_id,
                    frame_id,
                    str(pred_dict.get("sku_code", "UNKNOWN_SKU")),
                    float(pred_dict.get("class_confidence", 0.0)),
                    float(pred_dict.get("ood_similarity_score", 0.0)),
                    float(pred_dict.get("angle_deg", 0.0)),
                    float(pred_dict.get("angle_uncertainty_deg", 0.0)),
                    reasons,
                    top_k_json,
                    "PENDING_HUMAN_REVIEW",
                    None,
                    None,
                    None,
                    now_iso,
                    None,
                ),
            )
            conn.commit()
        return case_id

    def list_rejection_queue(
        self,
        plant_id: str,
        line_id: str,
        status: str = "PENDING_HUMAN_REVIEW",
        shift_number: Optional[int] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        self.validate_tenant_isolation(plant_id, line_id)
        query = "SELECT * FROM rejection_queue WHERE plant_id = ? AND line_id = ?"
        params: List[Any] = [plant_id, line_id]
        if status != "ALL":
            query += " AND status = ?"
            params.append(status)
        if shift_number is not None:
            query += " AND shift_number = ?"
            params.append(shift_number)
        query += " ORDER BY created_at_utc DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["top_k_evidence"] = json.loads(d.pop("top_k_evidence_json", "[]"))
            out.append(d)
        return out

    def submit_human_relabeling(
        self,
        plant_id: str,
        line_id: str,
        case_id: str,
        confirmed_sku: str,
        confirmed_angle_deg: float,
        reviewer_id: str,
        reviewer_role: str,
        notes: str = "",
    ) -> Dict[str, Any]:
        """Registra de forma idempotente la etiqueta y ángulo confirmados por el humano y lo envía a reentrenamiento."""
        self.validate_tenant_isolation(plant_id, line_id)
        now_iso = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM rejection_queue WHERE case_id = ? AND plant_id = ? AND line_id = ?",
                (case_id, plant_id, line_id),
            ).fetchone()
            if not row:
                raise KeyError(f"El caso '{case_id}' no existe en la cola de revisión de ({plant_id}, {line_id}).")

            existing = conn.execute(
                "SELECT * FROM retraining_dataset WHERE case_id = ?", (case_id,)
            ).fetchone()
            if existing:
                return {
                    "idempotent_replay": True,
                    "status": "ALREADY_RELABELED_AND_QUEUED",
                    "record_id": existing["record_id"],
                    "case_id": case_id,
                    "confirmed_sku": existing["confirmed_sku"],
                    "confirmed_angle_deg": existing["confirmed_angle_deg"],
                    "dataset_stage": existing["dataset_stage"],
                }

            rec_id = f"RTR-{hashlib.sha1(f'{case_id}:{confirmed_sku}'.encode('utf-8')).hexdigest()[:8].upper()}"
            conn.execute(
                """
                UPDATE rejection_queue
                SET status = 'CONFIRMED_FOR_RETRAINING',
                    confirmed_sku = ?,
                    confirmed_angle_deg = ?,
                    reviewed_by = ?,
                    reviewed_at_utc = ?
                WHERE case_id = ?
                """,
                (confirmed_sku, float(confirmed_angle_deg), reviewer_id, now_iso, case_id),
            )
            conn.execute(
                """
                INSERT INTO retraining_dataset VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rec_id,
                    case_id,
                    plant_id,
                    line_id,
                    row["frame_id"],
                    confirmed_sku,
                    float(confirmed_angle_deg),
                    reviewer_id,
                    reviewer_role,
                    notes,
                    "STAGING_FOR_RETRAINING",
                    now_iso,
                ),
            )
            conn.commit()

        return {
            "idempotent_replay": False,
            "status": "CONFIRMED_AND_SENT_TO_RETRAINING",
            "record_id": rec_id,
            "case_id": case_id,
            "frame_id": row["frame_id"],
            "confirmed_sku": confirmed_sku,
            "confirmed_angle_deg": round(float(confirmed_angle_deg), 2),
            "dataset_stage": "STAGING_FOR_RETRAINING",
            "timestamp_utc": now_iso,
        }

    def query_telemetry(
        self,
        plant_id: str,
        line_id: str,
        shift_number: Optional[int] = None,
        hour_slot: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Consulta métricas operativas por turno y hora correlacionando rechazos, cámara y versión de modelo."""
        self.validate_tenant_isolation(plant_id, line_id)
        with self._connect() as conn:
            if shift_number is not None:
                s_rows = conn.execute(
                    "SELECT * FROM shift_telemetry WHERE plant_id = ? AND line_id = ? AND shift_number = ?",
                    (plant_id, line_id, shift_number),
                ).fetchall()
            else:
                s_rows = conn.execute(
                    "SELECT * FROM shift_telemetry WHERE plant_id = ? AND line_id = ? ORDER BY shift_number",
                    (plant_id, line_id),
                ).fetchall()

            h_query = "SELECT * FROM hourly_telemetry WHERE plant_id = ? AND line_id = ?"
            h_params: List[Any] = [plant_id, line_id]
            if shift_number is not None:
                h_query += " AND shift_number = ?"
                h_params.append(shift_number)
            if hour_slot:
                h_query += " AND hour_slot = ?"
                h_params.append(hour_slot)
            h_rows = conn.execute(h_query, h_params).fetchall()

        return {
            "plant_id": plant_id,
            "line_id": line_id,
            "shifts": [dict(r) for r in s_rows],
            "hourly_breakdown": [dict(r) for r in h_rows],
        }

    def create_or_escalate_ticket(
        self,
        plant_id: str,
        line_id: str,
        station_id: str,
        priority: str,
        issue_category: str,
        diagnosis_summary: str,
        sop_citations: List[str],
        created_by_role: str,
    ) -> Dict[str, Any]:
        """Crea o escala un ticket de mantenimiento de forma idempotente."""
        self.validate_tenant_isolation(plant_id, line_id)
        idem_raw = f"{plant_id}:{line_id}:{station_id}:{issue_category}:{diagnosis_summary[:40].lower()}"
        idem_key = hashlib.sha256(idem_raw.encode("utf-8")).hexdigest()[:16]
        now_iso = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM maintenance_tickets WHERE idempotency_key = ?", (idem_key,)
            ).fetchone()
            if existing:
                return {
                    "idempotent_replay": True,
                    "ticket_id": existing["ticket_id"],
                    "status": existing["status"],
                    "priority": existing["priority"],
                    "station_id": existing["station_id"],
                    "issue_category": existing["issue_category"],
                    "diagnosis_summary": existing["diagnosis_summary"],
                    "sop_citations": json.loads(existing["sop_citations"]),
                }

            ticket_id = f"TCK-{uuid.uuid4().hex[:6].upper()}"
            conn.execute(
                """
                INSERT INTO maintenance_tickets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticket_id,
                    idem_key,
                    plant_id,
                    line_id,
                    station_id,
                    priority.upper(),
                    "OPEN_ESCALATED" if priority.upper() in ("HIGH", "CRITICAL") else "OPEN",
                    issue_category,
                    diagnosis_summary,
                    json.dumps(sop_citations, ensure_ascii=False),
                    created_by_role,
                    now_iso,
                ),
            )
            conn.commit()

        return {
            "idempotent_replay": False,
            "ticket_id": ticket_id,
            "status": "OPEN_ESCALATED" if priority.upper() in ("HIGH", "CRITICAL") else "OPEN",
            "priority": priority.upper(),
            "station_id": station_id,
            "issue_category": issue_category,
            "diagnosis_summary": diagnosis_summary,
            "sop_citations": sop_citations,
            "created_at_utc": now_iso,
        }

    def reset_for_reproducible_benchmark(self) -> None:
        """Restaura el estado inicial semilla para garantizar que los benchmarks sean 100% deterministas."""
        if self.db_path.exists():
            self.db_path.unlink()
        self._init_schema_and_seed()

    def create_risk_operation_request(
        self,
        plant_id: str,
        line_id: str,
        operation_type: str,
        parameters: Dict[str, Any],
        requested_by_role: str,
        reason: str,
    ) -> Dict[str, Any]:
        """Registra una solicitud de operación de riesgo en estado `PENDING_HUMAN_APPROVAL` (Idempotente)."""
        self.validate_tenant_isolation(plant_id, line_id)
        params_canonical = json.dumps(parameters, sort_keys=True, ensure_ascii=False)
        idem_raw = f"{plant_id}:{line_id}:{operation_type}:{params_canonical}"
        idem_key = hashlib.sha256(idem_raw.encode("utf-8")).hexdigest()[:16]
        now_iso = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM risk_operation_approvals WHERE idempotency_key = ? AND status = 'PENDING_HUMAN_APPROVAL'",
                (idem_key,),
            ).fetchone()
            if existing:
                apr_id_ex = existing["approval_id"]
                return {
                    "idempotent_replay": True,
                    "approval_id": apr_id_ex,
                    "operation_type": existing["operation_type"],
                    "parameters": json.loads(existing["parameters_json"]),
                    "status": existing["status"],
                    "requires_explicit_human_confirmation": True,
                    "message": (
                        f"Operación crítica '{operation_type}' ya cuenta con solicitud abierta '{apr_id_ex}' "
                        "pendiente de confirmación humana explícita."
                    ),
                }

            approval_id = f"APR-{uuid.uuid4().hex[:6].upper()}"
            unique_idem = f"{idem_key}-{approval_id}"
            conn.execute(
                """
                INSERT INTO risk_operation_approvals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    approval_id,
                    unique_idem,
                    plant_id,
                    line_id,
                    operation_type,
                    params_canonical,
                    requested_by_role,
                    reason,
                    "PENDING_HUMAN_APPROVAL",
                    None,
                    None,
                    None,
                    now_iso,
                    None,
                ),
            )
            conn.commit()

        return {
            "idempotent_replay": False,
            "approval_id": approval_id,
            "operation_type": operation_type,
            "parameters": parameters,
            "status": "PENDING_HUMAN_APPROVAL",
            "requires_explicit_human_confirmation": True,
            "message": (
                f"Operación crítica '{operation_type}' bloqueada preventivamente por el guardrail HITL. "
                f"Se requiere confirmación humana explícita invocando `confirm_critical_line_operation` con approval_id='{approval_id}'."
            ),
        }

    def confirm_risk_operation(
        self,
        plant_id: str,
        line_id: str,
        approval_id: str,
        approved: bool,
        approved_by: str,
        approver_role: str,
        decision_notes: str,
    ) -> Dict[str, Any]:
        """Aprueba o rechaza una operación crítica previamente registrada."""
        self.validate_tenant_isolation(plant_id, line_id)
        now_iso = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM risk_operation_approvals WHERE approval_id = ? AND plant_id = ? AND line_id = ?",
                (approval_id, plant_id, line_id),
            ).fetchone()
            if not row:
                raise KeyError(f"La solicitud de aprobación '{approval_id}' no existe.")

            if row["status"] != "PENDING_HUMAN_APPROVAL":
                return {
                    "idempotent_replay": True,
                    "approval_id": approval_id,
                    "operation_type": row["operation_type"],
                    "status": row["status"],
                    "approved_by": row["approved_by"],
                }

            new_status = "APPROVED_AND_EXECUTED" if approved else "REJECTED_BY_HUMAN"
            conn.execute(
                """
                UPDATE risk_operation_approvals
                SET status = ?, approved_by = ?, approver_role = ?, decision_notes = ?, decided_at_utc = ?
                WHERE approval_id = ?
                """,
                (new_status, approved_by, approver_role, decision_notes, now_iso, approval_id),
            )
            conn.commit()

        return {
            "idempotent_replay": False,
            "approval_id": approval_id,
            "operation_type": row["operation_type"],
            "parameters": json.loads(row["parameters_json"]),
            "status": new_status,
            "approved_by": approved_by,
            "approver_role": approver_role,
            "decided_at_utc": now_iso,
        }

    def register_sandbox_evaluation(self, eval_data: Dict[str, Any]) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sku_sandbox_registry VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    eval_data["sandbox_id"],
                    settings.plant_id,
                    settings.line_id,
                    eval_data["sku_code"],
                    eval_data["product_name"],
                    int(eval_data["views_evaluated"]),
                    float(eval_data["angular_coverage_pct"]),
                    float(eval_data["max_angular_gap_deg"]),
                    float(eval_data["sandbox_cv_classification_accuracy"]),
                    float(eval_data["sandbox_cv_angle_mae_deg"]),
                    1 if eval_data["meets_production_criteria"] else 0,
                    "SANDBOX_VALIDATED_PENDING_APPROVAL"
                    if eval_data["meets_production_criteria"]
                    else "SANDBOX_REJECTED_INSUFFICIENT_MATERIAL",
                    now_iso,
                ),
            )
            conn.commit()

    def build_shift_handover_report(
        self,
        plant_id: str,
        line_id: str,
        shift_number: int = 2,
    ) -> Dict[str, Any]:
        """Compila el reporte integral de entrega de turno con métricas, incidentes, tickets y cola pendiente."""
        self.validate_tenant_isolation(plant_id, line_id)
        with self._connect() as conn:
            shift_row = conn.execute(
                "SELECT * FROM shift_telemetry WHERE plant_id = ? AND line_id = ? AND shift_number = ?",
                (plant_id, line_id, shift_number),
            ).fetchone()
            if not shift_row:
                shift_row = conn.execute(
                    "SELECT * FROM shift_telemetry WHERE plant_id = ? AND line_id = ? LIMIT 1",
                    (plant_id, line_id),
                ).fetchone()

            open_tickets = conn.execute(
                "SELECT ticket_id, station_id, priority, status, issue_category, diagnosis_summary, sop_citations "
                "FROM maintenance_tickets WHERE plant_id = ? AND line_id = ? AND status LIKE 'OPEN%'",
                (plant_id, line_id),
            ).fetchall()

            pending_queue = conn.execute(
                "SELECT case_id, frame_id, candidate_sku, class_confidence, angle_uncertainty_deg, rejection_reason "
                "FROM rejection_queue WHERE plant_id = ? AND line_id = ? AND status = 'PENDING_HUMAN_REVIEW'",
                (plant_id, line_id),
            ).fetchall()

            pending_approvals = conn.execute(
                "SELECT approval_id, operation_type, parameters_json, requested_by_role, status "
                "FROM risk_operation_approvals WHERE plant_id = ? AND line_id = ? AND status = 'PENDING_HUMAN_APPROVAL'",
                (plant_id, line_id),
            ).fetchall()

        return {
            "handover_id": f"HND-S{shift_number}-{datetime.now(timezone.utc).strftime('%Y%m%d')}",
            "plant_id": plant_id,
            "line_id": line_id,
            "shift_metrics": dict(shift_row) if shift_row else {},
            "open_maintenance_tickets": [dict(t) for t in open_tickets],
            "pending_human_review_cases": [dict(c) for c in pending_queue],
            "pending_risk_approvals": [dict(a) for a in pending_approvals],
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    # =========================================================================
    # Gestión de Contexto Conversacional y Registro Inmutable de Auditoría
    # =========================================================================

    def append_conversation_turn(
        self,
        plant_id: str,
        line_id: str,
        session_id: str,
        user_role: str,
        user_message: str,
        agent_response: str,
        tools_used: List[str],
        tokens_used: int,
        max_window_turns: int = 6,
    ) -> None:
        """Persiste el turno conversacional y comprime turnos antiguos si supera la ventana deslizante."""
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversation_turns
                (plant_id, line_id, session_id, user_role, user_message, agent_response, tools_used, tokens_used, created_at_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    plant_id,
                    line_id,
                    session_id,
                    user_role,
                    user_message[:600],
                    agent_response[:1200],
                    json.dumps(tools_used),
                    int(tokens_used),
                    now_iso,
                ),
            )
            rows = conn.execute(
                "SELECT * FROM conversation_turns WHERE session_id = ? ORDER BY turn_id ASC",
                (session_id,),
            ).fetchall()

            if len(rows) > max_window_turns:
                older = rows[:-max_window_turns]
                summary_bullets = [
                    f"Turno {r['turn_id']} ({r['user_role']}): '{r['user_message'][:60]}' -> tools={r['tools_used']}"
                    for r in older[-8:]
                ]
                comp_text = " | ".join(summary_bullets)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO session_summaries VALUES (?, ?, ?, ?, ?)
                    """,
                    (session_id, plant_id, line_id, comp_text, now_iso),
                )
            conn.commit()

    def get_bounded_conversation_context(
        self,
        plant_id: str,
        line_id: str,
        session_id: str,
        max_recent_turns: int = 4,
    ) -> Dict[str, Any]:
        """Recupera únicamente la ventana acotada + resumen comprimido para controlar el contexto del LLM."""
        self.validate_tenant_isolation(plant_id, line_id)
        with self._connect() as conn:
            sum_row = conn.execute(
                "SELECT compressed_summary FROM session_summaries WHERE session_id = ? AND plant_id = ? AND line_id = ?",
                (session_id, plant_id, line_id),
            ).fetchone()
            recent_rows = conn.execute(
                """
                SELECT user_role, user_message, agent_response, tools_used, created_at_utc
                FROM conversation_turns
                WHERE session_id = ? AND plant_id = ? AND line_id = ?
                ORDER BY turn_id DESC LIMIT ?
                """,
                (session_id, plant_id, line_id, max_recent_turns),
            ).fetchall()

        return {
            "session_id": session_id,
            "compressed_older_history": sum_row["compressed_summary"] if sum_row else "",
            "recent_turns": [dict(r) for r in reversed(recent_rows)],
        }

    def log_tool_audit(
        self,
        request_id: str,
        session_id: str,
        plant_id: str,
        line_id: str,
        user_role: str,
        tool_name: str,
        arguments_redacted: Dict[str, Any],
        outcome_status: str,
        result_summary: str,
    ) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO audit_log
                (request_id, session_id, plant_id, line_id, user_role, tool_name, arguments_redacted_json, outcome_status, result_summary, created_at_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    session_id,
                    plant_id,
                    line_id,
                    user_role,
                    tool_name,
                    json.dumps(arguments_redacted, ensure_ascii=False),
                    outcome_status,
                    result_summary[:500],
                    now_iso,
                ),
            )
            conn.commit()


state_store = OperationalStateStore()
