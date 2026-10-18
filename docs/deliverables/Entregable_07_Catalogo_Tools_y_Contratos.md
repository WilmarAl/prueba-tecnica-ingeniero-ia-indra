# Entregable #7 — Patrón Agéntico, Descomposición y Catálogo de Tools y Contratos (Reto 3)

**Cargo:** AI Engineer (Senior+) — Indra Group  
**Versión del Catálogo:** `1.0.0` (11 Tools: 6 de Lectura / 5 de Escritura)  
**Esquema JSON / OpenAPI Máquina:** [`docs/deliverables/tools_contracts_schema.json`](tools_contracts_schema.json)

---

## 1. Patrón Agéntico Seleccionado y Descomposición (Sección 6.2)

### 1.1 Patrón Elegido: `Guardrail Determinístico -> Supervisor Router -> 4 Subagentes Especializados por Dominio -> Ejecutor Gobernado con Compuerta HITL`
En un entorno industrial 24/5 donde conviven consultas analíticas de lectura con operaciones críticas sobre un brazo robótico y modelos en producción, implementamos una arquitectura compuesta por:
1. **Middleware Determinístico de Entrada (`security.py` + `state_store.py`):** Valida el aislamiento de planta/línea (`plant_id`, `line_id`), detecta intentos de *Prompt Injection* y verifica permisos RBAC antes de cualquier razonamiento del modelo.
2. **Orquestador Supervisor con Descomposición en 4 Subagentes Especialistas (`orchestrator.py`):**
   - `vision_and_quality_specialist`: Gobierna `inspect_and_predict_images`, `list_human_review_queue`, `submit_human_relabeling`.
   - `telemetry_and_shift_analyst`: Gobierna `query_shift_and_hourly_telemetry`, `generate_shift_handover_report`, `search_plant_sops_and_manuals`.
   - `maintenance_diagnostics_engineer`: Gobierna `diagnose_camera_and_plc_health`, `search_plant_sops_and_manuals`, `create_or_escalate_maintenance_ticket`.
   - `mlops_and_risk_governance_controller`: Gobierna `evaluate_new_sku_onboarding`, `request_critical_line_operation`, `confirm_critical_line_operation`.
3. **Compuerta Human-in-the-Loop (HITL) en Dos Fases:** Toda operación de riesgo se divide en dos tools distintas (`request_critical_line_operation` $\rightarrow$ genera estado `PENDING_HUMAN_APPROVAL` con `approval_id`; y `confirm_critical_line_operation` $\rightarrow$ exige aprobación explícita y rol de aprobador).

### 1.2 Evidencia de Qué Aporta la Separación en 4 Subagentes
- **Reducción del espacio de búsqueda de herramientas:** Cada subagente expone únicamente **3 tools** en su dominio en lugar de las 11 tools simultáneas, reduciendo el consumo de tokens de definición de esquemas en **~58%** por paso y elevando la precisión de selección de tool al **100%** en el benchmark de 32 casos.
- **Segregación de funciones (Least Privilege):** El subagente de telemetría no tiene acceso en su catálogo a tools de escritura sobre la línea ni de modificación de umbrales.

### 1.3 Comparación con Al Menos Dos Alternativas Descartadas

| Criterio | Alternativa Descartada 1: Agente Monolítico ReAct Libre (11 Tools en un solo prompt) | Alternativa Descartada 2: Chatbot RAG / Flujo Estático de Intención Fija (Formulario Conversacional) | **Arquitectura Seleccionada: Supervisor + Subagentes por Dominio + Guardrails Determinísticos + HITL** |
|---|---|---|---|
| **Costo por Ejecución** | Alto (~1,400+ tokens/turno por inyectar los 11 JSON Schemas en cada iteración ReAct). | Muy bajo (~250 tokens), pero incapaz de resolver tareas compuestas. | **Óptimo (~310 tokens promedio; \$0.00008 USD/turno en cloud o \$0.00 en Ollama local).** |
| **Latencia** | Alta y variable (múltiples rondas LLM para decidir cada paso). | Baja (< 300 ms), pero sin encadenamiento dinámico. | **Baja y acotada (~18 ms orquestación determinística; ~1.8 s con síntesis en Ollama `llama3.1:8b`).** |
| **Riesgo Operativo** | **Inaceptable en planta:** riesgo de alucinar argumentos en tools de escritura o caer en loops infinitos. | Medio: no puede diagnosticar correlaciones multi-fuente (telemetría + cámara + SOP + ticket). | **Mínimo y controlado:** RBAC, validación Pydantic, anti-loop por firma SHA-256 y bloqueo HITL obligatorio. |
| **Mantenibilidad** | Baja: cambiar una tool afecta el prompt global y degrada otras intenciones. | Baja: requiere programar árboles rígidos por cada pregunta nueva. | **Alta:** contratos modulares independientes por subagente y catálogo autogenerado. |

### 1.4 Decisiones que NUNCA Quedan en Manos del LLM (Implementadas de Forma Determinística)
1. **inferencia de visión y decisión de agarre del robot (`safe_for_robot_grasp`):** Calculada 100% por el modelo clásico del Reto 2 y la compuerta matemática ($\sigma_\theta \le 10^\circ$), jamás por un prompt.
2. **Control de permisos por rol (RBAC) y aislamiento multi-planta:** Evaluado en `verify_tool_permission` y `validate_tenant_isolation` antes de ejecutar cualquier handler.
3. **Ejecución directa de operaciones de riesgo:** El código de `request_critical_line_operation` jamás modifica el motor productivo; persiste una orden `PENDING_HUMAN_APPROVAL` y exige `confirm_critical_line_operation`.
4. **Criterios de Terminación y Límites de Ejecución:**
   - `max_steps = 5` pasos máximos por turno.
   - `token_budget = 4000` tokens máximos por turno.
   - **Corte de Loop Agéntico (`LoopAndBudgetGuard`):** Calcula el hash canónico `SHA256(tool_name + canonical_json_args)`. Si una llamada se repite dentro del mismo turno, se aborta inmediatamente con `AGENTIC_LOOP_DETECTED`.
   - **Parada Segura ante Fallo 5xx/Timeout del Servicio de Visión:** Si la tool de visión reporta `503` o `504`, el orquestador detiene el flujo normal, consulta el protocolo de contingencia `[MAN-NET-09, Sección 1.3]` y ordena `HOLD_ROBOT_AND_ESCALATE`.

---

## 2. Matriz Resumen del Catálogo de 11 Tools (6 Lectura / 5 Escritura)

| # | Nombre de la Tool | Tipo | Subagente Principal | Roles Autorizados (RBAC) | Idempotencia / Efectos Laterales |
|---|---|---|---|---|---|
| 1 | `inspect_and_predict_images` | **READ** | Visión y Calidad | `supervisor`, `quality`, `maintenance`, `ml_engineer`, `plant_manager` | Lectura/Inferencia; encolado de rechazos idempotente por `frame_id`. |
| 2 | `list_human_review_queue` | **READ** | Visión y Calidad | `supervisor`, `quality`, `ml_engineer`, `plant_manager` | 100% Lectura sin efectos laterales. |
| 3 | `submit_human_relabeling` | **WRITE** | Visión y Calidad | `supervisor`, `quality`, `ml_engineer` | **Idempotente** por clave `UNIQUE(case_id)` en `retraining_dataset`. |
| 4 | `query_shift_and_hourly_telemetry` | **READ** | Telemetría y Turnos | Todos los roles de planta | 100% Lectura sin efectos laterales. |
| 5 | `diagnose_camera_and_plc_health` | **READ** | Mantenimiento | `supervisor`, `maintenance`, `quality`, `ml_engineer`, `plant_manager` | Diagnóstico pasivo de sensores sin alterar línea. |
| 6 | `search_plant_sops_and_manuals` | **READ** | Transversal (RAG) | Todos los roles de planta | 100% Lectura documental con citas exactas. |
| 7 | `create_or_escalate_maintenance_ticket` | **WRITE** | Mantenimiento | `maintenance`, `supervisor`, `quality` | **Idempotente** por hash SHA-256 de `(planta, línea, estación, categoría, resumen)`. |
| 8 | `evaluate_new_sku_onboarding` | **WRITE** | Gobierno y MLOps | `ml_engineer`, `quality` | Evalúa en `Sandbox` aislado; no altera producción. |
| 9 | `request_critical_line_operation` | **WRITE (HITL)** | Gobierno y MLOps | `supervisor`, `ml_engineer`, `plant_manager`, `quality` | **Idempotente**; crea solicitud `PENDING_HUMAN_APPROVAL` sin tocar línea física. |
| 10 | `confirm_critical_line_operation` | **WRITE (HITL)** | Gobierno y MLOps | `supervisor`, `plant_manager`, `ml_engineer` | **Idempotente**; aplica el cambio una única vez tras validar token y rol aprobador. |
| 11 | `generate_shift_handover_report` | **READ** | Telemetría y Turnos | `supervisor`, `plant_manager` | Consolidación de lectura para acta de cambio de turno. |

---

## 3. Especificación Detallada de Contratos por Tool

### 3.1 `inspect_and_predict_images` (READ)
- **Casos de Uso Cubiertos:** 1. Inspección de imagen (Supervisor / Calidad)
- **Descripción Funcional:** Valida la calidad óptica (nitidez Laplaciana, brillo, proporción de objeto) de una o varias imágenes, invoca el modelo clásico del Reto 2 para identificar el SKU, estimar el ángulo en [0°, 360°) con incertidumbre σ_θ, recuperar las Top-K vistas similares como evidencia y explicar si es seguro que el robot actúe.
- **Permisos y Restricciones (RBAC Determinístico):** Roles autorizados: `maintenance`, `ml_engineer`, `plant_manager`, `quality`, `supervisor`
- **Contrato de Salida:** JSON con `inspected_count`, `safe_grasp_count`, `rejected_count` y lista `results` (frame_id, sku_code, class_confidence, angle_deg, angle_uncertainty_deg, quality_report, top_k_similar_evidence, robot_action, confidence_explanation).
- **Manejo de Fallas y Errores:** Si el servicio de visión falla (HTTP 503 o Timeout 504), retorna estado estructurado `VISION_SERVICE_ERROR_503` / `VISION_SERVICE_TIMEOUT_504` con fallback `HOLD_ROBOT_AND_ESCALATE` sin inventar predicciones.
- **Comportamiento ante Efectos Laterales Repetidos (Idempotencia):** Operación de lectura/inferencia. Si una imagen es rechazada, su encolamiento en `rejection_queue` es idempotente (`INSERT OR IGNORE` por hash de `frame_id`).
- **Información que Recibe el Modelo (In-Context):** Métricas numéricas (confianza, ángulo, incertidumbre, varianza Laplaciana), códigos SKU y metadatos compactos de las Top-K vistas similares.
- **Información Excluida del Contexto del Modelo (Out-of-Context):** Arreglos de píxeles RGB (128x128x3), cadenas Base64 de imágenes, vectores de características de 2,080 dimensiones y rutas absolutas del sistema de archivos local.
- **Registro para Auditoría:** Registra en `audit_log`: request_id, session_id, user_role, sample_ids/escenarios consultados, decisión tomada y latencia.

**Contrato JSON Schema de Entrada y Validación (Pydantic v2):**
```json
{
  "description": "Contrato de entrada para `inspect_and_predict_images`.",
  "properties": {
    "sample_ids": {
      "description": "Lista de sample_ids del catálogo COIL-100 a inspeccionar (ej. ['obj003_ang025']).",
      "items": {
        "type": "string"
      },
      "title": "Sample Ids",
      "type": "array"
    },
    "synthetic_scenarios": {
      "description": "Escenarios ópticos/adversos a evaluar (normal_known, ood_unknown, blurry_degraded, ambiguous_angle).",
      "items": {
        "enum": [
          "normal_known",
          "ood_unknown",
          "blurry_degraded",
          "dark_underexposed",
          "ambiguous_angle"
        ],
        "type": "string"
      },
      "title": "Synthetic Scenarios",
      "type": "array"
    },
    "top_k": {
      "default": 3,
      "description": "Cantidad de vistas similares a recuperar como evidencia.",
      "maximum": 10,
      "minimum": 1,
      "title": "Top K",
      "type": "integer"
    },
    "station_id": {
      "default": "STATION-01",
      "pattern": "^STATION-\\d{2}$",
      "title": "Station Id",
      "type": "string"
    },
    "simulate_vision_fault": {
      "anyOf": [
        {
          "enum": [
            "error_503",
            "timeout"
          ],
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Opcional para pruebas de resiliencia: simula caída 503 o timeout 504 del servicio de visión.",
      "title": "Simulate Vision Fault"
    }
  },
  "title": "InspectImagesArgs",
  "type": "object"
}
```

---

### 3.2 `list_human_review_queue` (READ)
- **Casos de Uso Cubiertos:** 2. Cola de revisión humana (Supervisor)
- **Descripción Funcional:** Lista los casos rechazados por baja confianza, objeto desconocido (OOD) o incertidumbre angular > 10° que esperan revisión humana, incluyendo su evidencia Top-K.
- **Permisos y Restricciones (RBAC Determinístico):** Roles autorizados: `ml_engineer`, `plant_manager`, `quality`, `supervisor`
- **Contrato de Salida:** JSON con `count`, `filter_status` y lista de `cases` con `top_k_evidence`.
- **Manejo de Fallas y Errores:** Valida aislamiento de planta/línea y rango de turno (1..3).
- **Comportamiento ante Efectos Laterales Repetidos (Idempotencia):** 100% Lectura sin efectos laterales.
- **Información que Recibe el Modelo (In-Context):** Metadatos del caso (`case_id`, `frame_id`, `candidate_sku`, `rejection_reason`, `top_k_evidence`).
- **Información Excluida del Contexto del Modelo (Out-of-Context):** Archivos binarios de las capturas originales.
- **Registro para Auditoría:** Registra filtro aplicado, rol y número de registros devueltos.

**Contrato JSON Schema de Entrada y Validación (Pydantic v2):**
```json
{
  "description": "Contrato de entrada para `list_human_review_queue`.",
  "properties": {
    "status": {
      "default": "PENDING_HUMAN_REVIEW",
      "enum": [
        "PENDING_HUMAN_REVIEW",
        "CONFIRMED_FOR_RETRAINING",
        "ALL"
      ],
      "title": "Status",
      "type": "string"
    },
    "shift_number": {
      "anyOf": [
        {
          "maximum": 3,
          "minimum": 1,
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Filtro opcional por turno (1, 2 o 3).",
      "title": "Shift Number"
    },
    "limit": {
      "default": 10,
      "maximum": 50,
      "minimum": 1,
      "title": "Limit",
      "type": "integer"
    }
  },
  "title": "ListReviewQueueArgs",
  "type": "object"
}
```

---

### 3.3 `submit_human_relabeling` (WRITE)
- **Casos de Uso Cubiertos:** 2. Cola de revisión humana (Supervisor / Calidad)
- **Descripción Funcional:** Registra la etiqueta SKU y el ángulo en [0°, 360°) confirmados por el revisor humano para un caso de la cola de rechazo y lo envía al dataset de reentrenamiento (`STAGING_FOR_RETRAINING`).
- **Permisos y Restricciones (RBAC Determinístico):** Roles autorizados: `ml_engineer`, `quality`, `supervisor`
- **Contrato de Salida:** JSON con `record_id`, `case_id`, `confirmed_sku`, `confirmed_angle_deg`, `dataset_stage` y bandera `idempotent_replay`.
- **Manejo de Fallas y Errores:** Rechaza con error de validación si `confirmed_angle_deg` no está en [0, 360) o si el `case_id` no existe.
- **Comportamiento ante Efectos Laterales Repetidos (Idempotencia):** Idempotente por restricción `UNIQUE(case_id)` en `retraining_dataset`. Si se invoca dos veces sobre el mismo `case_id`, devuelve el registro existente con `idempotent_replay=True` sin duplicar filas.
- **Información que Recibe el Modelo (In-Context):** Confirmación del registro (`record_id`, `case_id`, `dataset_stage`).
- **Información Excluida del Contexto del Modelo (Out-of-Context):** Credenciales de base de datos y rutas internas de almacenamiento de entrenamiento.
- **Registro para Auditoría:** Registra quién re-etiquetó qué `case_id`, con qué SKU/ángulo y marca de tiempo UTC.

**Contrato JSON Schema de Entrada y Validación (Pydantic v2):**
```json
{
  "description": "Contrato de entrada para `submit_human_relabeling`.",
  "properties": {
    "case_id": {
      "description": "ID del caso en la cola (ej. CASE-1001).",
      "pattern": "^CASE-[A-Z0-9]+$",
      "title": "Case Id",
      "type": "string"
    },
    "confirmed_sku": {
      "description": "SKU confirmado por el humano (ej. SKU-007).",
      "pattern": "^SKU-\\d{3}$",
      "title": "Confirmed Sku",
      "type": "string"
    },
    "confirmed_angle_deg": {
      "description": "Ángulo real confirmado por el humano en [0.0, 360.0).",
      "exclusiveMaximum": 360.0,
      "minimum": 0.0,
      "title": "Confirmed Angle Deg",
      "type": "number"
    },
    "reviewer_id": {
      "default": "sup_turno_02",
      "maxLength": 40,
      "minLength": 3,
      "title": "Reviewer Id",
      "type": "string"
    },
    "notes": {
      "default": "Confirmado visualmente contra Top-K de referencia.",
      "maxLength": 300,
      "title": "Notes",
      "type": "string"
    }
  },
  "required": [
    "case_id",
    "confirmed_sku",
    "confirmed_angle_deg"
  ],
  "title": "SubmitHumanRelabelingArgs",
  "type": "object"
}
```

---

### 3.4 `query_shift_and_hourly_telemetry` (READ)
- **Casos de Uso Cubiertos:** 3. Consulta operativa (Supervisor / Gerente de planta)
- **Descripción Funcional:** Consulta métricas operativas estructuradas por turno (1, 2, 3) y desglose horario (ej. 03:00 a.m.), correlacionando throughput, tasa de rechazos, salud óptica de cámara y versión de modelo.
- **Permisos y Restricciones (RBAC Determinístico):** Roles autorizados: `maintenance`, `ml_engineer`, `plant_manager`, `quality`, `supervisor`, `viewer`
- **Contrato de Salida:** JSON con `shifts`, `hourly_breakdown` y `active_vision_health`.
- **Manejo de Fallas y Errores:** Valida formato de hora `HH:MM`, turno `1..3` y aislamiento de planta.
- **Comportamiento ante Efectos Laterales Repetidos (Idempotencia):** 100% Lectura sin efectos laterales.
- **Información que Recibe el Modelo (In-Context):** Tablas agregadas de KPIs por turno y eventos horarios.
- **Información Excluida del Contexto del Modelo (Out-of-Context):** Logs crudos de bajo nivel del PLC y trazas de red.
- **Registro para Auditoría:** Registra la consulta de telemetría con turno y franja horaria solicitada.

**Contrato JSON Schema de Entrada y Validación (Pydantic v2):**
```json
{
  "description": "Contrato de entrada para `query_shift_and_hourly_telemetry`.",
  "properties": {
    "shift_number": {
      "anyOf": [
        {
          "maximum": 3,
          "minimum": 1,
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Turno a consultar (1, 2 o 3).",
      "title": "Shift Number"
    },
    "hour_slot": {
      "anyOf": [
        {
          "pattern": "^\\d{2}:\\d{2}$",
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Hora específica en formato HH:MM (ej. '03:00', '14:00').",
      "title": "Hour Slot"
    }
  },
  "title": "QueryTelemetryArgs",
  "type": "object"
}
```

---

### 3.5 `diagnose_camera_and_plc_health` (READ)
- **Casos de Uso Cubiertos:** 4. Diagnóstico de falla (Mantenimiento)
- **Descripción Funcional:** Ejecuta diagnóstico técnico sobre la estación de empaque: intensidad del anillo LED de cámara, nitidez Laplaciana, presión de vacío de la ventosa (-kPa), holgura del eje J6 del robot y enlace PLC.
- **Permisos y Restricciones (RBAC Determinístico):** Roles autorizados: `maintenance`, `ml_engineer`, `plant_manager`, `quality`, `supervisor`
- **Contrato de Salida:** JSON con estado de cámara, enlace PLC, vacío neumático, holgura mecánica J6 y hallazgos recientes.
- **Manejo de Fallas y Errores:** Valida patrón de estación `^STATION-\d{2}$`.
- **Comportamiento ante Efectos Laterales Repetidos (Idempotencia):** Lectura diagnóstica no intrusiva.
- **Información que Recibe el Modelo (In-Context):** Parámetros físicos medidos y comparación contra rangos nominales.
- **Información Excluida del Contexto del Modelo (Out-of-Context):** Direcciones IP privadas e identificadores de bus industrial Modbus.
- **Registro para Auditoría:** Registra estación diagnosticada y estado reportado.

**Contrato JSON Schema de Entrada y Validación (Pydantic v2):**
```json
{
  "description": "Contrato de entrada para `diagnose_camera_and_plc_health`.",
  "properties": {
    "station_id": {
      "default": "STATION-01",
      "pattern": "^STATION-\\d{2}$",
      "title": "Station Id",
      "type": "string"
    },
    "include_recent_grasp_failures": {
      "default": true,
      "title": "Include Recent Grasp Failures",
      "type": "boolean"
    }
  },
  "title": "DiagnoseHardwareArgs",
  "type": "object"
}
```

---

### 3.6 `search_plant_sops_and_manuals` (READ)
- **Casos de Uso Cubiertos:** 4. Diagnóstico de falla (Mantenimiento); 5. Alta de SKU; 6. Operación de riesgo
- **Descripción Funcional:** Busca en los procedimientos operativos estándar (SOPs) y manuales técnicos de planta, devolviendo fragmentos relevantes con citas exactas obligatorias `[DOC-ID, Sección]`.
- **Permisos y Restricciones (RBAC Determinístico):** Roles autorizados: `maintenance`, `ml_engineer`, `plant_manager`, `quality`, `supervisor`, `viewer`
- **Contrato de Salida:** JSON con lista `documents` incluyendo `doc_id`, `title`, `section`, `citation` y `excerpt`.
- **Manejo de Fallas y Errores:** Rechaza consultas vacías o menores a 3 caracteres.
- **Comportamiento ante Efectos Laterales Repetidos (Idempotencia):** 100% Lectura RAG determinística.
- **Información que Recibe el Modelo (In-Context):** Únicamente los Top-K fragmentos relevantes y sus citas oficiales.
- **Información Excluida del Contexto del Modelo (Out-of-Context):** Documentos completos no relevantes para evitar inflar el contexto.
- **Registro para Auditoría:** Registra la consulta y los IDs de documentos recuperados.

**Contrato JSON Schema de Entrada y Validación (Pydantic v2):**
```json
{
  "description": "Contrato de entrada para `search_plant_sops_and_manuals`.",
  "properties": {
    "query": {
      "description": "Consulta técnica sobre fallas, cámara, robot o SKUs.",
      "maxLength": 200,
      "minLength": 3,
      "title": "Query",
      "type": "string"
    },
    "top_k_docs": {
      "default": 2,
      "maximum": 5,
      "minimum": 1,
      "title": "Top K Docs",
      "type": "integer"
    }
  },
  "required": [
    "query"
  ],
  "title": "SearchSopsArgs",
  "type": "object"
}
```

---

### 3.7 `create_or_escalate_maintenance_ticket` (WRITE)
- **Casos de Uso Cubiertos:** 4. Diagnóstico de falla (Mantenimiento / Supervisor)
- **Descripción Funcional:** Crea o escala un ticket de mantenimiento o calidad adjuntando el diagnóstico técnico, la estación afectada, la severidad y las citas obligatorias del SOP consultado.
- **Permisos y Restricciones (RBAC Determinístico):** Roles autorizados: `maintenance`, `quality`, `supervisor`
- **Contrato de Salida:** JSON con `ticket_id`, `status`, `priority`, `sop_citations` e `idempotent_replay`.
- **Manejo de Fallas y Errores:** Exige al menos una cita de SOP y restringe el uso a roles autorizados (maintenance, supervisor, calidad).
- **Comportamiento ante Efectos Laterales Repetidos (Idempotencia):** Idempotente mediante hash SHA-256 (`idempotency_key`) de `(plant_id, line_id, station_id, category, summary)`. Llamadas repetidas devuelven el mismo `ticket_id` sin duplicar tickets.
- **Información que Recibe el Modelo (In-Context):** ID del ticket creado, estado y citas asociadas.
- **Información Excluida del Contexto del Modelo (Out-of-Context):** Claves internas de integración con sistemas ERP/SAP.
- **Registro para Auditoría:** Registra creación/escalamiento del ticket con rol y severidad.

**Contrato JSON Schema de Entrada y Validación (Pydantic v2):**
```json
{
  "description": "Contrato de entrada para `create_or_escalate_maintenance_ticket`.",
  "properties": {
    "station_id": {
      "default": "STATION-01",
      "pattern": "^STATION-\\d{2}$",
      "title": "Station Id",
      "type": "string"
    },
    "priority": {
      "default": "HIGH",
      "enum": [
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL"
      ],
      "title": "Priority",
      "type": "string"
    },
    "issue_category": {
      "enum": [
        "OPTICAL_LED_DRIFT",
        "ROBOT_GRIPPER_VACUUM_OR_ALIGNMENT",
        "VISION_SERVICE_OUTAGE_5XX",
        "CAMERA_BLUR_OR_CONTAMINATION",
        "OTHER_PLANT_FAULT"
      ],
      "title": "Issue Category",
      "type": "string"
    },
    "diagnosis_summary": {
      "maxLength": 500,
      "minLength": 10,
      "title": "Diagnosis Summary",
      "type": "string"
    },
    "sop_citations": {
      "description": "Citas obligatorias de SOP/manuales (ej. ['[SOP-ROB-11, Sección 4.1]']).",
      "items": {
        "type": "string"
      },
      "minItems": 1,
      "title": "Sop Citations",
      "type": "array"
    }
  },
  "required": [
    "issue_category",
    "diagnosis_summary",
    "sop_citations"
  ],
  "title": "CreateTicketArgs",
  "type": "object"
}
```

---

### 3.8 `evaluate_new_sku_onboarding` (WRITE)
- **Casos de Uso Cubiertos:** 5. Alta de producto nuevo SKU (Ingeniero de ML / Calidad)
- **Descripción Funcional:** Recibe un conjunto de vistas de un producto nuevo (SKU), valida que el material sea suficiente (mínimo 12 vistas, sin sectores ciegos > 35°), evalúa su desempeño en un entorno controlado (Sandbox) y, si cumple criterios (Accuracy >= 95%, MAE <= 10°), genera la solicitud de aprobación humana para pasarlo a producción.
- **Permisos y Restricciones (RBAC Determinístico):** Roles autorizados: `ml_engineer`, `quality`
- **Contrato de Salida:** JSON con `sandbox_id`, `material_sufficient`, `views_evaluated`, `max_angular_gap_deg`, `sandbox_cv_classification_accuracy`, `sandbox_cv_angle_mae_deg`, `meets_production_criteria` y `production_promotion_request`.
- **Manejo de Fallas y Errores:** Si las vistas son insuficientes (< 12 o con sector ciego), bloquea el paso a producción y detalla las causas.
- **Comportamiento ante Efectos Laterales Repetidos (Idempotencia):** Evalúa en memoria aislada (`Sandbox`) sin modificar el artefacto productivo activo y registra el resultado en `sku_sandbox_registry`.
- **Información que Recibe el Modelo (In-Context):** Métricas agregadas de validación en sandbox y estado de la solicitud de aprobación.
- **Información Excluida del Contexto del Modelo (Out-of-Context):** Matrices de embeddings (N x 2080) y archivos de imágenes del nuevo SKU.
- **Registro para Auditoría:** Registra el `sandbox_id`, SKU evaluado, métricas obtenidas y usuario solicitante.

**Contrato JSON Schema de Entrada y Validación (Pydantic v2):**
```json
{
  "description": "Contrato de entrada para `evaluate_new_sku_onboarding`.",
  "properties": {
    "sku_code": {
      "description": "Código del nuevo SKU (ej. SKU-105).",
      "pattern": "^SKU-\\d{3}$",
      "title": "Sku Code",
      "type": "string"
    },
    "product_name": {
      "maxLength": 80,
      "minLength": 3,
      "title": "Product Name",
      "type": "string"
    },
    "reference_object_id": {
      "default": 19,
      "maximum": 100,
      "minimum": 1,
      "title": "Reference Object Id",
      "type": "integer"
    },
    "angles_provided_deg": {
      "anyOf": [
        {
          "items": {
            "type": "integer"
          },
          "type": "array"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Ángulos de las vistas suministradas. Si se omite, evalúa paquete estándar de 18 vistas.",
      "title": "Angles Provided Deg"
    }
  },
  "required": [
    "sku_code",
    "product_name"
  ],
  "title": "EvaluateNewSkuArgs",
  "type": "object"
}
```

---

### 3.9 `request_critical_line_operation` (WRITE)
- **Casos de Uso Cubiertos:** 6. Operación de riesgo (Cualquiera autorizado con aprobación humana)
- **Descripción Funcional:** Registra una solicitud para ejecutar una operación de riesgo (cambiar el umbral de confianza, pausar o reanudar la línea, enviar un ángulo manual al robot o promover una versión de modelo). Por diseño determinístico, NUNCA aplica el cambio de inmediato: crea una orden `PENDING_HUMAN_APPROVAL`.
- **Permisos y Restricciones (RBAC Determinístico):** Roles autorizados: `ml_engineer`, `plant_manager`, `quality`, `supervisor`
- **Contrato de Salida:** JSON con `approval_id`, `operation_type`, `parameters`, `status='PENDING_HUMAN_APPROVAL'`.
- **Manejo de Fallas y Errores:** Valida rangos seguros de umbral [0.40, 0.98] y ángulo [0, 360), además de permisos RBAC.
- **Comportamiento ante Efectos Laterales Repetidos (Idempotencia):** Idempotente sobre solicitudes abiertas con los mismos parámetros (`idempotency_key`). No altera la línea física hasta que se invoque `confirm_critical_line_operation`.
- **Información que Recibe el Modelo (In-Context):** `approval_id`, operación solicitada y parámetros.
- **Información Excluida del Contexto del Modelo (Out-of-Context):** Firmas internas de control del PLC.
- **Registro para Auditoría:** Registra toda solicitud de cambio de riesgo en `audit_log` y `risk_operation_approvals`.

**Contrato JSON Schema de Entrada y Validación (Pydantic v2):**
```json
{
  "description": "Contrato de entrada para `request_critical_line_operation`.",
  "properties": {
    "operation_type": {
      "enum": [
        "UPDATE_CONFIDENCE_THRESHOLD",
        "PAUSE_LINE",
        "RESUME_LINE",
        "SEND_MANUAL_ROBOT_ANGLE",
        "PROMOTE_MODEL_OR_SKU_TO_PRODUCTION"
      ],
      "title": "Operation Type",
      "type": "string"
    },
    "new_confidence_threshold": {
      "anyOf": [
        {
          "maximum": 0.98,
          "minimum": 0.4,
          "type": "number"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "New Confidence Threshold"
    },
    "manual_angle_deg": {
      "anyOf": [
        {
          "exclusiveMaximum": 360.0,
          "minimum": 0.0,
          "type": "number"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Manual Angle Deg"
    },
    "target_model_or_sku": {
      "anyOf": [
        {
          "maxLength": 60,
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Target Model Or Sku"
    },
    "reason": {
      "description": "Justificación técnica y operativa.",
      "maxLength": 300,
      "minLength": 8,
      "title": "Reason",
      "type": "string"
    }
  },
  "required": [
    "operation_type",
    "reason"
  ],
  "title": "RequestRiskOperationArgs",
  "type": "object"
}
```

---

### 3.10 `confirm_critical_line_operation` (WRITE)
- **Casos de Uso Cubiertos:** 6. Operación de riesgo — Confirmación Humana Explícita (HITL)
- **Descripción Funcional:** Ejecuta o rechaza una operación crítica previamente solicitada tras verificar el `approval_id`, el rol autorizado del aprobador humano (`supervisor`, `plant_manager`, `ml_engineer`) y la justificación.
- **Permisos y Restricciones (RBAC Determinístico):** Roles autorizados: `ml_engineer`, `plant_manager`, `supervisor`
- **Contrato de Salida:** JSON con `approval_id`, `status` (`APPROVED_AND_EXECUTED` o `REJECTED_BY_HUMAN`), `approved_by` y `decided_at_utc`.
- **Manejo de Fallas y Errores:** Rechaza si el `approval_id` no existe o si el rol del usuario no tiene privilegios de aprobador.
- **Comportamiento ante Efectos Laterales Repetidos (Idempotencia):** Idempotente: una vez decidida una solicitud (`APPROVED_AND_EXECUTED` / `REJECTED_BY_HUMAN`), llamadas posteriores no vuelven a ejecutar el efecto lateral y retornan `idempotent_replay=True`.
- **Información que Recibe el Modelo (In-Context):** Estado final de la aprobación y marca de tiempo.
- **Información Excluida del Contexto del Modelo (Out-of-Context):** Tokens internos de sesión.
- **Registro para Auditoría:** Firma inmutable en `audit_log` con aprobador, rol, notas y timestamp UTC.

**Contrato JSON Schema de Entrada y Validación (Pydantic v2):**
```json
{
  "description": "Contrato de entrada para `confirm_critical_line_operation`.",
  "properties": {
    "approval_id": {
      "description": "ID de aprobación pendiente (ej. APR-9042).",
      "pattern": "^APR-[A-Z0-9]+$",
      "title": "Approval Id",
      "type": "string"
    },
    "approved": {
      "description": "True para aprobar y ejecutar; False para rechazar.",
      "title": "Approved",
      "type": "boolean"
    },
    "approved_by": {
      "description": "Nombre/ID del aprobador humano.",
      "maxLength": 50,
      "minLength": 3,
      "title": "Approved By",
      "type": "string"
    },
    "decision_notes": {
      "description": "Comentario de auditoría del aprobador.",
      "maxLength": 300,
      "minLength": 5,
      "title": "Decision Notes",
      "type": "string"
    }
  },
  "required": [
    "approval_id",
    "approved",
    "approved_by",
    "decision_notes"
  ],
  "title": "ConfirmRiskOperationArgs",
  "type": "object"
}
```

---

### 3.11 `generate_shift_handover_report` (READ)
- **Casos de Uso Cubiertos:** 7. Entrega de turno (Supervisor)
- **Descripción Funcional:** Genera el reporte estructurado de cambio de turno consolidando métricas del turno, incidentes horarios, tickets de mantenimiento abiertos, casos pendientes en la cola humana y aprobaciones pendientes.
- **Permisos y Restricciones (RBAC Determinístico):** Roles autorizados: `plant_manager`, `supervisor`
- **Contrato de Salida:** JSON con `handover_id`, `shift_metrics`, `open_maintenance_tickets`, `pending_human_review_cases` y `pending_risk_approvals`.
- **Manejo de Fallas y Errores:** Restringido por RBAC a `supervisor` y `plant_manager`.
- **Comportamiento ante Efectos Laterales Repetidos (Idempotencia):** Lectura consolidada determinística.
- **Información que Recibe el Modelo (In-Context):** Resumen ejecutivo estructurado del turno saliente.
- **Información Excluida del Contexto del Modelo (Out-of-Context):** Datos personales de operarios individuales.
- **Registro para Auditoría:** Registra generación del acta de entrega de turno.

**Contrato JSON Schema de Entrada y Validación (Pydantic v2):**
```json
{
  "description": "Contrato de entrada para `generate_shift_handover_report`.",
  "properties": {
    "shift_number": {
      "default": 2,
      "description": "Número de turno que entrega (1, 2 o 3).",
      "maximum": 3,
      "minimum": 1,
      "title": "Shift Number",
      "type": "integer"
    }
  },
  "title": "GenerateHandoverArgs",
  "type": "object"
}
```

