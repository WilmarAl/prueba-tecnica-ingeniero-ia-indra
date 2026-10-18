# =============================================================================
# Arquitectura y Diagramas de Secuencia — Sistema IA Pick & Pack (Indra Group)
# =============================================================================

## 1. Principio de Diseño: Separación entre Camino Crítico Determinístico y Plano Agéntico

El problema industrial exige **alta cadencia (throughput de línea 24/5)**, **seguridad física para el brazo robótico (tolerancia de agarre $\pm 10^\circ$)** y **gobernanza operativa**.

| Dimensión | Camino Crítico en Línea (Visión + Compuerta Robot) | Plano de Control y Excepciones (Asistente Agéntico) |
|---|---|---|
| **Naturaleza** | **100% Determinístico** (ML Clásico + Estadística Circular + Guardrail Matemático) | **Agéntico Gobernado** (Orquestador + Tools tipadas + RBAC + Human-in-the-Loop) |
| **Ubicación** | **Edge / Planta Local** (Acoplado a cámara industrial y controlador PLC/Robot) | **Out-of-Band (Plano de Supervisión)** (Sin bloquear el ciclo de agarre) |
| **Latencia** | **< 25 ms** en CPU estándar (determinista, sin red externa) | **0.5 s – 4.0 s** (Consultas interactivas de supervisores, calidad, mantenimiento, ML) |
| **Costo Marginal** | **~\$0.00 por imagen** (Cómputo local amortizado) | **Bajo demanda** (Solo se invoca ante excepciones, diagnósticos o auditorías) |
| **Riesgo Mitigado** | Evita alucinaciones de LLMs, variabilidad de red y paradas no programadas en el brazo robótico | Evita acciones destructivas mediante permisos estrictos (RBAC) y aprobación humana explícita (HITL) |

---

## 2. Diagrama de Arquitectura General del Sistema

```mermaid
flowchart TB
    subgraph Planta["PLANTA INDUSTRIAL — Camino Crítico en Tiempo Real (Determinístico < 25ms)"]
        CAM["Cámara RGB Cenital\n(128x128 px)"] --> V_API["API Servicio de Visión\n(FastAPI Independiente - Puerto 8000)"]
        
        subgraph PipelineML["Motor ML Clásico Versionado (v1.0.0)"]
            FEAT["Extracción de Características\n(HOG + HSV Spatial Hist + LBP/Momentos)"]
            CLS["Clasificador Calibrado + Detector OOD\n(Probabilidad + Distancia en Embedding)"]
            ANG["Regresor Angular Circular\n(sin θ, cos θ -> atan2 + Incertidumbre σ_θ)"]
             SIM["Índice Vectorial Top-K\n(Recuperación de Vistas Similares)"]
            FEAT --> CLS & ANG & SIM
        end

        V_API --> PipelineML
        CLS & ANG --> GATE{"Compuerta Determinística\n¿Objeto Conocido?\n¿Confianza >= τ_cls?\n¿Error Angular Esperado <= 10°?"}
        GATE -->|"SÍ: Seguro Actuar"| PLC["Controlador Brazo Robótico\n(Comando PICK + Rotación Δθ)"]
        GATE -->|"NO: Rechazo / Incertidumbre Alta / OOD"| DIVERT["Carril de Desvío Seguro\n+ Registro en Cola de Excepción"]
    end

    subgraph PlanoControl["PLANO OPERATIVO Y DE GOBIERNO — Copiloto Agéntico (Puerto 8001)"]
        ACTORS["Usuarios de Planta:\n- Supervisor de Turno\n- Calidad\n- Mantenimiento\n- Ingeniero ML"] --> UI["Consola Web / API Conversacional"]
        UI --> GUARD_IN["Guardrail de Entrada:\nRBAC + Sanitización + Control de Contexto"]
        GUARD_IN --> ORCH["Orquestador Agéntico\n(Supervisor + Router + Límite de Pasos/Loops)"]
        
        ORCH --> T_READ["Tools de Lectura (Inspección, Cola,\nTelemetría, RAG SOPs, Salud Cámara, Reporte Turno)"]
        ORCH --> T_WRITE["Tools de Escritura y Riesgo (Re-etiquetado,\nTickets, Sandbox Nuevo SKU, Operaciones Críticas HITL)"]
        
        T_READ & T_WRITE <-->|"HTTP / REST"| V_API
        T_READ & T_WRITE <--> DB[("Base Operativa & Auditoría\n(SQLite: Sesiones, Cola, Tickets, Métricas, Audit Log)")]
        T_READ <--> KB[("Base de Conocimiento\n(SOPs, Manuales con Citas)")]
    end

    DIVERT -.->|"Alimenta asíncronamente"| DB
```

---

## 3. Diagrama de Secuencia 1: Flujo Crítico `Imagen -> Robot` (Determinístico)

```mermaid
sequenceDiagram
    autonumber
    participant CAM as Cámara Línea
    participant VIS as API Servicio Visión (/predict)
    participant EXT as Extractor Clásico (HOG+Color+LBP)
    participant MDL as Clasificador + Regresor Circular + OOD
    participant GATE as Guardrail Físico (±10°)
    participant ROB as Brazo Robótico (PLC)
    participant DB as Cola de Revisión / Telemetría

    CAM->>VIS: POST /api/v1/predict (Imagen RGB 128x128, station_id, frame_id)
    VIS->>VIS: Validar integridad y calidad óptica (Varianza Laplaciano, Exposición)
    alt Imagen degradada (Blur severo / Subexposición)
        VIS-->>ROB: decision="REJECT_IMAGE_QUALITY", safe_to_act=false, action="DIVERT"
        VIS->>DB: Registrar incidente óptico y encolar frame
    else Calidad óptica válida
        VIS->>EXT: Extraer vector de características x ∈ R^d
        EXT-->>VIS: Vector normalizado + proyección PCA
        VIS->>MDL: Inferir Clase, Distancia OOD, (sin θ, cos θ) y Top-K similares
        MDL-->>VIS: class_id, conf_cls, ood_score, angle_deg, angle_uncertainty_deg, top_k
        VIS->>GATE: Evaluar reglas determinísticas (ood_score <= τ_ood, conf_cls >= τ_cls, σ_θ <= 10°)
        alt Cumple todos los criterios de seguridad
            GATE-->>VIS: safe_to_act=true, robot_command="PICK_AND_ROTATE"
            VIS-->>ROB: HTTP 200 {safe_to_act: true, class_id, angle_deg, uncertainty_deg}
            VIS->>DB: Registrar telemetría exitosa (latencia, confianza)
        else Caso no confiable o Objeto Desconocido (OOD)
            GATE-->>VIS: safe_to_act=false, robot_command="DIVERT_TO_INSPECTION"
            VIS-->>ROB: HTTP 200 {safe_to_act: false, reason, fallback_action: "DIVERT_TO_INSPECTION"}
            VIS->>DB: Insertar caso en Cola de Revisión Humana con evidencia Top-K
        end
    end
```

---

## 4. Diagrama de Secuencia 2: Flujo Agéntico Complejo con Aprobación Humana (HITL)

Caso ilustrado: **Diagnóstico de incremento de rechazos + Solicitud de Operación de Riesgo (Ajuste de umbral / Envío de comando o Promoción de modelo) con Aprobación Humana Explícita**.

```mermaid
sequenceDiagram
    autonumber
    actor SUP as Supervisor / Ing. ML
    participant API as API Agente (/chat)
    participant RBAC as Guardrail RBAC & Loop Detector
    participant ORCH as Orquestador Agéntico
    participant TOOLS as Catálogo de Tools (11 Tools)
    participant DB as Estado Operativo & Audit Log

    SUP->>API: "¿Por qué subieron los rechazos a las 3 a.m. en el turno 2 y baja el umbral temporalmente a 0.65?"
    API->>RBAC: Verificar sesión, aislamiento (plant_id, line_id) y rol del usuario
    RBAC->>ORCH: Contexto validado (presupuesto max_steps=5)
    
    ORCH->>TOOLS: 1. Call `query_line_telemetry(shift=2, hour="03:00")`
    TOOLS->>DB: Consultar métricas e incidentes por hora
    DB-->>TOOLS: Datos estructurados (pico de rechazos por deriva de iluminación LED en cámara)
    TOOLS-->>ORCH: Resultado estructurado
    
    ORCH->>TOOLS: 2. Call `search_maintenance_sops(query="deriva iluminación cámara rechazos")`
    TOOLS-->>ORCH: Fragmentos SOP con cita exacta `[SOP-OPT-04, Sec 3.2]`
    
    ORCH->>TOOLS: 3. Call `request_risk_operation(operation="update_confidence_threshold", new_value=0.65)`
    Note over TOOLS,DB: Regla Determinística: Ninguna operación de riesgo se ejecuta sin aprobación humana explícita
    TOOLS->>DB: Crear registro `PENDING_APPROVAL` (approval_id="APR-9042") + Audit Log
    TOOLS-->>ORCH: Estado `REQUIRES_HUMAN_APPROVAL` (approval_id="APR-9042", impacto estimado)
    
    ORCH-->>SUP: Diagnóstico con datos y cita [SOP-OPT-04] + Solicitud de confirmación explícita para `APR-9042`
    
    SUP->>API: "Aprobar operación APR-9042 con justificación: contingencia controlada mientras mantenimiento calibra LED"
    API->>RBAC: Validar permiso de aprobador (Rol Supervisor/ML Lead)
    RBAC->>TOOLS: 4. Call `confirm_risk_operation(approval_id="APR-9042", approved=true, justification="...")`
    TOOLS->>DB: Aplicar cambio gobernado + Registrar firma en Audit Log inmutable
    TOOLS-->>SUP: Confirmación de ejecución auditable
```

---

## 5. Estructura del Repositorio

```text
prueba-tecnica-ingeniero-ia-indra/
├── .env.example                       # Configuración parametrizada sin secretos (Reto 4)
├── Dockerfile                         # Empaquetado reproducible de servicios
├── docker-compose.yml                 # Orquestación local de 1 comando (Visión + Agente)
├── pyproject.toml                     # Definición del paquete y configuración de tests
├── requirements.txt                   # Dependencias fijadas
├── README.md                          # Mapa maestro de entregables y guía de ejecución rápida
├── data/
│   ├── raw/                           # Dataset COIL-100 descargado automáticamente
│   ├── processed/                     # Particiones estrictas (Protocolos A, B, C, D) y features
│   └── knowledge_base/                # SOPs, manuales técnicos y procedimientos de planta
├── models/                            # Artefactos versionados (.joblib) y Model Card (.json/.md)
├── notebooks/                         # Entregable #2: Notebook reproducible de punta a punta (.ipynb)
├── src/
│   ├── config.py                      # Configuración centralizada (Pydantic Settings)
│   ├── ml/                            # Reto 2: Extracción clásica, clasificadores, regresor circular, OOD
│   ├── vision_service/                # Reto 4: API FastAPI independiente de inferencia de visión
│   ├── agent_service/                 # Reto 3: Orquestador agéntico, estado, memoria, RBAC y seguridad
│   │   └── tools/                     # Catálogo de 11 tools de lectura/escritura con contratos estrictos
│   └── ui/                            # Consola interactiva para defensa técnica en vivo
├── scripts/                           # Scripts reproducibles de entrenamiento, evaluación (30+ casos) y PDFs
├── tests/                             # Suite de tests unitarios, protocolos y 8 escenarios adversos
└── docs/
    ├── architecture/                  # Diagramas de arquitectura y secuencia (Reto 4)
    └── deliverables/                  # Entregables oficiales (#3 Reporte PDF, #7 Tools, #8 Threat Model, #12 Azure FinOps)
```
