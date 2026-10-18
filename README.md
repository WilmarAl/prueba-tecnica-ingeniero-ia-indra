# Sistema de Visión Clásica y Copiloto Agéntico para Línea de Pick & Pack Robótico — Indra Group

Solución integral de extremo a extremo para la **Prueba Técnica de AI Engineer (Senior+)** de **Indra Group**. Sustituye la tarea manual de identificación de producto y alineación angular en una línea industrial de empaque (24/5, 3 turnos) mediante un **motor determinístico de Visión por Computador y ML Clásico** acoplado a un **Copiloto Agéntico de Operación y Gobierno**.

---

## Mapa de Entregables Obligatorios (Sección 9 de la Prueba)

Todos los entregables están centralizados en este repositorio bajo una estructura trazable:

| # | Entregable Oficial | Reto | Ubicación en el Repositorio | Formato |
|---|---|---|---|---|
| **2** | **Notebook de experimentación reproducible** | Reto 2 | [`notebooks/01_coil100_classical_ml_experiments.ipynb`](notebooks/01_coil100_classical_ml_experiments.ipynb) | `.ipynb` ejecutable de punta a punta |
| **3** | **Reporte de resultados por protocolo, análisis de errores y Model Card** | Reto 2 | [`docs/deliverables/Entregable_03_Reporte_Resultados_y_Model_Card.pdf`](docs/deliverables/Entregable_03_Reporte_Resultados_y_Model_Card.pdf) *(y fuente en `.md`)* | `PDF` + `Markdown` |
| **5** | **Repositorio de código con historial de commits razonable** | Retos 2, 3, 4 | Raíz del repositorio Git (`git log --oneline`) | `Git` |
| **7** | **Catálogo de tools y contratos** | Reto 3 | [`docs/deliverables/Entregable_07_Catalogo_Tools_y_Contratos.md`](docs/deliverables/Entregable_07_Catalogo_Tools_y_Contratos.md) + [`docs/deliverables/tools_contracts_schema.json`](docs/deliverables/tools_contracts_schema.json) | `Markdown` + `JSON Schema / OpenAPI` |
| **8** | **Threat model y controles + Evaluación de 30+ casos** | Reto 3 | [`docs/deliverables/Entregable_08_Threat_Model_y_Controles.pdf`](docs/deliverables/Entregable_08_Threat_Model_y_Controles.pdf) *(y fuente en `.md`)* | `PDF` + `Markdown` |
| **10** | **Servicio desplegado: Despliegue local de un comando + Diagramas** | Reto 4 | [`docker-compose.yml`](docker-compose.yml) / [`scripts/run_mvp.ps1`](scripts/run_mvp.ps1) + [`docs/architecture/architecture_and_sequences.md`](docs/architecture/architecture_and_sequences.md) | `Docker / Script 1-Comando` + `README` |
| **12** | **Arquitecturas Cloud (Azure vs. AWS) + Modelo de Costos FinOps** | Reto 6 | [`docs/deliverables/Entregable_12_Arquitectura_Cloud_Azure_FinOps.pdf`](docs/deliverables/Entregable_12_Arquitectura_Cloud_Azure_FinOps.pdf) + [`docs/deliverables/Entregable_12_Modelo_Costos_FinOps_Azure.xlsx`](docs/deliverables/Entregable_12_Modelo_Costos_FinOps_Azure.xlsx) | `Documento PDF/MD` + `Hoja de cálculo .xlsx` |

---

## Principio de Diseño: ¿Qué es Determinístico y Qué es Agéntico?

1. **Camino Crítico en Planta (`Imagen -> Modelo Clásico -> Brazo Robótico`): 100% Determinístico (< 25 ms, Costo Marginal ~\$0, Riesgo Acotado).**
   - Un LLM nunca debe decidir en tiempo real el ángulo de giro de un brazo industrial físico.
   - El servicio de visión (`src/vision_service/`) extrae descriptores clásicos (**HOG + Histogramas Espaciales HSV/Lab + LBP/Momentos**), predice la clase con probabilidades calibradas, estima la orientación circular mediante regresión trigonométrica $(\sin\theta, \cos\theta) \rightarrow \operatorname{atan2}$ junto con su incertidumbre angular $\sigma_\theta$, y evalúa una **compuerta de seguridad determinística** atada a la tolerancia mecánica del robot ($\pm 10^\circ$) y a un detector Out-of-Distribution (Protocolo D).
2. **Plano de Supervisión, Excepciones y Gobierno (`Copiloto Operativo`): Arquitectura Agéntica Gobernada.**
   - Los supervisores de turno, calidad, mantenimiento e ingeniería de ML interactúan con el **Asistente Agéntico (`src/agent_service/`)** para inspeccionar casos dudosos, auditar la cola de rechazo humano, correlacionar métricas de turno con fallas ópticas, consultar SOPs con citas exactas, dar de alta nuevos SKUs en sandbox y ejecutar operaciones de riesgo bajo **permisos RBAC y aprobación humana explícita (Human-in-the-Loop)**.

Consulte el detalle arquitectónico y los diagramas de secuencia en [`docs/architecture/architecture_and_sequences.md`](docs/architecture/architecture_and_sequences.md).

---

## Arquitectura del Proyecto

```text
prueba-tecnica-ingeniero-ia-indra/
├── .env.example                       # Variables de entorno documentadas sin secretos
├── Dockerfile                         # Imagen base reproducible para los servicios
├── docker-compose.yml                 # Despliegue orquestado en un solo comando
├── pyproject.toml                     # Configuración del paquete Python y pytest
├── requirements.txt                   # Dependencias del proyecto
├── data/
│   ├── raw/                           # Dataset COIL-100 (7,200 imágenes RGB 128x128)
│   ├── processed/                     # Particiones de los Protocolos A, B, C, D
│   └── knowledge_base/                # Manuales SOP de planta y mantenimiento para RAG con citas
├── models/                            # Artefacto productivo versionado y Model Card
├── notebooks/                         # Notebook reproducible de experimentación (Reto 2)
├── src/
│   ├── config.py                      # Configuración centralizada separada del código
│   ├── ml/                            # Pipeline de ML clásico, regresión circular y OOD
│   ├── vision_service/                # Microservicio FastAPI de inferencia de visión (Puerto 8000)
│   ├── agent_service/                 # Copiloto agéntico, memoria, RBAC, guardrails y 11 tools (Puerto 8001)
│   └── ui/                            # Consola web unificada para demostración en vivo
├── scripts/                           # Automatización de entrenamiento, evaluación (30+ casos) y reportes
├── tests/                             # Pruebas unitarias, de integración y escenarios adversos
└── docs/
    ├── architecture/                  # Diagramas C4/Mermaid de arquitectura y secuencia
    └── deliverables/                  # Entregables oficiales listos para evaluación
```

---

## Ejecución Rápida (1 Comando) y Demo Pública en Vivo (Cloudflare)

### 🌐 Demo Pública Desplegada en Cloudflare (`FastAPI` — 24/7)
El MVP completo (**Reto 4 — Entregable #10**) construido con **FastAPI** se encuentra desplegado y accesible públicamente en Cloudflare bajo el dominio personalizado:
- **Consola Operativa Interactiva (Visión + Copiloto Agéntico + 9 Escenarios Adversos + HITL):** [https://temporal.labiaco.com/](https://temporal.labiaco.com/)
- **FastAPI Swagger OpenAPI Unificado (`/docs`):** [https://temporal.labiaco.com/docs](https://temporal.labiaco.com/docs)
- **FastAPI Swagger Servicio Independiente de Visión (`/vision/docs`):** [https://temporal.labiaco.com/vision/docs](https://temporal.labiaco.com/vision/docs)
- **Health Check & Telemetría (`/health`):** [https://temporal.labiaco.com/health](https://temporal.labiaco.com/health)

Para actualizar el despliegue en Cloudflare en un solo comando:
```powershell
.\.venv\Scripts\python.exe scripts/build_cloudflare_edge_bundle.py; npx wrangler deploy --config cloudflare/wrangler.jsonc
```

### Opción A: Con Docker Compose (Recomendado para Producción/Evaluación Local)
```bash
cp .env.example .env
docker compose up --build
```

### Opción B: Ejecución Local Reproducible (Windows PowerShell / Linux)
```powershell
# Levanta Servicio de Visión (:8000) + Copiloto Agéntico y Consola Web (:8001) en un solo comando
.\scripts\run_mvp.ps1
```
```bash
# En Linux / macOS / WSL
./scripts/run_mvp.sh
```
- **Consola Operativa Unificada (Local):** `http://localhost:8001/`
- **API Swagger Copiloto Agéntico (Local):** `http://localhost:8001/docs`
- **API Swagger Motor de Inferencia de Visión (Local):** `http://localhost:8000/docs`

---

## Resumen Ejecutivo de Resultados Obtenidos

### 1. Reto 2 — Visión por Computador y ML Clásico sobre COIL-100 (`7,200 imágenes RGB 128×128`)
| Protocolo Experimental | Exactitud SKU (Top-1) | Error Circular MAE ($\theta \in S^1$) | Tasa Dentro de Tolerancia Robot ($\le 10^\circ$) | Tasa de Agarre Seguro Post-Guardrail | Latencia Total CPU |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Protocolo A (Interpolación Angular)** | **99.78%** | **2.90°** *(vs 4.83° MobileNetV3)* | **96.08%** | **99.96%** | **13.03 ms/img** |
| **Protocolo B (Sector Ciego Continuo 60°)** | **92.92%** | **18.96°** *(con simetría)* | **66.44%** | **Guardrail bloquea 80.5% del sector ciego** | **13.10 ms/img** |
| **Protocolo C (Escasez: 1 vista cada 30°)** | **98.25%** | **9.50°** *(5.74° en aceptados)* | **81.75%** | **94.82%** | **12.95 ms/img** |
| **Protocolo D (Objetos Desconocidos / OOD)** | **AUROC = 0.9984** | **TPR OOD = 100.0%** | **F1 OOD = 0.9339** | **0 picks a ciegas sobre SKUs no vistos** | **13.05 ms/img** |

### 2. Reto 3 & Reto 4 — Copiloto Operativo Multiagente (Benchmark de 32 Casos + 9 Escenarios Adversos)
- **Task Success Rate (`32/32` casos):** **100.0%**
- **Tool Selection & Argument Accuracy:** **100.0%** sobre el catálogo de **11 tools gobernadas** (6 lectura + 5 escritura con idempotencia).
- **Grounded Citation Accuracy:** **100.0%** de citas verificables sobre [`plant_sops_manual.json`](data/knowledge_base/plant_sops_manual.json) (`[SOP-OPT-04]`, `[SOP-ROB-11]`, `[SOP-QUAL-02]`, `[SOP-ML-07]`).
- **Seguridad y Contención Adversaria:** Bloqueo determinístico del 100% de ataques de *Prompt Injection (OWASP LLM01)*, violaciones RBAC, accesos *cross-tenant*, loops agénticos (`LoopAndBudgetGuard`) y caídas `HTTP 503 / 504` del servicio de visión.
- **Soporte LLM Local:** Integración nativa con **Ollama** (`llama3.1:8b` / `qwen2.5-coder:3b`) + fallback determinístico estructurado.

### 3. Reto 6 — Arquitectura Cloud Híbrida (Edge + Azure vs AWS) y FinOps
- **Costo Cloud Mensual en Microsoft Azure (1 Estación, 3 Turnos = 135,000 objetos/mes):** **$241.55 USD/mes** (**18.4% menor** que el equivalente en AWS de **$296.10 USD/mes**).
- **Economía de Escala (10 Estaciones = 1.35M objetos/mes):** **$684.00 USD/mes** (**$68.40 USD/estación/mes**, reducción unitaria del **71.7%**).
- **Caso de Negocio frente al Baseline Manual de 3 Turnos ($6,725.00 USD/mes):** Ahorro neto de **$5,481.45 USD/mes** (**$65,777.40 USD/año** por estación), con **Payback de 3.38 meses** y **ROI Año 1 de +255.6%**.

---

## Comandos de Verificación y Reproducción Completa

```powershell
# 1. Ejecutar toda la suite de pruebas automatizadas (12 tests: ML Clásico, Vision API, Agente, 9 Escenarios Adversos y UI)
.\.venv\Scripts\pytest.exe tests/ -v

# 2. Re-ejecutar evaluación de 32 casos del Agente y regenerar Entregables #7 y #8
.\.venv\Scripts\python.exe scripts/generate_reto3_deliverables.py

# 3. Regenerar Modelo de Costos FinOps (.xlsx) y Reporte de Arquitectura Cloud Azure vs AWS (Entregable #12)
.\.venv\Scripts\python.exe scripts/generate_reto6_deliverables.py
```

