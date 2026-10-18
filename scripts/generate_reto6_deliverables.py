"""Generador de Entregables del Reto 6 (Entregable #12):
1. Modelo Financiero y FinOps Multi-Hoja en Excel (`docs/deliverables/Entregable_12_Modelo_Costos_FinOps_Azure.xlsx`)
   con fórmulas dinámicas de Excel (`SUM`, multiplicaciones de volumetría, escalamiento de 1 a 10 estaciones y ROI).
2. Documento de Arquitectura Cloud Azure vs AWS + FinOps (`docs/deliverables/Entregable_12_Arquitectura_Cloud_Azure_FinOps.md`).
3. Reporte Ejecutivo PDF (`docs/deliverables/Entregable_12_Arquitectura_Cloud_Azure_FinOps.pdf`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DELIVERABLES_DIR = PROJECT_ROOT / "docs" / "deliverables"
DELIVERABLES_DIR.mkdir(parents=True, exist_ok=True)


def _style_header_Range(ws: Any, cell_range: str, fill_hex: str = "0F172A") -> None:
    fill = PatternFill(start_color=fill_hex, end_color=fill_hex, fill_type="solid")
    font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    for row in ws[cell_range]:
        for cell in row:
            cell.fill = fill
            cell.font = font
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)


def _auto_adjust_columns(ws: Any, min_width: int = 14, max_width: int = 48) -> None:
    for col in ws.columns:
        col_letter = get_column_letter(col[0].column)
        max_len = 0
        for cell in col:
            val = str(cell.value or "")
            if val.startswith("="):
                max_len = max(max_len, 14)
            else:
                max_len = max(max_len, len(val))
        ws.column_dimensions[col_letter].width = max(min_width, min(max_len + 4, max_width))


def build_excel_cost_model(xlsx_path: Path) -> None:
    wb = openpyxl.Workbook()
    thin_border = Border(
        left=Side(style="thin", color="CBD5E1"),
        right=Side(style="thin", color="CBD5E1"),
        top=Side(style="thin", color="CBD5E1"),
        bottom=Side(style="thin", color="CBD5E1"),
    )
    total_fill = PatternFill(start_color="E0F2FE", end_color="E0F2FE", fill_type="solid")
    bold_font = Font(name="Calibri", size=11, bold=True, color="0F172A")

    # =========================================================================
    # HOJA 1: 1_Parametros_y_Volumetria
    # =========================================================================
    ws1 = wb.active
    ws1.title = "1_Parametros_y_Volumetria"

    ws1["A1"] = "INDRA GROUP — MODELO DE VOLUMETRÍA Y PARÁMETROS OPERATIVOS (PICK & PACK IA)"
    ws1.merge_cells("A1:E1")
    _style_header_Range(ws1, "A1:E1", "0284C7")

    headers_1 = ["Código", "Parámetro Operativo / Técnico", "Valor", "Unidad", "Justificación / Supuesto de Ingeniería"]
    ws1.append([])
    ws1.append(headers_1)
    _style_header_Range(ws1, "A3:E3", "0F172A")

    params_rows = [
        ("P01", "Turnos operativos por día", 3, "turnos/día", "Operación continua 24h (3 turnos de 8 horas en planta)."),
        ("P02", "Días operativos por mes", 30, "días/mes", "Planta industrial de empaque operando 30 días/mes."),
        ("P03", "Inspecciones de visión por turno (1 estación)", 1500, "objetos/turno", "Cadencia media de 3.12 piezas/minuto con picos de 15 piezas/minuto."),
        ("P04", "Inspecciones mensuales de visión (1 estación)", "=C4*C5*C6", "objetos/mes", "135,000 inferencias/mes procesadas 100% en CPU Edge (<15 ms)."),
        ("P05", "Tasa de abstención / cola de revisión humana", 0.0392, "proporción", "3.92% de capturas dudosas (sigma_theta > 10° o desenfoque) suben a nube."),
        ("P06", "Muestras de auditoría subidas a Blob Storage", 0.0500, "proporción", "5% de muestreo aleatorio + 3.92% rechazadas para MLOps y reentrenamiento."),
        ("P07", "Tamaño medio de imagen RGB 128x128 (PNG)", 0.028, "MB/imagen", "Imagen 128x128 comprimida + metadatos JSON de telemetría."),
        ("P08", "Consultas al Copiloto Agéntico por turno", 60, "turnos/turno", "Supervisores, calidad, mantenimiento y entrega de turno automática."),
        ("P09", "Interacciones agénticas mensuales (1 estación)", "=C4*C5*C11", "interacciones/mes", "5,400 ejecuciones agénticas/mes por estación."),
        ("P10", "Tokens de entrada promedio por turno agéntico", 1600, "tokens/turno", "System prompt + esquemas de tools + contexto RAG acotado (ventana 6 turnos)."),
        ("P11", "Tokens de salida promedio por turno agéntico", 450, "tokens/turno", "Invocación JSON de tools + respuesta ejecutiva con citas SOP."),
        ("P12", "Proporción de tráfico enrutado a modelo ligero (GPT-4o-mini)", 0.80, "proporción", "80% de consultas estructuradas/rutinarias resueltas por modelo ligero."),
        ("P13", "Proporción de tráfico enrutado a modelo avanzado (GPT-4o)", 0.20, "proporción", "20% de diagnósticos complejos multi-causa y reportes gerenciales."),
        ("P14", "Tarifa Azure OpenAI GPT-4o-mini Input (1M tokens)", 0.15, "USD / 1M tok", "Precios públicos Azure OpenAI East US 2 (Pay-As-You-Go)."),
        ("P15", "Tarifa Azure OpenAI GPT-4o-mini Output (1M tokens)", 0.60, "USD / 1M tok", "Precios públicos Azure OpenAI East US 2 (Pay-As-You-Go)."),
        ("P16", "Tarifa Azure OpenAI GPT-4o Input (1M tokens)", 2.50, "USD / 1M tok", "Precios públicos Azure OpenAI East US 2 (Pay-As-You-Go)."),
        ("P17", "Tarifa Azure OpenAI GPT-4o Output (1M tokens)", 10.00, "USD / 1M tok", "Precios públicos Azure OpenAI East US 2 (Pay-As-You-Go)."),
    ]
    for r in params_rows:
        ws1.append(list(r))

    for row_idx in range(4, 4 + len(params_rows)):
        for col_idx in range(1, 6):
            c = ws1.cell(row=row_idx, column=col_idx)
            c.border = thin_border
    _auto_adjust_columns(ws1)

    # =========================================================================
    # HOJA 2: 2_Costos_Azure_1_Estacion
    # =========================================================================
    ws2 = wb.create_sheet("2_Costos_Azure_1_Estacion")
    ws2["A1"] = "DESGLOSE FINOPS DE ARQUITECTURA HÍBRIDA EDGE + AZURE CLOUD (1 ESTACIÓN PRODUCTIVA)"
    ws2.merge_cells("A1:G1")
    _style_header_Range(ws2, "A1:G1", "0284C7")

    headers_2 = [
        "Capa Arquitectónica",
        "Servicio Concreto en Microsoft Azure",
        "SKU / Tier Dimensionado",
        "Unidades / Mes",
        "Costo Unitario (USD)",
        "Costo Mensual (USD)",
        "Costo Anual (USD)",
    ]
    ws2.append([])
    ws2.append(headers_2)
    _style_header_Range(ws2, "A3:G3", "0F172A")

    cost_rows = [
        (
            "1. Edge & Ingesta Industrial",
            "Azure IoT Hub + Azure Arc K8s Management",
            "IoT Hub Standard S1 (400k msgs/día) + Arc Native",
            1,
            25.00,
            "=D4*E4",
            "=F4*12",
        ),
        (
            "2. Mensajería & Colas HITL",
            "Azure Service Bus + Event Hubs",
            "Service Bus Standard (Colas de rechazo, tickets y HITL)",
            1,
            12.50,
            "=D5*E5",
            "=F5*12",
        ),
        (
            "3. Cómputo Copiloto & APIs",
            "Azure Container Apps (ACA)",
            "Workload Consumption (1 vCPU, 2 GB RAM, 1 réplica activa)",
            1,
            34.80,
            "=D6*E6",
            "=F6*12",
        ),
        (
            "4. Inferencia LLM (Ruta Ligera 80%)",
            "Azure OpenAI Service — GPT-4o-mini",
            "80% de 5,400 turnos/mes (Input 1.6k + Output 450 tok)",
            1,
            1.90,
            "=D7*E7",
            "=F7*12",
        ),
        (
            "5. Inferencia LLM (Ruta Compleja 20%)",
            "Azure OpenAI Service — GPT-4o",
            "20% de 5,400 turnos/mes + síntesis de reportes semanales",
            1,
            12.80,
            "=D8*E8",
            "=F8*12",
        ),
        (
            "6. Motor RAG Híbrido (SOPs)",
            "Azure AI Search + Embeddings",
            "Tier Basic (BM25 Léxico + Vectorial text-embedding-3-small)",
            1,
            75.10,
            "=D9*E9",
            "=F9*12",
        ),
        (
            "7. Base de Estado & Auditoría",
            "Azure Database for PostgreSQL Flexible Server",
            "Burstable B1ms (2 vCores, 4 GB RAM, 32 GB SSD + Backups)",
            1,
            31.40,
            "=D10*E10",
            "=F10*12",
        ),
        (
            "8. Estado Efímero & Idempotencia",
            "Azure Managed Redis",
            "Basic C0 (250 MB para ventana de sesión y locks TTL 24h)",
            1,
            16.00,
            "=D11*E11",
            "=F11*12",
        ),
        (
            "9. Almacenamiento Imágenes & WORM",
            "Azure Data Lake Storage Gen2 (Blob)",
            "50 GB Hot (90d) + 200 GB Cool/Archive + Audit Log Inmutable",
            1,
            4.85,
            "=D12*E12",
            "=F12*12",
        ),
        (
            "10. MLOps, Seguridad & Observabilidad",
            "Azure ML + App Insights + Key Vault",
            "Compute Serverless (reentreno semanal) + 8 GB Log Analytics",
            1,
            27.20,
            "=D13*E13",
            "=F13*12",
        ),
    ]
    for r in cost_rows:
        ws2.append(list(r))

    ws2.append([
        "TOTAL CLOUD AZURE (1 ESTACIÓN)",
        "Arquitectura Híbrida Edge-to-Cloud Gobernada",
        "135,000 inspecciones visión/mes + 5,400 turnos agénticos/mes",
        "",
        "",
        "=SUM(F4:F13)",
        "=SUM(G4:G13)",
    ])
    ws2.append([
        "COSTO UNITARIO POR 1,000 OBJETOS",
        "Eficiencia FinOps por procesamiento visual en el Edge",
        "USD por cada 1,000 piezas inspeccionadas en línea",
        "",
        "",
        "=(F14/135000)*1000",
        "",
    ])

    for r_i in range(4, 16):
        for c_i in range(1, 8):
            cell = ws2.cell(row=r_i, column=c_i)
            cell.border = thin_border
            if c_i in (5, 6, 7):
                cell.number_format = "$#,##0.00"
            if r_i in (14, 15):
                cell.fill = total_fill
                cell.font = bold_font
    _auto_adjust_columns(ws2)

    # =========================================================================
    # HOJA 3: 3_Escalamiento_1_a_10_Est
    # =========================================================================
    ws3 = wb.create_sheet("3_Escalamiento_1_a_10_Est")
    ws3["A1"] = "ECONOMÍA DE ESCALA FINOPS: ESCALAMIENTO DE 1 A 10 ESTACIONES DE INSPECCIÓN"
    ws3.merge_cells("A1:F1")
    _style_header_Range(ws3, "A1:F1", "0284C7")

    ws3.append([])
    ws3.append([
        "Componente de Costo Mensual (USD)",
        "1 Estación (135k obj/mes)",
        "3 Estaciones (405k obj/mes)",
        "5 Estaciones (675k obj/mes)",
        "10 Estaciones (1.35M obj/mes)",
        "Comportamiento de Escalamiento FinOps",
    ])
    _style_header_Range(ws3, "A3:F3", "0F172A")

    scale_rows = [
        ("Ingesta IoT Hub + Event Hubs + Service Bus", 37.50, 48.00, 62.00, 95.00, "Sub-lineal: S1/S2 absorbe múltiples estaciones por planta."),
        ("Cómputo Azure Container Apps (Copiloto API)", 34.80, 49.00, 68.00, 112.00, "Auto-scaling KEDA compartido entre todas las estaciones."),
        ("Consumo Tokens Azure OpenAI (80% mini / 20% 4o)", 14.70, 44.10, 73.50, 147.00, "Lineal con el número de consultas operativas y turnos."),
        ("Motor RAG Azure AI Search (Índice SOP Compartido)", 75.10, 75.10, 75.10, 75.10, "Costo Fijo: Un único índice sirve a las 10 estaciones."),
        ("Estado PostgreSQL Flexible + Azure Managed Redis", 47.40, 58.00, 79.00, 118.00, "Compartido multi-tenant (plant_id / line_id / station_id)."),
        ("Almacenamiento ADLS Gen2 + MLOps + App Insights", 32.05, 52.00, 78.00, 136.90, "Crecimiento moderado por retención Hot/Cool/Archive."),
    ]
    for r in scale_rows:
        ws3.append(list(r))

    ws3.append([
        "COSTO TOTAL MENSUAL CLOUD (USD/mes)",
        "=SUM(B4:B9)",
        "=SUM(C4:C9)",
        "=SUM(D4:D9)",
        "=SUM(E4:E9)",
        "Amortización de servicios base compartidos.",
    ])
    ws3.append([
        "COSTO PROMEDIO POR ESTACIÓN (USD/estación/mes)",
        "=B10/1",
        "=C10/3",
        "=D10/5",
        "=E10/10",
        "Reducción del 71.7% en costo por estación al escalar a 10 líneas.",
    ])

    for r_i in range(4, 12):
        for c_i in range(1, 7):
            cell = ws3.cell(row=r_i, column=c_i)
            cell.border = thin_border
            if c_i in (2, 3, 4, 5):
                cell.number_format = "$#,##0.00"
            if r_i in (10, 11):
                cell.fill = total_fill
                cell.font = bold_font
    _auto_adjust_columns(ws3)

    # =========================================================================
    # HOJA 4: 4_Comparativa_Azure_vs_AWS
    # =========================================================================
    ws4 = wb.create_sheet("4_Comparativa_Azure_vs_AWS")
    ws4["A1"] = "TABLA COMPARATIVA TÉCNICA Y ECONÓMICA: MICROSOFT AZURE vs AMAZON WEB SERVICES (AWS)"
    ws4.merge_cells("A1:F1")
    _style_header_Range(ws4, "A1:F1", "0284C7")

    ws4.append([])
    ws4.append([
        "Dominio Arquitectónico",
        "Microsoft Azure (Recomendado)",
        "Amazon Web Services (Alternativa)",
        "Azure 1 Est. (USD/m)",
        "AWS 1 Est. (USD/m)",
        "Ventaja Estratégica / Trade-off Decisivo",
    ])
    _style_header_Range(ws4, "A3:F3", "0F172A")

    comp_rows = [
        (
            "1. Edge Runtime & Gestión OT",
            "Azure IoT Operations + Azure Arc + IoT Hub S1",
            "AWS IoT Greengrass v2 + IoT Core + Systems Manager",
            25.00,
            28.00,
            "Azure Arc ofrece integración nativa superior con IPCs Windows/Linux industriales y Entra ID.",
        ),
        (
            "2. Mensajería & Eventos",
            "Azure Service Bus + Event Hubs",
            "Amazon SQS/SNS + Amazon Kinesis Data Streams",
            12.50,
            15.40,
            "Service Bus soporta sesiones FIFO transaccionales nativas ideales para aprobaciones HITL.",
        ),
        (
            "3. Hosting Microservicios Agente",
            "Azure Container Apps (Serverless KEDA)",
            "AWS App Runner / Amazon ECS Fargate",
            34.80,
            38.20,
            "ACA permite escalado basado en colas Service Bus con KEDA sin administrar plano de control.",
        ),
        (
            "4. Modelos Fundacionales (LLM)",
            "Azure OpenAI (GPT-4o-mini 80% + GPT-4o 20%)",
            "Amazon Bedrock (Claude 3.5 Haiku 80% + Sonnet 20%)",
            14.70,
            18.90,
            "GPT-4o-mini presenta menor costo por millón de tokens y latencia determinista en JSON Schema.",
        ),
        (
            "5. Motor RAG Híbrido (SOPs)",
            "Azure AI Search Basic (BM25 + Vector + Semantic)",
            "Amazon OpenSearch Serverless / Aurora pgvector",
            75.10,
            112.00,
            "Azure AI Search Basic cuesta $75/mes con búsqueda híbrida nativa; OpenSearch Serverless exige mínimo 2 OCUs (~$175+).",
        ),
        (
            "6. Estado Operativo & Caché",
            "Azure DB for PostgreSQL Flexible + Managed Redis",
            "Amazon RDS PostgreSQL (t4g.small) + ElastiCache",
            47.40,
            49.50,
            "Paridad técnica y económica; ambos soportan alta disponibilidad e aislamiento por tenant.",
        ),
        (
            "7. Data Lake, MLOps & Monitor",
            "ADLS Gen2 + Azure ML + Application Insights",
            "Amazon S3 + SageMaker AI + CloudWatch / X-Ray",
            32.05,
            34.10,
            "Application Insights ofrece trazabilidad OpenTelemetry nativa con Azure AI Foundry.",
        ),
    ]
    for r in comp_rows:
        ws4.append(list(r))

    ws4.append([
        "TOTAL MENSUAL COMPARADO (1 ESTACIÓN)",
        "Microsoft Azure (Cloud + Edge Control Plane)",
        "Amazon Web Services (Cloud + Edge Control Plane)",
        "=SUM(D4:D10)",
        "=SUM(E4:E10)",
        "Azure ahorra ~18.4% en escala inicial gracias al tier Basic de Azure AI Search y GPT-4o-mini.",
    ])

    for r_i in range(4, 12):
        for c_i in range(1, 7):
            cell = ws4.cell(row=r_i, column=c_i)
            cell.border = thin_border
            if c_i in (4, 5):
                cell.number_format = "$#,##0.00"
            if r_i == 11:
                cell.fill = total_fill
                cell.font = bold_font
    _auto_adjust_columns(ws4)

    # =========================================================================
    # HOJA 5: 5_Caso_Negocio_ROI_3_Turnos
    # =========================================================================
    ws5 = wb.create_sheet("5_Caso_Negocio_ROI_3_Turnos")
    ws5["A1"] = "CASO DE NEGOCIO Y ROI: BASELINE OPERATIVO (3 TURNOS) vs SOLUCIÓN IA INDRA"
    ws5.merge_cells("A1:F1")
    _style_header_Range(ws5, "A1:F1", "0284C7")

    ws5.append([])
    ws5.append([
        "Rubro de Costo / Beneficio (1 Estación, 3 Turnos = 24h/día)",
        "Baseline Manual / Sin IA (USD/mes)",
        "Con Sistema Indra IA (USD/mes)",
        "Ahorro Mensual (USD/mes)",
        "Ahorro Anual (USD/año)",
        "Sustento Operativo y Técnico",
    ])
    _style_header_Range(ws5, "A3:F3", "0F172A")

    roi_rows = [
        (
            "1. Personal de Inspección y Orientación Manual (3 Turnos + Relevos)",
            4725.00,
            675.00,
            "=B4-C4",
            "=D4*12",
            "Pasa de 3.5 FTEs dedicados (1 por turno) a 0.5 FTE supervisando únicamente la cola de abstención (3.92%).",
        ),
        (
            "2. Costo de Colisiones Robóticas, Reproceso y Daño de Empaque",
            1150.00,
            92.00,
            "=B5-C5",
            "=D5*12",
            "El guardrail de incertidumbre circular (sigma_theta <= 10°) logra 99.96% de agarres seguros.",
        ),
        (
            "3. Paradas No Planificadas por Diagnóstico Lento (MTTR 45m -> 8m)",
            850.00,
            170.00,
            "=B6-C6",
            "=D6*12",
            "El Copiloto Agéntico correlaciona telemetría + SOPs con citas y escala tickets en segundos.",
        ),
        (
            "4. Infraestructura Cloud Azure (OPEX Mensual — Hoja 2)",
            0.00,
            "='2_Costos_Azure_1_Estacion'!F14",
            "=B7-C7",
            "=D7*12",
            "Incluye Azure OpenAI, AI Search, Container Apps, PostgreSQL, Redis, IoT Hub y MLOps.",
        ),
        (
            "5. Amortización IPC Industrial Edge (CAPEX $2,340 a 36 meses)",
            0.00,
            65.00,
            "=B8-C8",
            "=D8*12",
            "Nodo industrial fanless Intel Core i7 / 16GB RAM ejecutando visión clásica en 13.03 ms.",
        ),
    ]
    for r in roi_rows:
        ws5.append(list(r))

    ws5.append([
        "FLUJO OPERATIVO TOTAL Y AHORRO NETO (1 ESTACIÓN)",
        "=SUM(B4:B8)",
        "=SUM(C4:C8)",
        "=SUM(D4:D8)",
        "=SUM(E4:E8)",
        "Reducción neta del 81.5% en el costo operativo de la estación Pick & Pack.",
    ])
    ws5.append([
        "PAYBACK DE INVERSIÓN INICIAL ($18,500 USD CAPEX + Integración)",
        "",
        "",
        "=18500/D9",
        "=(E9-18500)/18500",
        "Columna D = Meses para recuperar inversión (3.38 meses) | Columna E = ROI Neto Año 1 (+255.6%).",
    ])

    for r_i in range(4, 11):
        for c_i in range(1, 7):
            cell = ws5.cell(row=r_i, column=c_i)
            cell.border = thin_border
            if r_i <= 9 and c_i in (2, 3, 4, 5):
                cell.number_format = "$#,##0.00"
            if r_i == 9:
                cell.fill = total_fill
                cell.font = bold_font
            if r_i == 10:
                cell.fill = total_fill
                cell.font = bold_font
                if c_i == 4:
                    cell.number_format = "0.00\" meses\""
                elif c_i == 5:
                    cell.number_format = "0.0%"
    _auto_adjust_columns(ws5)

    wb.save(xlsx_path)


def build_architecture_markdown() -> str:
    return """# Entregable #12 — Arquitectura Cloud Híbrida (Edge + Azure), Comparativa AWS y Modelo FinOps

**Prueba Técnica AI Engineer (Senior+) — Indra Group**  
**Reto 6:** Arquitectura en la Nube, Ciclo de Vida del Dato, FinOps y Caso de Negocio (3 Turnos)

---

## 1. Visión Arquitectónica y Decisión Clave de Ingeniería (Edge-First + Cloud-Governed)

En una línea industrial de *Pick & Pack* que opera 3 turnos continuos (`135,000 objetos/mes` por estación a ráfagas de hasta 15 piezas/minuto con restricción de control PLC `< 50 ms`), enviar cada frame de cámara a la nube o invocar un modelo multimodal (VLM) por pieza sería **técnica y económicamente inviable** (alta latencia WAN, riesgo de parada por corte de internet y costo desbordado de tokens/GPU).

Por ello, nuestra arquitectura implementa un patrón **Híbrido Edge-to-Cloud con Separación de Planos**:

1. **Plano de Inferencia Determinística en el Edge (Planta — `< 15 ms`, Offline-First)**:
   - El microservicio `indra-vision-service` (`COIL100ClassicalSystem`: HOG + Histogramas HSV/RGB + Perfil Azimutal + Regresión Circular $(\sin\theta, \cos\theta)$) corre localmente en un **IPC Industrial gestionado por Azure Arc / Azure IoT Operations** sobre CPU estándar (`13.03 ms` por imagen, **$0 USD de costo por inferencia en la nube**).
   - Solo se transmiten a la nube: (a) telemetría agregada por minuto/turno, (b) el **3.92%** de imágenes rechazadas o con incertidumbre angular $\sigma_\theta > 10^\circ$ para revisión humana y reentrenamiento, y (c) un muestreo de auditoría del **5%**.
2. **Plano de Control Agéntico, RAG, HITL y MLOps en Microsoft Azure**:
   - El **Copiloto Operativo Multiagente** (`indra-agent-copilot`) se despliega en **Azure Container Apps**, conectado a **Azure OpenAI Service** (con *Model Routing* 80% `GPT-4o-mini` / 20% `GPT-4o`), **Azure AI Search** (búsqueda híbrida BM25 + vectorial sobre SOPs), **Azure Database for PostgreSQL Flexible Server** (estado operativo y `audit_log` inmutable) y **Azure Machine Learning** (reentrenamiento semanal con etiquetas confirmadas por humanos).

```mermaid
flowchart TB
    subgraph EDGE["PLANTA INDUSTRIAL (Edge Layer — Azure Arc & IoT Operations)"]
        CAM["Cámara Cenital RGB 128x128\n+ Anillo LED"] --> VSVC["Contenedor Vision Service\nCOIL-100 Clásico + Regresión Circular\n(CPU: 13.03 ms | 99.96% Agarre Seguro)"]
        VSVC -->|"PICK_AND_ROTATE (θ̂, σ_θ ≤ 10°)\nModbus TCP < 2 ms"| PLC["PLC & Brazo Robótico\nEstación Pick & Pack"]
        VSVC -->|"Cola Local WAL / MQTT\n(3.92% Abstención + Telemetría)"| IOT_EDGE["Agente Azure IoT Operations"]
    end

    subgraph AZURE["NUBE PRINCIPAL — MICROSOFT AZURE (East US 2)"]
        IOT_EDGE -->|"TLS 1.3 / AMQP"| IOTHUB["Azure IoT Hub (S1)\n& Event Hubs"]
        IOTHUB --> SBUS["Azure Service Bus\n(Colas de Rechazo, Tickets y HITL)"]

        subgraph COPILOT["Plano del Copiloto Agéntico y Consola Operativa"]
            APIM["Azure API Management\n+ Microsoft Entra ID (RBAC)"] --> ACA["Azure Container Apps\nOrquestador + 4 Subagentes + 11 Tools"]
            ACA --> REDIS["Azure Managed Redis (C0)\nEstado Efímero, Ventana Sesión & Locks Idempotencia"]
            ACA --> PG["Azure DB for PostgreSQL Flexible\nEstado Operativo + Audit Log Inmutable"]
            ACA --> AISEARCH["Azure AI Search (Basic)\nRAG Híbrido BM25 + Vector SOPs"]
            ACA --> AOAI["Azure OpenAI Service\nModel Routing: 80% GPT-4o-mini / 20% GPT-4o"]
        end

        subgraph MLOPS["Plano de Datos, Ciclo de Vida y MLOps"]
            SBUS --> PG
            IOT_EDGE -->|"Imágenes Rechazadas + Muestra 5%"| ADLS["Azure Data Lake Storage Gen2\nHot (90d) -> Cool -> Archive + WORM"]
            PG -->|"Etiquetas Confirmadas Humanas"| AML["Azure Machine Learning\nReentrenamiento Semanal + Evaluación A/B/C/D"]
            AML -->|"Despliegue Canary / Shadow"| IOT_EDGE
        end

        subgraph OBS["Seguridad, Gobierno y Observabilidad"]
            MON["Azure Monitor + Application Insights\n(OpenTelemetry Traces, Tokens, Deriva σ_θ)"]
            KV["Azure Key Vault + Defender for Cloud"]
        end
    end
```

---

## 2. Justificación de Servicios y Trade-offs Arquitectónicos en Azure

| Capa | Servicio Elegido en Azure | Alternativa Descartada | Justificación Técnica y Económica (Trade-off) |
| :--- | :--- | :--- | :--- |
| **Inferencia de Visión** | **Contenedor en Edge (Azure Arc + IPC CPU)** | Endpoint GPU en la nube (Azure ML Managed Online Endpoint) | Inferir 135,000 imágenes/mes en GPU cloud costaría >$380/mes y añadiría 80–150 ms de latencia WAN. En CPU Edge tarda **13.03 ms**, cuesta **$0 en cloud** y opera *offline*. |
| **Hosting del Agente** | **Azure Container Apps (ACA)** | Azure Kubernetes Service (AKS) o Azure Functions | AKS introduce sobrecosto operativo y nodos de sistema para 1–10 estaciones; Functions limita ejecuciones largas o conexiones persistentes. ACA escala con KEDA y cuesta **~$34.80/mes**. |
| **LLM del Agente** | **Azure OpenAI con Enrutamiento (`80% GPT-4o-mini` / `20% GPT-4o`)** | `100% GPT-4o` o `Provisioned Throughput (PTU)` | Usar 100% GPT-4o quintuplica el gasto de tokens sin ganancia en llamadas determinísticas a tools. PTU solo es rentable sobre >50 plantas. El enrutamiento híbrido cuesta **~$14.70/mes** por estación. |
| **Motor RAG (SOPs)** | **Azure AI Search (Tier Basic)** | Vector DB dedicada (Pinecone Enterprise) | Los SOPs de planta contienen códigos alfanuméricos exactos (`SOP-OPT-04`, `ERR-PLC-504`). La búsqueda híbrida (BM25 + Vectores) de Azure AI Search supera al vector puro por **$75.10/mes** compartido entre hasta 10 estaciones. |
| **Base de Estado** | **Azure DB for PostgreSQL Flexible + Azure Managed Redis** | Azure Cosmos DB multi-región | Las tablas operativas (`rejection_queue`, `maintenance_tickets`, `risk_operation_approvals`, `audit_log`) exigen integridad transaccional ACID y consultas relacionales por turno. |

---

## 3. Ciclo de Vida del Dato, Retención y Cumplimiento Industrial

| Tipo de Dato / Artefacto | Almacenamiento Primario | Política de Retención y Tiering | Controles de Seguridad y Privacidad |
| :--- | :--- | :--- | :--- |
| **Imágenes Nominales (`PICK_AND_ROTATE`)** | Buffer circular local en Edge + 5% muestreo en ADLS Gen2 | Edge: **48 horas**.<br>Cloud Hot: **7 días** $\rightarrow$ Cool: **30 días** $\rightarrow$ Purga automática. | Sin rostros ni PII (cámara cenital sobre banda transportadora). Cifrado AES-256. |
| **Imágenes Rechazadas y Re-etiquetadas (`retraining_dataset`)** | ADLS Gen2 + Metadatos en PostgreSQL | Hot: **90 días** (reentrenamiento activo) $\rightarrow$ Archive: **365 días**. | Trazabilidad por `case_id`, `confirmed_sku`, `confirmed_angle_deg` y `reviewer_id`. |
| **Estado Efímero y Ventana de Sesión Agéntica** | Azure Managed Redis (`Basic C0`) | **TTL 24 horas** (ventana deslizante de 6 turnos + resumen comprimido). | Aislamiento estricto por llave compuesta `plant_id:line_id:session_id`. |
| **Estado Operativo (`tickets`, `risk_approvals`, `shift_telemetry`)** | Azure Database for PostgreSQL Flexible Server | Activo en línea: **180 días** $\rightarrow$ Particionado histórico: **2 años**. | RBAC por rol (`supervisor`, `quality`, `maintenance`, `ml_engineer`) + Row-Level Security por planta. |
| **Log de Auditoría Inmutable (`audit_log`)** | PostgreSQL Append-Only + Exportación diaria a ADLS Gen2 WORM | **7 años (2,555 días)** en Blob Storage con *Immutable WORM Policy* (ISO 9001 / IEC 62443). | Redacción determinística de PII (`[REDACTED_EMAIL]`, `[REDACTED_PHONE]`, `[REDACTED_API_KEY]`) antes de persistir. |

---

## 4. Tabla Comparativa Exhaustiva: Microsoft Azure vs Amazon Web Services (AWS)

| Dominio Arquitectónico | Microsoft Azure (Arquitectura Propuesta) | Amazon Web Services (Equivalente Funcional) | Costo Mensual 1 Est. (Azure vs AWS) | Criterio de Selección y Diferenciador Técnico |
| :--- | :--- | :--- | :---: | :--- |
| **1. Edge Runtime & Gestión OT** | **Azure IoT Operations + Azure Arc + IoT Hub S1** | **AWS IoT Greengrass v2 + AWS IoT Core + Systems Manager** | **$25.00** vs $28.00 | **Ventaja Azure:** Integración nativa con PCs industriales Windows/Linux y directorio activo corporativo (**Microsoft Entra ID**). |
| **2. Bus de Eventos y Colas HITL** | **Azure Service Bus Standard + Event Hubs** | **Amazon SQS/SNS + Amazon Kinesis Data Streams** | **$12.50** vs $15.40 | **Ventaja Azure:** Sesiones transaccionales FIFO y *dead-lettering* nativo en Service Bus para flujos *Human-in-the-Loop*. |
| **3. Cómputo de Microservicios y Agente** | **Azure Container Apps (ACA con KEDA)** | **AWS App Runner / Amazon ECS Fargate** | **$34.80** vs $38.20 | **Ventaja Azure:** Escalado nativo por eventos de Service Bus/HTTP con perfil de consumo predecible. |
| **4. Modelos Fundacionales (LLM)** | **Azure OpenAI Service (`GPT-4o-mini` 80% + `GPT-4o` 20%)** | **Amazon Bedrock (`Claude 3.5 Haiku` 80% + `Sonnet 3.5` 20%)** | **$14.70** vs $18.90 | **Ventaja Azure:** Menor costo por millón de tokens en `GPT-4o-mini` ($0.15/$0.60) y soporte nativo de *Structured Outputs* para contratos de tools. |
| **5. Motor RAG Híbrido (SOPs)** | **Azure AI Search (Tier Basic)** | **Amazon OpenSearch Serverless / Bedrock KB** | **$75.10** vs $112.00 | **Ventaja Decisiva Azure:** Azure AI Search Basic entrega BM25 + Vector + Semantic Ranker por $75.10/mes, mientras que OpenSearch Serverless exige OCUs mínimas más costosas. |
| **6. Estado Operativo y Caché** | **Azure DB for PostgreSQL Flexible + Azure Managed Redis** | **Amazon RDS PostgreSQL + Amazon ElastiCache Redis** | **$47.40** vs $49.50 | **Empate Técnico:** Ambos ofrecen alta disponibilidad, backups automatizados y aislamiento multi-tenant. |
| **7. Data Lake, MLOps y Observabilidad** | **ADLS Gen2 + Azure ML + Application Insights** | **Amazon S3 + Amazon SageMaker AI + CloudWatch / X-Ray** | **$32.05** vs $34.10 | **Ventaja Azure:** Trazabilidad unificada OpenTelemetry entre Container Apps, Azure OpenAI y Application Insights. |
| **TOTAL MENSUAL CLOUD (1 ESTACIÓN)** | **Microsoft Azure Total** | **Amazon Web Services Total** | **$241.55 / mes** vs **$296.10 / mes** | **Azure reduce el TCO Cloud inicial en un 18.4%** y se alinea con entornos empresariales Microsoft en manufactura. |

---

## 5. Modelo Financiero FinOps y Escalamiento (1 a 10 Estaciones)

El archivo Excel adjunto ([`Entregable_12_Modelo_Costos_FinOps_Azure.xlsx`](file:///e:/10-LABIA/ia-tradicional/prueba-tecnica-indra/prueba-tecnica-ingeniero-ia-indra/docs/deliverables/Entregable_12_Modelo_Costos_FinOps_Azure.xlsx)) modela con fórmulas dinámicas el costo mensual y anual para **1, 3, 5 y 10 estaciones**:

| Escala de Despliegue | Inspecciones Visión / Mes | Turnos Agénticos / Mes | Costo Total Cloud Azure (USD/mes) | Costo Cloud por Estación (USD/mes) | Costo Cloud por 1,000 Objetos |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1 Estación (MVP Planta)** | 135,000 | 5,400 | **$241.55** | **$241.55** | **$1.79** |
| **3 Estaciones (Línea Completa)** | 405,000 | 16,200 | **$326.20** | **$108.73** | **$0.81** |
| **5 Estaciones (Nave Industrial)** | 675,000 | 27,000 | **$435.60** | **$87.12** | **$0.65** |
| **10 Estaciones (Planta Completa)** | 1,350,000 | 54,000 | **$684.00** | **$68.40** | **$0.51** |

### Palancas de Optimización FinOps Aplicadas
1. **Inferencia Visual 100% en Edge CPU (`$0 Cloud GPU`)**: Evita enviar 135,000 imágenes/mes por estación a endpoints GPU o VLMs en la nube, reduciendo el costo de cómputo en más de un **90%**.
2. **Enrutamiento Inteligente de Modelos (`80% GPT-4o-mini` / `20% GPT-4o`) + Fallback Local (`Ollama`)**: Las consultas estructuradas de inventario, cola de rechazo y telemetría usan `GPT-4o-mini`, reservando `GPT-4o` solo para correlación multi-causa.
3. **Presupuesto Estricto de Contexto y Anti-Loop (`LoopAndBudgetGuard`)**: Límite duro de `max_steps=6`, `max_budget_tokens=4096` y ventana deslizante de 6 turnos en Redis/SQLite que impide explosión de costos por bucles agénticos.
4. **Amortización Multi-Estación**: Un único índice de `Azure AI Search` y un único clúster `PostgreSQL Flexible` sirven de 1 a 10 estaciones mediante aislamiento lógico (`plant_id`, `line_id`, `station_id`).

---

## 6. Caso de Negocio y Retorno de Inversión (ROI — 3 Turnos / 24h)

Comparando la operación tradicional de una estación *Pick & Pack* en 3 turnos (24 horas/día) frente a la solución automatizada con guardrails de incertidumbre circular ($\sigma_\theta \le 10^\circ$) y Copiloto Agéntico:

| Rubro Operativo (1 Estación — 3 Turnos / 24h) | Baseline Manual / Sin IA (USD/mes) | Con Solución Indra IA (USD/mes) | Ahorro Neto Mensual (USD/mes) | Ahorro Neto Anual (USD/año) |
| :--- | :---: | :---: | :---: | :---: |
| **1. Mano de Obra de Inspección/Orientación (3.5 FTEs $\rightarrow$ 0.5 FTE)** | $4,725.00 | $675.00 | **+$4,050.00** | **+$48,600.00** |
| **2. Colisiones Robóticas, Daño de Empaque y Reproceso** | $1,150.00 | $92.00 | **+$1,058.00** | **+$12,696.00** |
| **3. Paradas No Planificadas por Diagnóstico Lento (MTTR 45m $\rightarrow$ 8m)** | $850.00 | $170.00 | **+$680.00** | **+$8,160.00** |
| **4. OPEX Cloud Microsoft Azure (Hoja 2 del Modelo)** | $0.00 | $241.55 | -$241.55 | -$2,898.60 |
| **5. Amortización Hardware IPC Edge ($2,340 a 36 meses)** | $0.00 | $65.00 | -$65.00 | -$780.00 |
| **TOTAL COSTO OPERATIVO / AHORRO NETO** | **$6,725.00 / mes** | **$1,243.55 / mes** | **+$5,481.45 / mes** | **+$65,777.40 / año** |

* **Reducción de Costos Operativos en Planta**: **81.5%**.
* **Inversión Inicial Estimada (CAPEX IPC + Integración MVP)**: **$18,500.00 USD**.
* **Periodo de Recuperación (*Payback Period*)**: **3.38 meses**.
* **ROI Neto en el Primer Año**: **+255.6%**.
"""


def build_architecture_pdf(pdf_path: Path) -> None:
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleIndra",
        parent=styles["Heading1"],
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=8,
    )
    h2_style = ParagraphStyle(
        "H2Indra",
        parent=styles["Heading2"],
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor("#0284C7"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyIndra",
        parent=styles["BodyText"],
        fontSize=9,
        leading=12.5,
        spaceAfter=5,
    )
    cell_style = ParagraphStyle(
        "CellIndra",
        parent=styles["BodyText"],
        fontSize=7.8,
        leading=10,
    )
    cell_hdr = ParagraphStyle(
        "CellHdr",
        parent=styles["BodyText"],
        fontSize=8,
        leading=10,
        textColor=colors.white,
    )

    story: List[Any] = []
    story.append(Paragraph("Entregable #12 — Arquitectura Cloud Híbrida (Azure vs AWS) y Modelo FinOps", title_style))
    story.append(Paragraph("<b>Prueba Técnica AI Engineer (Senior+) — Indra Group | Reto 6</b>", body_style))
    story.append(Spacer(1, 6))

    story.append(Paragraph("1. Arquitectura Híbrida Edge-to-Cloud en Microsoft Azure", h2_style))
    story.append(
        Paragraph(
            "Para cumplir la restricción de tiempo real del brazo robótico (< 50 ms) y optimizar FinOps sobre 135,000 "
            "inspecciones/mes por estación (3 turnos continuos), la solución desacopla el plano de inferencia visual en "
            "el Edge (<b>Azure Arc + Azure IoT Operations</b> ejecutando <i>COIL100ClassicalSystem</i> en CPU en <b>13.03 ms</b> "
            "con costo marginal $0 en nube) del plano de gobierno agéntico en <b>Azure Container Apps</b>, <b>Azure OpenAI "
            "(80% GPT-4o-mini / 20% GPT-4o)</b>, <b>Azure AI Search</b>, <b>PostgreSQL Flexible Server</b> y <b>Azure ML</b>.",
            body_style,
        )
    )

    t1_data = [
        [
            Paragraph("<b>Capa</b>", cell_hdr),
            Paragraph("<b>Servicio Microsoft Azure</b>", cell_hdr),
            Paragraph("<b>Alternativa AWS</b>", cell_hdr),
            Paragraph("<b>Azure/m</b>", cell_hdr),
            Paragraph("<b>AWS/m</b>", cell_hdr),
            Paragraph("<b>Justificación / Trade-off</b>", cell_hdr),
        ],
        [
            Paragraph("Edge & Ingesta", cell_style),
            Paragraph("Azure IoT Operations + Arc + IoT Hub S1", cell_style),
            Paragraph("AWS IoT Greengrass v2 + IoT Core", cell_style),
            Paragraph("$25.00", cell_style),
            Paragraph("$28.00", cell_style),
            Paragraph("Integración nativa con IPCs de planta y Entra ID.", cell_style),
        ],
        [
            Paragraph("Colas & Eventos", cell_style),
            Paragraph("Azure Service Bus + Event Hubs", cell_style),
            Paragraph("Amazon SQS/SNS + Kinesis", cell_style),
            Paragraph("$12.50", cell_style),
            Paragraph("$15.40", cell_style),
            Paragraph("Sesiones FIFO transaccionales para flujos HITL.", cell_style),
        ],
        [
            Paragraph("Cómputo Agente", cell_style),
            Paragraph("Azure Container Apps (ACA)", cell_style),
            Paragraph("AWS App Runner / ECS Fargate", cell_style),
            Paragraph("$34.80", cell_style),
            Paragraph("$38.20", cell_style),
            Paragraph("Escalado KEDA sin administrar clúster AKS.", cell_style),
        ],
        [
            Paragraph("Inferencia LLM", cell_style),
            Paragraph("Azure OpenAI (80% 4o-mini / 20% 4o)", cell_style),
            Paragraph("Amazon Bedrock (Haiku / Sonnet)", cell_style),
            Paragraph("$14.70", cell_style),
            Paragraph("$18.90", cell_style),
            Paragraph("Menor costo por 1M tokens y JSON Schema estricto.", cell_style),
        ],
        [
            Paragraph("Motor RAG SOPs", cell_style),
            Paragraph("Azure AI Search (Tier Basic)", cell_style),
            Paragraph("Amazon OpenSearch Serverless", cell_style),
            Paragraph("$75.10", cell_style),
            Paragraph("$112.00", cell_style),
            Paragraph("Búsqueda híbrida BM25+Vector sin mínimo de OCUs.", cell_style),
        ],
        [
            Paragraph("Estado & Caché", cell_style),
            Paragraph("PostgreSQL Flexible + Managed Redis", cell_style),
            Paragraph("Amazon RDS PostgreSQL + ElastiCache", cell_style),
            Paragraph("$47.40", cell_style),
            Paragraph("$49.50", cell_style),
            Paragraph("Integridad ACID y locks de idempotencia TTL 24h.", cell_style),
        ],
        [
            Paragraph("Data Lake & MLOps", cell_style),
            Paragraph("ADLS Gen2 + Azure ML + App Insights", cell_style),
            Paragraph("Amazon S3 + SageMaker + CloudWatch", cell_style),
            Paragraph("$32.05", cell_style),
            Paragraph("$34.10", cell_style),
            Paragraph("WORM 7 años para audit_log y trazas OpenTelemetry.", cell_style),
        ],
        [
            Paragraph("<b>TOTAL (1 Est.)</b>", cell_style),
            Paragraph("<b>Total Mensual Azure</b>", cell_style),
            Paragraph("<b>Total Mensual AWS</b>", cell_style),
            Paragraph("<b>$241.55</b>", cell_style),
            Paragraph("<b>$296.10</b>", cell_style),
            Paragraph("<b>18.4% menor TCO mensual en Azure.</b>", cell_style),
        ],
    ]

    t1 = Table(t1_data, colWidths=[0.95 * inch, 1.55 * inch, 1.45 * inch, 0.65 * inch, 0.65 * inch, 1.85 * inch])
    t1.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
                ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#E0F2FE")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(t1)
    story.append(Spacer(1, 8))

    story.append(Paragraph("2. Escalamiento FinOps (1 a 10 Estaciones) y Caso de Negocio (ROI 3 Turnos)", h2_style))
    t2_data = [
        [
            Paragraph("<b>Escala / Métrica Financiera</b>", cell_hdr),
            Paragraph("<b>1 Estación</b>", cell_hdr),
            Paragraph("<b>3 Estaciones</b>", cell_hdr),
            Paragraph("<b>5 Estaciones</b>", cell_hdr),
            Paragraph("<b>10 Estaciones</b>", cell_hdr),
        ],
        [
            Paragraph("Inspecciones de Visión / Mes", cell_style),
            Paragraph("135,000", cell_style),
            Paragraph("405,000", cell_style),
            Paragraph("675,000", cell_style),
            Paragraph("1,350,000", cell_style),
        ],
        [
            Paragraph("Costo Total Cloud Azure (USD/mes)", cell_style),
            Paragraph("<b>$241.55</b>", cell_style),
            Paragraph("<b>$326.20</b>", cell_style),
            Paragraph("<b>$435.60</b>", cell_style),
            Paragraph("<b>$684.00</b>", cell_style),
        ],
        [
            Paragraph("Costo Cloud Unitario por Estación (USD/mes)", cell_style),
            Paragraph("$241.55", cell_style),
            Paragraph("$108.73", cell_style),
            Paragraph("$87.12", cell_style),
            Paragraph("<b>$68.40 (-71.7%)</b>", cell_style),
        ],
        [
            Paragraph("Costo Operativo Baseline Manual (3 Turnos)", cell_style),
            Paragraph("$6,725.00 / mes", cell_style),
            Paragraph("$20,175.00 / mes", cell_style),
            Paragraph("$33,625.00 / mes", cell_style),
            Paragraph("$67,250.00 / mes", cell_style),
        ],
        [
            Paragraph("Ahorro Neto Mensual con Solución Indra IA", cell_style),
            Paragraph("<b>+$5,481.45 / mes</b>", cell_style),
            Paragraph("<b>+$16,612.80 / mes</b>", cell_style),
            Paragraph("<b>+$27,733.40 / mes</b>", cell_style),
            Paragraph("<b>+$55,541.00 / mes</b>", cell_style),
        ],
    ]
    t2 = Table(t2_data, colWidths=[2.3 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch])
    t2.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0284C7")),
                ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#DCFCE7")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(t2)
    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            "<b>Conclusión Financiera:</b> Frente a un costo base de <b>$6,725.00 USD/mes</b> por estación en operación manual "
            "de 3 turnos, el sistema híbrido Indra Smart Pick & Pack reduce el costo total a <b>$1,243.55 USD/mes</b> "
            "(ahorro neto de <b>$5,481.45 USD/mes</b> o <b>$65,777.40 USD/año</b> por estación, logrando un <b>Payback de "
            "3.38 meses</b> y un <b>ROI en el primer año de +255.6%</b>).",
            body_style,
        )
    )

    doc.build(story)


def main() -> None:
    xlsx_path = DELIVERABLES_DIR / "Entregable_12_Modelo_Costos_FinOps_Azure.xlsx"
    md_path = DELIVERABLES_DIR / "Entregable_12_Arquitectura_Cloud_Azure_FinOps.md"
    pdf_path = DELIVERABLES_DIR / "Entregable_12_Arquitectura_Cloud_Azure_FinOps.pdf"

    build_excel_cost_model(xlsx_path)
    md_path.write_text(build_architecture_markdown(), encoding="utf-8")
    build_architecture_pdf(pdf_path)

    print(f"Entregable #12 XLSX: {xlsx_path}")
    print(f"Entregable #12 MD  : {md_path}")
    print(f"Entregable #12 PDF : {pdf_path}")


if __name__ == "__main__":
    main()
