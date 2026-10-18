"""Guardrails Determinísticos de Seguridad, RBAC, Anti-Loop y Tratamiento de Información Sensible (Reto 3).

Implementa en código determinístico todas las decisiones críticas que NUNCA deben quedar
en manos del LLM (Sección 6.2, 6.4 y 6.5):
1. Control de Acceso Basado en Roles (`RBAC_POLICY_MATRIX`) por cada una de las 11 tools.
2. Detección determinística de intentos de Prompt Injection / Jailbreak orientados a
   saltarse la aprobación humana explícita (HITL) o fijar umbrales inseguros.
3. Detección y corte automático de Loops Agénticos (`LoopAndBudgetGuard`) mediante firma
   SHA-256 canónica de `(tool_name, canonical_args)` y límite estricto de pasos/tokens.
4. Redacción de información sensible (PII, correos, teléfonos, tokens, claves API y tensores/base64)
   antes de persistir logs de auditoría o construir el contexto del modelo.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Set, Tuple

from src.agent_service.state_store import ExecutionState

# Roles válidos en planta (Sección 6.1)
VALID_ROLES: Set[str] = {
    "supervisor",
    "quality",
    "maintenance",
    "ml_engineer",
    "plant_manager",
    "viewer",
}

# Matriz determinística de permisos por Tool (RBAC)
RBAC_POLICY_MATRIX: Dict[str, Set[str]] = {
    # Lectura / Inferencia / Diagnóstico
    "inspect_and_predict_images": {"supervisor", "quality", "maintenance", "ml_engineer", "plant_manager"},
    "list_human_review_queue": {"supervisor", "quality", "ml_engineer", "plant_manager"},
    "query_shift_and_hourly_telemetry": {"supervisor", "quality", "maintenance", "ml_engineer", "plant_manager", "viewer"},
    "diagnose_camera_and_plc_health": {"supervisor", "maintenance", "quality", "ml_engineer", "plant_manager"},
    "search_plant_sops_and_manuals": {"supervisor", "quality", "maintenance", "ml_engineer", "plant_manager", "viewer"},
    "generate_shift_handover_report": {"supervisor", "plant_manager"},
    # Escritura / Operación / Gobierno
    "submit_human_relabeling": {"supervisor", "quality", "ml_engineer"},
    "create_or_escalate_maintenance_ticket": {"maintenance", "supervisor", "quality"},
    "evaluate_new_sku_onboarding": {"ml_engineer", "quality"},
    "request_critical_line_operation": {"supervisor", "ml_engineer", "plant_manager", "quality"},
    "confirm_critical_line_operation": {"supervisor", "plant_manager", "ml_engineer"},
}

# Patrones determinísticos de ataque / Prompt Injection (OWASP LLM01)
PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignora\s+(todas\s+)?las\s+(instrucciones|reglas)", re.IGNORECASE),
    re.compile(r"ignore\s+(all\s+)?(previous\s+)?(instructions|rules)", re.IGNORECASE),
    re.compile(r"sin\s+aprobaci[oó]n\s+humana", re.IGNORECASE),
    re.compile(r"bypass\s+(human\s+)?approval", re.IGNORECASE),
    re.compile(r"modo\s+desarrollador\s+sin\s+restricciones", re.IGNORECASE),
    re.compile(r"override\s+security\s+guardrail", re.IGNORECASE),
]

# Patrones para enmascarar PII o secretos en logs y contexto (Sección 6.4)
SENSITIVE_PATTERNS = [
    (re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"), "[REDACTED_EMAIL]"),
    (re.compile(r"\b(?:sk|apr|tok|key)-[a-zA-Z0-9]{12,}\b", re.IGNORECASE), "[REDACTED_SECRET_TOKEN]"),
    (re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[REDACTED_PHONE]"),
]


def verify_tool_permission(user_role: str, tool_name: str) -> Tuple[bool, str]:
    """Verifica de forma determinística si el rol tiene permiso para invocar la tool."""
    role_norm = user_role.strip().lower()
    if role_norm not in VALID_ROLES:
        return False, f"RBAC_INVALID_ROLE: El rol '{user_role}' no está reconocido en la planta."

    allowed = RBAC_POLICY_MATRIX.get(tool_name)
    if allowed is None:
        return False, f"UNKNOWN_TOOL_ERROR: La tool '{tool_name}' no existe en el catálogo autorizado."

    if role_norm not in allowed:
        return (
            False,
            f"RBAC_PERMISSION_DENIED: El rol '{role_norm}' no tiene autorización para ejecutar '{tool_name}'. "
            f"Roles permitidos: {sorted(allowed)}.",
        )
    return True, "AUTHORIZED"


def detect_prompt_injection_or_policy_bypass(message: str) -> Tuple[bool, List[str]]:
    """Detecta intentos de manipular al agente para omitir guardrails o aprobaciones humanas."""
    triggers: List[str] = []
    for pat in PROMPT_INJECTION_PATTERNS:
        if pat.search(message):
            triggers.append(f"PROMPT_INJECTION_DETECTED ({pat.pattern})")
    return len(triggers) > 0, triggers


def redact_sensitive_data(payload: Any) -> Any:
    """Enmascara PII, secretos y recorta cadenas Base64/tensores para proteger el contexto y la auditoría."""
    if isinstance(payload, str):
        if len(payload) > 500 and ("base64" in payload[:40].lower() or " " not in payload[:120]):
            return f"[OMITTED_BINARY_OR_BASE64_PAYLOAD_LEN_{len(payload)}]"
        out = payload
        for pat, repl in SENSITIVE_PATTERNS:
            out = pat.sub(repl, out)
        return out
    if isinstance(payload, dict):
        redacted = {}
        for k, v in payload.items():
            if k.lower() in ("image_base64", "raw_bytes", "api_key", "secret", "password", "filepath"):
                if k.lower() == "filepath":
                    # Solo conservar el nombre de archivo sin exponer rutas internas del servidor al LLM
                    redacted[k] = str(v).replace("\\", "/").split("/")[-1]
                else:
                    redacted[k] = "[REDACTED_OUT_OF_LLM_CONTEXT]"
            else:
                redacted[k] = redact_sensitive_data(v)
        return redacted
    if isinstance(payload, list):
        return [redact_sensitive_data(item) for item in payload]
    return payload


class LoopAndBudgetGuard:
    """Controlador determinístico de terminación, límite de pasos, presupuesto de tokens y loops (Sección 6.2 & 6.5)."""

    @staticmethod
    def compute_call_signature(tool_name: str, args: Dict[str, Any]) -> str:
        canonical = json.dumps(args, sort_keys=True, default=str)
        return f"{tool_name}:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:12]}"

    @classmethod
    def check_and_register_step(
        cls,
        state: ExecutionState,
        tool_name: str,
        args: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Valida antes de ejecutar una tool si se viola el límite de pasos, presupuesto o si hay un loop."""
        if state.step_count >= state.max_steps:
            state.terminated_reason = f"MAX_STEPS_EXCEEDED ({state.max_steps})"
            return (
                False,
                f"EXECUTION_LIMIT_REACHED: Se alcanzó el límite máximo de {state.max_steps} pasos por turno.",
            )

        if (state.prompt_tokens + state.completion_tokens) >= state.token_budget:
            state.terminated_reason = f"TOKEN_BUDGET_EXCEEDED ({state.token_budget})"
            return (
                False,
                f"TOKEN_BUDGET_REACHED: Se alcanzó el presupuesto máximo de {state.token_budget} tokens.",
            )

        sig = cls.compute_call_signature(tool_name, args)
        if sig in state.tool_call_signatures:
            state.loop_detected = True
            state.terminated_reason = f"AGENTIC_LOOP_DETECTED_ON_{tool_name}"
            return (
                False,
                f"AGENTIC_LOOP_ABORTED: Intento repetido de invocar '{tool_name}' con los mismos argumentos "
                f"(signature={sig}). Ejecución detenida determinísticamente.",
            )

        state.step_count += 1
        state.tool_call_signatures.append(sig)
        return True, "STEP_ALLOWED"
