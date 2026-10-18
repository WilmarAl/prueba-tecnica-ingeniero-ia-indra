"""Configuración centralizada y separada del código (12-Factor App).

Cumple con el requisito del Reto 4: 'Código organizado por responsabilidades y
configuración separada del código'.
"""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Parámetros operativos y de configuración del sistema."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Identificación de planta y entorno
    app_env: str = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    plant_id: str = Field(default="PLANT-BOG-01", alias="PLANT_ID")
    line_id: str = Field(default="LINE-PACK-02", alias="LINE_ID")

    # Servicio de Inferencia de Visión (Reto 2 & Reto 4)
    vision_service_host: str = Field(default="0.0.0.0", alias="VISION_SERVICE_HOST")
    vision_service_port: int = Field(default=8000, alias="VISION_SERVICE_PORT")
    vision_service_url: str = Field(default="http://localhost:8000", alias="VISION_SERVICE_URL")
    model_artifact_path: str = Field(
        default="models/coil100_classical_v1.0.0.joblib",
        alias="MODEL_ARTIFACT_PATH",
    )
    model_card_path: str = Field(
        default="models/model_card.json",
        alias="MODEL_CARD_PATH",
    )
    model_version: str = Field(default="v1.0.0", alias="MODEL_VERSION")

    # Guardrails Determinísticos hacia el Brazo Robótico
    robot_angle_tolerance_deg: float = Field(default=10.0, alias="ROBOT_ANGLE_TOLERANCE_DEG")
    class_confidence_threshold: float = Field(default=0.70, alias="CLASS_CONFIDENCE_THRESHOLD")
    ood_distance_percentile_threshold: float = Field(
        default=95.0, alias="OOD_DISTANCE_PERCENTILE_THRESHOLD"
    )
    image_min_laplacian_var: float = Field(default=35.0, alias="IMAGE_MIN_LAPLACIAN_VAR")

    # Asistente Agéntico de Operación (Reto 3 & Reto 4)
    agent_service_host: str = Field(default="0.0.0.0", alias="AGENT_SERVICE_HOST")
    agent_service_port: int = Field(default=8001, alias="AGENT_SERVICE_PORT")
    agent_max_steps: int = Field(default=5, alias="AGENT_MAX_STEPS")
    agent_token_budget_per_turn: int = Field(default=4000, alias="AGENT_TOKEN_BUDGET_PER_TURN")
    agent_state_db_path: str = Field(
        default="data/operational_state.sqlite", alias="AGENT_STATE_DB_PATH"
    )

    # Proveedor LLM (Soporte nativo para Ollama local + Híbrido Determinístico + Cloud)
    llm_provider: str = Field(default="ollama", alias="LLM_PROVIDER")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.1:8b", alias="OLLAMA_MODEL")
    ollama_timeout_sec: int = Field(default=45, alias="OLLAMA_TIMEOUT_SEC")

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")


settings = Settings()
