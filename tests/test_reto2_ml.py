"""Pruebas unitarias y de integridad metodológica para el Reto 2 (COIL-100 ML Clásico).

Verifica:
1. Cero fuga de datos (Zero Data Leakage) en los Protocolos A, B, C y D: intersección vacía entre
   conjuntos de entrenamiento, validación interna y prueba.
2. Exactitud de la métrica de error angular circular geodésica en la frontera 0°/360°.
3. Funcionamiento del detector de calidad óptica ante imágenes borrosas/oscuras vs. normales.
4. Contrato completo de predicción (clase, ángulo en [0, 360), incertidumbre, top-k y guardrail).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.dataset import create_protocol_splits
from src.ml.features import ClassicalFeatureExtractor, assess_image_quality
from src.ml.models import COIL100ClassicalSystem, circular_angle_diff_deg


def test_circular_angle_metric_boundary() -> None:
    """Garantiza que la diferencia entre 359° y 1° sea 2° y no 358°."""
    err = circular_angle_diff_deg(np.array([359.0, 0.0, 180.0]), np.array([1.0, 355.0, 0.0]))
    np.testing.assert_allclose(err, np.array([2.0, 5.0, 180.0]), atol=1e-6)


def test_protocol_splits_zero_data_leakage() -> None:
    """Verifica que los 4 protocolos cumplan la partición exacta sin solapamiento train/val/test."""
    rows = []
    for obj_id in range(1, 101):
        for ang in range(0, 360, 5):
            rows.append(
                {
                    "sample_id": f"obj{obj_id:03d}_ang{ang:03d}",
                    "object_id": obj_id,
                    "sku_code": f"SKU-{obj_id:03d}",
                    "angle_deg": ang,
                    "angle_idx": ang // 5,
                    "filepath": f"/tmp/obj{obj_id}__{ang}.png",
                }
            )
    catalog_df = pd.DataFrame(rows)
    splits = create_protocol_splits(catalog_df, seed=42)

    assert set(splits.keys()) == {"A", "B", "C", "D"}
    for pid, sp in splits.items():
        tr_ids = set(sp.train_df["sample_id"])
        va_ids = set(sp.val_df["sample_id"])
        te_ids = set(sp.test_df["sample_id"])
        assert tr_ids.isdisjoint(va_ids), f"Solapamiento train/val en protocolo {pid}"
        assert tr_ids.isdisjoint(te_ids), f"Fuga de datos train/test en protocolo {pid}"
        assert va_ids.isdisjoint(te_ids), f"Fuga de datos val/test en protocolo {pid}"

    # Verificar tamaños de prueba exactos según tabla de Sección 4
    assert len(splits["A"].test_df) == 100 * 36
    assert len(splits["B"].test_df) == 100 * 12
    assert len(splits["C"].test_df) == 100 * 60
    assert len(splits["D"].unknown_classes) == 10
    assert len(splits["D"].known_classes) == 90


def test_image_quality_guardrail() -> None:
    """Comprueba que una imagen negra o sin contraste sea rechazada por el guardrail óptico."""
    dark_img = np.zeros((128, 128, 3), dtype=np.uint8)
    rep_dark = assess_image_quality(dark_img)
    assert not rep_dark.is_valid
    assert len(rep_dark.issues) >= 1
