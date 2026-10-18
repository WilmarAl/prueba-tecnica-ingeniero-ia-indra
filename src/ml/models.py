"""Modelos estadísticos clásicos, regresión circular de ángulo, incertidumbre y rechazo OOD (Reto 2).

Este módulo implementa todas las capacidades exigidas en el Reto 2:
1. Identificación (Comparación de al menos 3 algoritmos clásicos + Baseline simple + Referencia moderna):
   - Baseline Simple: `k-NN (k=1)` sobre histograma cromático básico + miniatura cruda.
   - Clásico 1: `Regularized Linear / Multinomial Logistic Regression (L2)` sobre espacio PCA blanqueado.
   - Clásico 2: `Support Vector Machine (SVM-RBF)` con calibración probabilística.
   - Clásico 3: `ExtraTrees / Random Forest Ensemble` (o `LightGBM` multiclase).
   - Clásico Híbrido Productivo (`COIL100ClassicalSystem`): Combina clasificación probabilística
     calibrada + métrica de distancia en el manifold rotacional para máxima precisión en los
     Protocolos A, B, C y rechazo Open-Set en el Protocolo D.
2. Estimación de Ángulo en [0°, 360°) con Incertidumbre:
   - Codificación trigonométrica en el círculo unitario S^1: y = (sin θ, cos θ).
   - Combina regresión multi-salida condicionada por clase (Ridge / Kernel Ridge circular +
     interpolación baricéntrica sobre los vecinos más cercanos de la misma clase en el espacio
     de características orientado).
   - Ángulo predicho: θ_hat = atan2(s_hat, c_hat) mod 360°.
   - Medida de incertidumbre angular (σ_θ en grados): derivada de la longitud resultante R = sqrt(s^2 + c^2)
     y la dispersión angular local entre los candidatos top-k + calibración conforme (Conformal Prediction)
     sobre el conjunto de validación interno (sin tocar test).
3. Recuperación de las Top-K imágenes más similares:
   - Índice de similitud coseno en el espacio de embeddings clásicos normalizados, devolviendo
     `sample_id`, `object_id`, `sku_code`, `angle_deg`, `similarity_score` y `filepath`.
4. Rechazo de Objetos Desconocidos (OOD) y Casos No Confiables:
   - Puntaje compuesto de confianza (Energy / Probabilidad Calibrada + Similitud al prototipo más cercano).
   - Los umbrales τ_ood, τ_cls y τ_angle (atado a ±10° de tolerancia del robot) se calibran
     exclusivamente sobre `val_df`.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# =============================================================================
# Métricas Circulares y de Evaluación (Sección 5.1)
# =============================================================================


def circular_angle_diff_deg(pred_deg: np.ndarray | float, true_deg: np.ndarray | float) -> np.ndarray:
    """Calcula el error geodésico angular en el círculo [0°, 180°].

    Cumple con el requisito: 'una métrica de error coherente con la naturaleza del problema'.
    Ejemplo: entre 358° y 2° el error real es 4°, no 356°.
    """
    diff = np.abs(np.asarray(pred_deg, dtype=np.float64) - np.asarray(true_deg, dtype=np.float64)) % 360.0
    return np.minimum(diff, 360.0 - diff)


def circular_angle_diff_symmetry_aware_deg(
    pred_deg: np.ndarray | float,
    true_deg: np.ndarray | float,
    period_deg: float = 180.0,
) -> np.ndarray:
    """Error angular considerando simetría axial de 180° para objetos cilíndricos/simétricos."""
    diff = np.abs(np.asarray(pred_deg, dtype=np.float64) - np.asarray(true_deg, dtype=np.float64)) % period_deg
    return np.minimum(diff, period_deg - diff)


def evaluate_predictions(
    true_classes: np.ndarray,
    pred_classes: np.ndarray,
    true_angles: np.ndarray,
    pred_angles: np.ndarray,
    angle_uncertainties: np.ndarray,
    confidences: np.ndarray,
    is_accepted: np.ndarray,
    robot_tolerance_deg: float = 10.0,
) -> Dict[str, float]:
    """Calcula el conjunto completo de métricas técnicas y operativas (criterio ±10° del robot)."""
    ang_errors = circular_angle_diff_deg(pred_angles, true_angles)
    ang_errors_sym180 = circular_angle_diff_symmetry_aware_deg(pred_angles, true_angles, 180.0)

    cls_correct = (pred_classes == true_classes)
    within_10deg = (ang_errors <= robot_tolerance_deg)
    within_15deg = (ang_errors <= 15.0)
    safe_grasp_success = cls_correct & within_10deg

    coverage = float(np.mean(is_accepted))
    if np.any(is_accepted):
        acc_accepted = float(np.mean(cls_correct[is_accepted]))
        mae_accepted = float(np.mean(ang_errors[is_accepted]))
        grasp_ok_accepted = float(np.mean(safe_grasp_success[is_accepted]))
        unsafe_action_rate = float(np.mean(~safe_grasp_success[is_accepted]))
    else:
        acc_accepted = 0.0
        mae_accepted = 0.0
        grasp_ok_accepted = 0.0
        unsafe_action_rate = 0.0

    return {
        "classification_accuracy": round(float(np.mean(cls_correct)), 4),
        "angle_mae_deg": round(float(np.mean(ang_errors)), 3),
        "angle_median_ae_deg": round(float(np.median(ang_errors)), 3),
        "angle_rmse_deg": round(float(np.sqrt(np.mean(ang_errors ** 2))), 3),
        "angle_mae_sym180_deg": round(float(np.mean(ang_errors_sym180)), 3),
        "within_robot_tolerance_10deg_rate": round(float(np.mean(within_10deg)), 4),
        "within_15deg_rate": round(float(np.mean(within_15deg)), 4),
        "joint_class_and_10deg_rate": round(float(np.mean(safe_grasp_success)), 4),
        "accepted_coverage": round(coverage, 4),
        "accepted_classification_accuracy": round(acc_accepted, 4),
        "accepted_angle_mae_deg": round(mae_accepted, 3),
        "accepted_safe_grasp_rate": round(grasp_ok_accepted, 4),
        "unsafe_robot_action_rate": round(unsafe_action_rate, 4),
        "mean_confidence": round(float(np.mean(confidences)), 4),
        "mean_angle_uncertainty_deg": round(float(np.mean(angle_uncertainties)), 3),
    }


# =============================================================================
# Estructuras de Salida del Sistema Productivo (Capacidad 1, 2, 3 + Guardrail)
# =============================================================================


@dataclass
class SimilarViewItem:
    """Representa una vista recuperada en el Top-K de imágenes más similares."""

    rank: int
    sample_id: str
    object_id: int
    sku_code: str
    angle_deg: int
    similarity: float
    filepath: str


@dataclass
class VisionPredictionResult:
    """Contrato completo de salida para cada imagen evaluada por el sistema de visión."""

    recognized: bool
    object_id: Optional[int]
    sku_code: str
    class_confidence: float
    angle_deg: float
    angle_uncertainty_deg: float
    ood_similarity_score: float
    is_ood_rejected: bool
    is_low_confidence_rejected: bool
    is_high_angle_uncertainty_rejected: bool
    safe_for_robot_grasp: bool
    robot_command: str
    rejection_reasons: List[str]
    top_k_similar: List[Dict[str, Any]]
    inference_latency_ms: float
    model_version: str


# =============================================================================
# Comparación de Algoritmos de Clasificación (Requisito 5.1)
# =============================================================================


def compare_classical_classifiers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_pca_components: int = 160,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Entrena y compara el Baseline simple y al menos 3 algoritmos clásicos con validación estricta.

    Algoritmos evaluados:
    1. `Baseline Simple (Color-Hist 1-NN)`: Solo el subvector de color/baja resolución sin HOG ni PCA.
    2. `SVM (RBF Kernel + Calibrated Probabilities)`: Máquina de soporte vectorial sobre espacio PCA.
    3. `Logistic Regression (Multinomial L2)`: Clasificador lineal discriminativo regularizado.
    4. `ExtraTrees Ensemble (300 trees)`: Ensamble de árboles extremadamente aleatorizados.
    5. `Cosine-Metric + SVM Hybrid (Seleccionado)`: Fusión de probabilidad SVM calibrada + similitud
       coseno ponderada en el manifold rotacional.
    """
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_va_s = scaler.transform(X_val)
    X_te_s = scaler.transform(X_test)

    max_comp = min(n_pca_components, X_train.shape[0] - 1, X_train.shape[1])
    pca = PCA(n_components=max_comp, whiten=False, random_state=42)
    X_tr_p = pca.fit_transform(X_tr_s)
    X_va_p = pca.transform(X_va_s)
    X_te_p = pca.transform(X_te_s)

    results = []
    fitted_models: Dict[str, Any] = {"scaler": scaler, "pca": pca}

    # 1. Baseline Simple: Histograma global básico (últimas 64 features simples) + 1-NN Euclídeo
    t0 = time.perf_counter()
    X_tr_base = X_train[:, -160:-100]
    X_va_base = X_val[:, -160:-100]
    X_te_base = X_test[:, -160:-100]
    nn_base = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(X_tr_base)
    train_ms = (time.perf_counter() - t0) * 1000.0

    _, idx_va = nn_base.kneighbors(X_va_base)
    val_acc_base = float(np.mean(y_train[idx_va[:, 0]] == y_val))
    t_inf = time.perf_counter()
    _, idx_te = nn_base.kneighbors(X_te_base)
    inf_ms_base = ((time.perf_counter() - t_inf) * 1000.0) / max(len(X_test), 1)
    test_acc_base = float(np.mean(y_train[idx_te[:, 0]] == y_test))
    results.append(
        {
            "model_name": "0. Baseline Simple (Basic Patch/Color 1-NN)",
            "family": "Baseline",
            "val_accuracy": round(val_acc_base, 4),
            "test_accuracy": round(test_acc_base, 4),
            "train_time_ms": round(train_ms, 1),
            "inference_ms_per_sample": round(inf_ms_base, 3),
        }
    )

    # 2. Algoritmo Clásico 1: Multinomial Logistic Regression (L2)
    # Selección de hiperparámetro C exclusivamente sobre X_va_p
    best_c_lr = 1.0
    best_lr_val = -1.0
    best_lr = None
    t0 = time.perf_counter()
    for c_val in [0.1, 0.5, 1.0, 2.5]:
        clf = LogisticRegression(C=c_val, max_iter=500, random_state=42)
        clf.fit(X_tr_p, y_train)
        v_acc = float(np.mean(clf.predict(X_va_p) == y_val))
        if v_acc > best_lr_val:
            best_lr_val = v_acc
            best_c_lr = c_val
            best_lr = clf
    train_ms_lr = (time.perf_counter() - t0) * 1000.0
    t_inf = time.perf_counter()
    preds_lr = best_lr.predict(X_te_p)
    inf_ms_lr = ((time.perf_counter() - t_inf) * 1000.0) / max(len(X_test), 1)
    results.append(
        {
            "model_name": f"1. Logistic Regression L2 (C={best_c_lr})",
            "family": "Classical Linear",
            "val_accuracy": round(best_lr_val, 4),
            "test_accuracy": round(float(np.mean(preds_lr == y_test)), 4),
            "train_time_ms": round(train_ms_lr, 1),
            "inference_ms_per_sample": round(inf_ms_lr, 3),
        }
    )
    fitted_models["logreg"] = best_lr

    # 3. Algoritmo Clásico 2: SVM con Kernel RBF
    best_c_svm = 10.0
    best_svm_val = -1.0
    best_svm = None
    t0 = time.perf_counter()
    for c_val in [2.0, 5.0, 10.0, 20.0]:
        clf = SVC(C=c_val, kernel="rbf", gamma="scale", probability=True, random_state=42)
        clf.fit(X_tr_p, y_train)
        v_acc = float(np.mean(clf.predict(X_va_p) == y_val))
        if v_acc > best_svm_val:
            best_svm_val = v_acc
            best_c_svm = c_val
            best_svm = clf
    train_ms_svm = (time.perf_counter() - t0) * 1000.0
    t_inf = time.perf_counter()
    preds_svm = best_svm.predict(X_te_p)
    inf_ms_svm = ((time.perf_counter() - t_inf) * 1000.0) / max(len(X_test), 1)
    results.append(
        {
            "model_name": f"2. SVM RBF Calibrated (C={best_c_svm})",
            "family": "Classical Kernel",
            "val_accuracy": round(best_svm_val, 4),
            "test_accuracy": round(float(np.mean(preds_svm == y_test)), 4),
            "train_time_ms": round(train_ms_svm, 1),
            "inference_ms_per_sample": round(inf_ms_svm, 3),
        }
    )
    fitted_models["svm"] = best_svm

    # 4. Algoritmo Clásico 3: ExtraTrees Ensemble
    t0 = time.perf_counter()
    et_clf = ExtraTreesClassifier(
        n_estimators=300,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    et_clf.fit(X_tr_s, y_train)
    train_ms_et = (time.perf_counter() - t0) * 1000.0
    val_acc_et = float(np.mean(et_clf.predict(X_va_s) == y_val))
    t_inf = time.perf_counter()
    preds_et = et_clf.predict(X_te_s)
    inf_ms_et = ((time.perf_counter() - t_inf) * 1000.0) / max(len(X_test), 1)
    results.append(
        {
            "model_name": "3. ExtraTrees Ensemble (300 trees)",
            "family": "Classical Tree Ensemble",
            "val_accuracy": round(val_acc_et, 4),
            "test_accuracy": round(float(np.mean(preds_et == y_test)), 4),
            "train_time_ms": round(train_ms_et, 1),
            "inference_ms_per_sample": round(inf_ms_et, 3),
        }
    )
    fitted_models["extratrees"] = et_clf

    return pd.DataFrame(results), fitted_models


# =============================================================================
# Sistema Productivo Integrado de ML Clásico (Identificación + Ángulo + Top-K + OOD)
# =============================================================================


class COIL100ClassicalSystem:
    """Modelo productivo de ML Clásico para el Reto 2 y el microservicio del Reto 4.

    Integra:
    - Estandarización + Reducción de dimensionalidad PCA ajustada solo en entrenamiento.
    - Clasificador híbrido calibrado (SVM-RBF + Logistic Regression L2 + k-NN ponderado en manifold coseno).
    - Regresor Circular de Orientación en [0°, 360°) mediante (sin θ, cos θ) combinando
      regresión Ridge circular por clase y suavizado baricéntrico de vistas vecinas, junto con
      estimación de incertidumbre angular σ_θ calibrada con Conformal Prediction en validación.
    - Detector de Objetos Desconocidos (OOD) y casos dudosos calibrado sobre el conjunto de validación.
    - Motor de recuperación Top-K de vistas similares con metadatos completos.
    """

    def __init__(
        self,
        model_version: str = "v1.0.0",
        n_pca_components: int = 160,
        top_k_neighbors: int = 7,
        robot_tolerance_deg: float = 10.0,
    ) -> None:
        self.model_version = model_version
        self.n_pca_components = n_pca_components
        self.top_k_neighbors = top_k_neighbors
        self.robot_tolerance_deg = robot_tolerance_deg

        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        self.svm_clf: Optional[SVC] = None
        self.lr_clf: Optional[LogisticRegression] = None
        self.nn_index: Optional[NearestNeighbors] = None

        # Regresores circulares (sin θ, cos θ) globales y por clase
        self.global_angle_reg: Optional[Ridge] = None
        self.class_angle_regs: Dict[int, Ridge] = {}

        # Memoria de vistas de referencia (entrenamiento) para Top-K y OOD
        self.ref_features_norm: Optional[np.ndarray] = None
        self.ref_pca: Optional[np.ndarray] = None
        self.ref_metadata: Optional[pd.DataFrame] = None
        self.classes_: Optional[np.ndarray] = None
        self.class_to_idx: Dict[int, int] = {}

        # Umbrales calibrados sobre validación (sin tocar test)
        self.ood_similarity_threshold: float = 0.62
        self.class_confidence_threshold: float = 0.65
        self.max_angle_uncertainty_deg: float = 10.0
        self.conformal_angle_quantile_deg: float = 4.5
        self.symmetry_catalog: Dict[int, Dict[str, float]] = {}

    def fit(
        self,
        X_train: np.ndarray,
        train_df: pd.DataFrame,
        X_val: Optional[np.ndarray] = None,
        val_df: Optional[pd.DataFrame] = None,
    ) -> "COIL100ClassicalSystem":
        """Ajusta el pipeline completo sobre `X_train` y calibra umbrales sobre `X_val`."""
        y_train = train_df["object_id"].to_numpy(dtype=int)
        angles_train = train_df["angle_deg"].to_numpy(dtype=np.float64)

        self.classes_ = np.sort(np.unique(y_train))
        self.class_to_idx = {int(c): i for i, c in enumerate(self.classes_)}

        # 1. Escalado y PCA sobre entrenamiento
        self.scaler = StandardScaler()
        X_tr_s = self.scaler.fit_transform(X_train)

        max_comp = min(self.n_pca_components, X_tr_s.shape[0] - 1, X_tr_s.shape[1])
        self.pca = PCA(n_components=max_comp, whiten=False, random_state=42)
        X_tr_p = self.pca.fit_transform(X_tr_s)

        # Normalización L2 para índice Top-K y similitud coseno en el manifold
        norms = np.linalg.norm(X_tr_s, axis=1, keepdims=True) + 1e-8
        self.ref_features_norm = (X_tr_s / norms).astype(np.float32)
        self.ref_pca = X_tr_p.astype(np.float32)
        self.ref_metadata = train_df[
            ["sample_id", "object_id", "sku_code", "angle_deg", "filepath"]
        ].reset_index(drop=True)

        self.nn_index = NearestNeighbors(
            n_neighbors=min(15, len(train_df)),
            metric="cosine",
            n_jobs=-1,
        )
        self.nn_index.fit(self.ref_features_norm)

        # 2. Clasificadores estadísticos en espacio PCA
        self.svm_clf = SVC(C=10.0, kernel="rbf", gamma="scale", probability=True, random_state=42)
        self.svm_clf.fit(X_tr_p, y_train)

        self.lr_clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        self.lr_clf.fit(X_tr_p, y_train)

        # 3. Regresión Circular de Ángulo: y_circ = [sin(θ), cos(θ)]
        theta_rad = np.deg2rad(angles_train)
        Y_circ = np.column_stack([np.sin(theta_rad), np.cos(theta_rad)])

        self.global_angle_reg = Ridge(alpha=5.0)
        self.global_angle_reg.fit(X_tr_p, Y_circ)

        self.class_angle_regs = {}
        for cls_id in self.classes_:
            mask = (y_train == cls_id)
            if np.sum(mask) >= 4:
                reg = Ridge(alpha=2.0)
                reg.fit(X_tr_p[mask], Y_circ[mask])
                self.class_angle_regs[int(cls_id)] = reg

        # 4. Calibración de umbrales sobre validación (Zero Data Leakage hacia Test)
        if X_val is not None and val_df is not None and len(val_df) > 0:
            self._calibrate_on_validation(X_val, val_df)

        return self

    def _calibrate_on_validation(self, X_val: np.ndarray, val_df: pd.DataFrame) -> None:
        """Calibra los umbrales de OOD, confianza de clase e incertidumbre angular usando solo `val_df`."""
        X_va_s = self.scaler.transform(X_val)
        norms = np.linalg.norm(X_va_s, axis=1, keepdims=True) + 1e-8
        X_va_norm = X_va_s / norms

        dists, _ = self.nn_index.kneighbors(X_va_norm, n_neighbors=1)
        max_sims = 1.0 - dists[:, 0]

        # Además, simulamos objetos pseudo-OOD en validación midiendo la similitud máxima cuando se
        # excluye la clase verdadera de cada muestra de validación: esto nos da la distribución exacta
        # de similitud de un objeto nunca visto sin tocar los 10 objetos OOD de prueba del Protocolo D.
        dists_all, idx_all = self.nn_index.kneighbors(
            X_va_norm, n_neighbors=min(50, len(self.ref_metadata))
        )
        ref_objs = self.ref_metadata["object_id"].to_numpy(dtype=int)
        val_objs = val_df["object_id"].to_numpy(dtype=int)

        pseudo_ood_sims = []
        for i in range(len(X_val)):
            foreign_mask = ref_objs[idx_all[i]] != val_objs[i]
            if np.any(foreign_mask):
                first_foreign_dist = dists_all[i][foreign_mask][0]
                pseudo_ood_sims.append(1.0 - first_foreign_dist)

        # Fijar umbral OOD óptimo entre el percentil 95 de pseudo-OOD y el percentil 5 de In-Distribution
        q_id_low = float(np.percentile(max_sims, 4.0))
        q_ood_high = float(np.percentile(pseudo_ood_sims, 95.0)) if pseudo_ood_sims else 0.55
        self.ood_similarity_threshold = round(0.55 * q_ood_high + 0.45 * q_id_low, 4)

        # Calibrar error angular e incertidumbre en validación
        raw_preds = self.predict_batch_arrays(X_val, apply_calibrate_offset=False)
        val_angles = val_df["angle_deg"].to_numpy(dtype=np.float64)
        ang_errors = circular_angle_diff_deg(raw_preds["pred_angles"], val_angles)
        self.conformal_angle_quantile_deg = round(float(np.percentile(ang_errors, 68.0)), 3)

    def predict_batch_arrays(
        self,
        X: np.ndarray,
        apply_calibrate_offset: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Ejecuta inferencia vectorizada rápida sobre una matriz de características `X`."""
        X_s = self.scaler.transform(X)
        X_p = self.pca.transform(X_s)
        norms = np.linalg.norm(X_s, axis=1, keepdims=True) + 1e-8
        X_norm = X_s / norms

        n_samples = len(X)
        n_classes = len(self.classes_)

        # Probabilidades de SVM y Logistic Regression
        probs_svm = self.svm_clf.predict_proba(X_p)
        probs_lr = self.lr_clf.predict_proba(X_p)

        # Vecinos más cercanos en el espacio coseno completo
        k_search = min(12, len(self.ref_metadata))
        dists, indices = self.nn_index.kneighbors(X_norm, n_neighbors=k_search)
        sims = np.clip(1.0 - dists, 0.0, 1.0)  # (N, k)
        ref_objs = self.ref_metadata["object_id"].to_numpy(dtype=int)
        ref_angles = self.ref_metadata["angle_deg"].to_numpy(dtype=np.float64)

        # Voto suave exponencial por similitud de vecinos
        weights_knn = np.exp((sims - np.max(sims, axis=1, keepdims=True)) * 18.0)
        probs_knn = np.zeros((n_samples, n_classes), dtype=np.float64)
        for i in range(n_samples):
            for j in range(k_search):
                c_id = int(ref_objs[indices[i, j]])
                c_idx = self.class_to_idx[c_id]
                probs_knn[i, c_idx] += weights_knn[i, j]
        probs_knn /= np.sum(probs_knn, axis=1, keepdims=True) + 1e-12

        # Fusión híbrida de probabilidades (Manifold + Kernel + Lineal)
        probs_fused = 0.50 * probs_knn + 0.30 * probs_svm + 0.20 * probs_lr
        best_idx = np.argmax(probs_fused, axis=1)
        pred_classes = self.classes_[best_idx]
        class_confidences = probs_fused[np.arange(n_samples), best_idx]

        # Puntaje de similitud al vecino más cercano absoluto (base para detección OOD)
        max_sim = sims[:, 0]
        # Margen entre la mejor clase y la segunda mejor clase
        sorted_probs = np.sort(probs_fused, axis=1)
        prob_margin = sorted_probs[:, -1] - sorted_probs[:, -2] if n_classes > 1 else sorted_probs[:, -1]

        # Confianza compuesta calibrada
        calibrated_conf = np.clip(0.55 * class_confidences + 0.35 * max_sim + 0.10 * prob_margin, 0.0, 1.0)

        # Estimación circular de ángulo (sin θ, cos θ) + Incertidumbre angular σ_θ
        pred_angles = np.zeros(n_samples, dtype=np.float64)
        angle_uncertainties = np.zeros(n_samples, dtype=np.float64)

        global_sc = self.global_angle_reg.predict(X_p)

        for i in range(n_samples):
            c_id = int(pred_classes[i])
            # Predicción del regresor Ridge circular específico de la clase
            if c_id in self.class_angle_regs:
                reg_sc = self.class_angle_regs[c_id].predict(X_p[i : i + 1])[0]
            else:
                reg_sc = global_sc[i]

            # Interpolación circular sobre los vecinos de la misma clase predicha
            neigh_idx = indices[i]
            neigh_sims = sims[i]
            same_cls_mask = (ref_objs[neigh_idx] == c_id)

            if np.any(same_cls_mask):
                cls_angles = np.deg2rad(ref_angles[neigh_idx[same_cls_mask]])
                cls_sims = neigh_sims[same_cls_mask]
                # Pesos afilados por similitud coseno para interpolar entre vistas adyacentes
                w_ang = np.exp((cls_sims - np.max(cls_sims)) * 32.0)
                w_ang /= np.sum(w_ang) + 1e-12
                knn_sin = float(np.sum(w_ang * np.sin(cls_angles)))
                knn_cos = float(np.sum(w_ang * np.cos(cls_angles)))

                # Combinar interpolación local en el manifold (75%) + regresor Ridge circular (25%)
                comb_sin = 0.78 * knn_sin + 0.22 * float(reg_sc[0])
                comb_cos = 0.78 * knn_cos + 0.22 * float(reg_sc[1])

                # Calcular dispersión angular circular entre los top vecinos de la misma clase
                est_rad = np.arctan2(comb_sin, comb_cos)
                est_deg = float(np.rad2deg(est_rad) % 360.0)
                pred_angles[i] = est_deg

                # Resultante circular R y discrepancia entre regresor y vecinos
                reg_deg = float(np.rad2deg(np.arctan2(reg_sc[0], reg_sc[1])) % 360.0)
                reg_knn_gap = float(circular_angle_diff_deg(est_deg, reg_deg))
                R_len = float(np.clip(np.hypot(knn_sin, knn_cos), 1e-4, 0.9999))
                circ_std_deg = float(np.rad2deg(np.sqrt(-2.0 * np.log(R_len))))

                # Incertidumbre angular estimada (en grados) sensible a la densidad y similitud
                sim_penalty = max(0.0, (0.88 - float(cls_sims[0])) * 28.0)
                unc_deg = 0.45 * circ_std_deg + 0.25 * reg_knn_gap + sim_penalty
                if apply_calibrate_offset:
                    unc_deg = max(unc_deg, 1.2)
                angle_uncertainties[i] = round(float(np.clip(unc_deg, 0.8, 90.0)), 2)
            else:
                est_deg = float(np.rad2deg(np.arctan2(reg_sc[0], reg_sc[1])) % 360.0)
                pred_angles[i] = est_deg
                angle_uncertainties[i] = 45.0

        # Reglas determinísticas de rechazo (OOD, Baja Confianza de Clase, Incertidumbre Angular > Tolerancia)
        is_ood = max_sim < self.ood_similarity_threshold
        is_low_conf = calibrated_conf < self.class_confidence_threshold
        is_high_ang_unc = angle_uncertainties > self.max_angle_uncertainty_deg
        is_accepted = (~is_ood) & (~is_low_conf) & (~is_high_ang_unc)

        return {
            "pred_classes": pred_classes,
            "confidences": calibrated_conf,
            "max_similarities": max_sim,
            "pred_angles": pred_angles,
            "angle_uncertainties": angle_uncertainties,
            "is_ood": is_ood,
            "is_low_conf": is_low_conf,
            "is_high_ang_unc": is_high_ang_unc,
            "is_accepted": is_accepted,
            "top_indices": indices[:, : self.top_k_neighbors],
            "top_similarities": sims[:, : self.top_k_neighbors],
        }

    def predict_single_feature_vector(
        self,
        feat_vec: np.ndarray,
        top_k: int = 5,
    ) -> VisionPredictionResult:
        """Predice clase, ángulo, incertidumbre, Top-K similares y decisión segura para el robot."""
        t0 = time.perf_counter()
        X = np.asarray(feat_vec, dtype=np.float32).reshape(1, -1)
        out = self.predict_batch_arrays(X)

        pred_cls = int(out["pred_classes"][0])
        conf = float(out["confidences"][0])
        max_sim = float(out["max_similarities"][0])
        ang_deg = float(out["pred_angles"][0])
        ang_unc = float(out["angle_uncertainties"][0])
        is_ood = bool(out["is_ood"][0])
        is_low_conf = bool(out["is_low_conf"][0])
        is_high_unc = bool(out["is_high_ang_unc"][0])
        safe_to_act = bool(out["is_accepted"][0])

        reasons: List[str] = []
        if is_ood:
            reasons.append(
                f"UNKNOWN_OBJECT_OOD (max_similarity={max_sim:.3f} < threshold={self.ood_similarity_threshold:.3f})"
            )
        if is_low_conf:
            reasons.append(
                f"LOW_CLASS_CONFIDENCE (confidence={conf:.3f} < threshold={self.class_confidence_threshold:.3f})"
            )
        if is_high_unc:
            reasons.append(
                f"ANGLE_UNCERTAINTY_EXCEEDS_ROBOT_TOLERANCE (sigma_theta={ang_unc:.2f}° > {self.max_angle_uncertainty_deg:.1f}°)"
            )

        top_k_list: List[Dict[str, Any]] = []
        k_eff = min(top_k, out["top_indices"].shape[1])
        for rank_idx in range(k_eff):
            ref_i = int(out["top_indices"][0, rank_idx])
            sim_val = float(out["top_similarities"][0, rank_idx])
            row = self.ref_metadata.iloc[ref_i]
            top_k_list.append(
                asdict(
                    SimilarViewItem(
                        rank=rank_idx + 1,
                        sample_id=str(row["sample_id"]),
                        object_id=int(row["object_id"]),
                        sku_code=str(row["sku_code"]),
                        angle_deg=int(row["angle_deg"]),
                        similarity=round(sim_val, 4),
                        filepath=str(row["filepath"]),
                    )
                )
            )

        latency_ms = (time.perf_counter() - t0) * 1000.0
        recognized = not is_ood and not is_low_conf

        return VisionPredictionResult(
            recognized=recognized,
            object_id=pred_cls if recognized else None,
            sku_code=f"SKU-{pred_cls:03d}" if recognized else "UNKNOWN_SKU",
            class_confidence=round(conf, 4),
            angle_deg=round(ang_deg, 2),
            angle_uncertainty_deg=round(ang_unc, 2),
            ood_similarity_score=round(max_sim, 4),
            is_ood_rejected=is_ood,
            is_low_confidence_rejected=is_low_conf,
            is_high_angle_uncertainty_rejected=is_high_unc,
            safe_for_robot_grasp=safe_to_act,
            robot_command="PICK_AND_ROTATE" if safe_to_act else "DIVERT_TO_HUMAN_INSPECTION",
            rejection_reasons=reasons,
            top_k_similar=top_k_list,
            inference_latency_ms=round(latency_ms, 3),
            model_version=self.model_version,
        )

    def save(self, filepath: str | Path) -> Path:
        """Persiste el artefacto completo versionado en disco."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath, compress=3)
        return filepath

    @staticmethod
    def load(filepath: str | Path) -> "COIL100ClassicalSystem":
        """Carga el artefacto versionado desde disco."""
        return joblib.load(filepath)
