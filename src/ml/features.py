"""Extracción de características clásicas para reconocimiento de objetos y estimación angular (Reto 2).

Diseñado específicamente para el problema COIL-100 en planta de empaque:
1. `HOG (Histogram of Oriented Gradients)` sobre la imagen en escala de grises:
   Captura la estructura geométrica local y los bordes orientados, esenciales tanto para
   discriminar forma de SKU como para estimar el ángulo de rotación en [0°, 360°).
2. `Spatial Pyramid HSV & RGB Color Histograms` (Rejilla 2x2 + Global):
   Captura la distribución cromática invariante y localizada (etiquetas de envases, tapas,
   logos asimétricos que rompen la ambigüedad 0° vs 180°).
3. `Momentos Geométricos y de Hu / Perfiles Radiales-Azimutales`:
   Capturan el centroide de masa cromática, la orientación del eje principal de inercia (momento μ_11, μ_20, μ_02)
   y la firma azimutal en 36 sectores angulares (10° por sector alrededor del centro del objeto),
   lo que proporciona una señal direccional explícita para la regresión circular (sin θ, cos θ).
4. `Diagnóstico Óptico de Calidad de Imagen (Guardrail de Entrada)`:
   Calcula determinísticamente nitidez (varianza del Laplaciano), exposición media, saturación y
   relación señal/fondo para rechazar imágenes borrosas, sobreexpuestas o vacías antes de actuar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import hog
from scipy.ndimage import laplace


@dataclass(frozen=True)
class ImageQualityReport:
    """Resultado determinístico de validación de calidad óptica de una imagen de cámara."""

    is_valid: bool
    laplacian_variance: float
    mean_brightness: float
    foreground_ratio: float
    saturation_mean: float
    issues: List[str]


def assess_image_quality(
    img_rgb: np.ndarray,
    min_laplacian_var: float = 25.0,
    min_brightness: float = 8.0,
    max_brightness: float = 245.0,
    min_foreground_ratio: float = 0.03,
) -> ImageQualityReport:
    """Evalúa determinísticamente si la imagen capturada por la cámara es apta para inferencia."""
    if img_rgb is None or img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        return ImageQualityReport(
            is_valid=False,
            laplacian_variance=0.0,
            mean_brightness=0.0,
            foreground_ratio=0.0,
            saturation_mean=0.0,
            issues=["INVALID_CHANNELS_OR_EMPTY_FRAME"],
        )

    gray_f = rgb2gray(img_rgb) * 255.0
    lap_var = float(np.var(laplace(gray_f)))
    mean_b = float(np.mean(gray_f))
    fg_mask = gray_f > 18.0
    fg_ratio = float(np.mean(fg_mask))
    hsv = rgb2hsv(img_rgb)
    sat_mean = float(np.mean(hsv[:, :, 1][fg_mask])) if np.any(fg_mask) else 0.0

    issues: List[str] = []
    if lap_var < min_laplacian_var:
        issues.append(f"LOW_SHARPNESS_BLUR (laplacian_var={lap_var:.1f} < {min_laplacian_var:.1f})")
    if mean_b < min_brightness:
        issues.append(f"UNDEREXPOSED_DARK_FRAME (brightness={mean_b:.1f})")
    if mean_b > max_brightness:
        issues.append(f"OVEREXPOSED_GLARE_FRAME (brightness={mean_b:.1f})")
    if fg_ratio < min_foreground_ratio:
        issues.append(f"EMPTY_CONVEYOR_OR_TINY_OBJECT (fg_ratio={fg_ratio:.3f})")

    return ImageQualityReport(
        is_valid=len(issues) == 0,
        laplacian_variance=round(lap_var, 2),
        mean_brightness=round(mean_b, 2),
        foreground_ratio=round(fg_ratio, 4),
        saturation_mean=round(sat_mean, 4),
        issues=issues,
    )


class ClassicalFeatureExtractor:
    """Extractor de características clásicas multimodales (HOG + Color Espacial + Firma Azimutal/Inercia).

    Produce un vector compacto, interpretativo y altamente discriminativo tanto para
    identificación de los 100 SKUs como para regresión circular de orientación.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (64, 64),
        hog_orientations: int = 9,
        hog_pixels_per_cell: Tuple[int, int] = (8, 8),
        hog_cells_per_block: Tuple[int, int] = (2, 2),
        hsv_bins: Tuple[int, int, int] = (16, 8, 8),
        azimuth_sectors: int = 36,
    ) -> None:
        self.image_size = image_size
        self.hog_orientations = hog_orientations
        self.hog_pixels_per_cell = hog_pixels_per_cell
        self.hog_cells_per_block = hog_cells_per_block
        self.hsv_bins = hsv_bins
        self.azimuth_sectors = azimuth_sectors

        # Precomputar rejilla polar centrada para descriptores azimutales y momentos
        h, w = self.image_size
        yy, xx = np.mgrid[0:h, 0:w]
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        self._dy = (yy - cy) / (h / 2.0)
        self._dx = (xx - cx) / (w / 2.0)
        self._radius = np.hypot(self._dx, self._dy)
        self._angle_bins = np.floor(
            ((np.arctan2(self._dy, self._dx) + np.pi) / (2.0 * np.pi)) * azimuth_sectors
        ).astype(int)
        self._angle_bins = np.clip(self._angle_bins, 0, azimuth_sectors - 1)

    def extract_single(self, img_rgb: np.ndarray) -> np.ndarray:
        """Extrae el vector completo de características clásicas para una imagen RGB."""
        from PIL import Image

        if img_rgb.shape[:2] != self.image_size:
            pil_img = Image.fromarray(img_rgb).resize(self.image_size, Image.Resampling.BILINEAR)
            img = np.asarray(pil_img, dtype=np.uint8)
        else:
            img = img_rgb

        img_float = img.astype(np.float32) / 255.0
        gray = rgb2gray(img)
        hsv = rgb2hsv(img)

        # 1. Descriptor HOG (Geometría y bordes orientados)
        hog_vec = hog(
            gray,
            orientations=self.hog_orientations,
            pixels_per_cell=self.hog_pixels_per_cell,
            cells_per_block=self.hog_cells_per_block,
            block_norm="L2-Hys",
            feature_vector=True,
        ).astype(np.float32)

        # 2. Histogramas de Color HSV Globales (sobre máscara de objeto para ignorar fondo negro)
        fg_mask = gray > 0.06
        if np.sum(fg_mask) < 16:
            fg_mask = np.ones_like(gray, dtype=bool)

        h_hist, _ = np.histogram(hsv[:, :, 0][fg_mask], bins=self.hsv_bins[0], range=(0.0, 1.0), density=False)
        s_hist, _ = np.histogram(hsv[:, :, 1][fg_mask], bins=self.hsv_bins[1], range=(0.0, 1.0), density=False)
        v_hist, _ = np.histogram(hsv[:, :, 2][fg_mask], bins=self.hsv_bins[2], range=(0.0, 1.0), density=False)
        n_fg = max(float(np.sum(fg_mask)), 1.0)
        global_hsv = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32) / n_fg

        # 3. Pirámide Espacial 2x2 de Color (RGB + HSV estadísticos por cuadrante) + Rejilla 4x4 compacta
        # Captura la asimetría espacial del envase cuando gira en la plataforma
        h, w = self.image_size
        grid_stats = []
        for gy in range(4):
            y0, y1 = (gy * h) // 4, ((gy + 1) * h) // 4
            for gx in range(4):
                x0, x1 = (gx * w) // 4, ((gx + 1) * w) // 4
                patch_rgb = img_float[y0:y1, x0:x1]
                patch_hsv = hsv[y0:y1, x0:x1]
                grid_stats.extend(np.mean(patch_rgb, axis=(0, 1)))
                grid_stats.extend(np.std(patch_rgb, axis=(0, 1)))
                grid_stats.extend(np.mean(patch_hsv[:, :, :2], axis=(0, 1)))
        spatial_color_vec = np.asarray(grid_stats, dtype=np.float32)

        # 4. Firma Azimutal Direccional (36 sectores angulares x 3 canales RGB + gradiente radial)
        # Esencial para estimación de orientación continua en [0°, 360°)
        azimuth_rgb = np.zeros((self.azimuth_sectors, 3), dtype=np.float32)
        azimuth_rad = np.zeros(self.azimuth_sectors, dtype=np.float32)
        for s in range(self.azimuth_sectors):
            sec_mask = (self._angle_bins == s) & fg_mask
            if np.any(sec_mask):
                azimuth_rgb[s] = np.mean(img_float[sec_mask], axis=0)
                azimuth_rad[s] = np.mean(self._radius[sec_mask] * gray[sec_mask])

        # 5. Momentos Geométricos y de Orientación de Inercia (2do orden)
        weights = gray * fg_mask
        w_sum = max(float(np.sum(weights)), 1e-6)
        cx_w = float(np.sum(self._dx * weights) / w_sum)
        cy_w = float(np.sum(self._dy * weights) / w_sum)
        mu20 = float(np.sum(((self._dx - cx_w) ** 2) * weights) / w_sum)
        mu02 = float(np.sum(((self._dy - cy_w) ** 2) * weights) / w_sum)
        mu11 = float(np.sum((self._dx - cx_w) * (self._dy - cy_w) * weights) / w_sum)
        # Centroide cromático R, G, B (vector dipolo de color que rota con el ángulo θ)
        dipole_feats = []
        for c in range(3):
            cw = img_float[:, :, c] * fg_mask
            c_sum = max(float(np.sum(cw)), 1e-6)
            dipole_feats.append(float(np.sum(self._dx * cw) / c_sum))
            dipole_feats.append(float(np.sum(self._dy * cw) / c_sum))

        moments_vec = np.asarray(
            [cx_w, cy_w, mu20, mu02, mu11, float(np.mean(fg_mask))] + dipole_feats,
            dtype=np.float32,
        )

        return np.concatenate(
            [
                hog_vec,
                global_hsv * 2.5,
                spatial_color_vec,
                azimuth_rgb.ravel(),
                azimuth_rad,
                moments_vec * 2.0,
            ]
        )

    def extract_batch(self, image_paths: List[str]) -> np.ndarray:
        """Extrae matriz de características `(N, D)` a partir de una lista de rutas de imagen."""
        from src.ml.dataset import load_image_rgb

        feats = [self.extract_single(load_image_rgb(p, target_size=self.image_size)) for p in image_paths]
        return np.vstack(feats)


def compute_rotational_symmetry_index(
    catalog_df,
    feature_matrix: np.ndarray,
) -> Dict[int, Dict[str, float]]:
    """Analiza para cada objeto de COIL-100 su índice de simetría rotacional a 180° y 90°.

    Por qué es vital en una línea de empaque robótica:
    Un cilindro o envase simétrico a 180° tiene vistas casi indistinguibles en θ y θ + 180°.
    Identificar qué SKUs presentan simetría bimodal explica los errores aparentes cercanos a 180°
    y permite ajustar la lógica de agarre del efector final.
    """
    symmetry_info: Dict[int, Dict[str, float]] = {}
    for obj_id, group in catalog_df.groupby("object_id"):
        g_sorted = group.sort_values("angle_deg")
        idx = g_sorted.index.to_numpy()
        feats = feature_matrix[idx]  # (72, D) ordenados de 0° a 355° cada 5°
        if len(feats) != 72:
            continue

        # Normalizar filas para similitud coseno
        norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
        feats_norm = feats / norms

        # Similitud entre vistas vecinas (Δ = 5° -> shift 1)
        sim_5deg = float(np.mean(np.sum(feats_norm * np.roll(feats_norm, 1, axis=0), axis=1)))
        # Similitud a 180° (shift 36 posiciones de 5°)
        sim_180deg = float(np.mean(np.sum(feats_norm * np.roll(feats_norm, 36, axis=0), axis=1)))
        # Similitud a 90° (shift 18 posiciones de 5°)
        sim_90deg = float(np.mean(np.sum(feats_norm * np.roll(feats_norm, 18, axis=0), axis=1)))

        symmetry_ratio_180 = sim_180deg / max(sim_5deg, 1e-6)
        symmetry_info[int(obj_id)] = {
            "sim_5deg": round(sim_5deg, 4),
            "sim_90deg": round(sim_90deg, 4),
            "sim_180deg": round(sim_180deg, 4),
            "symmetry_ratio_180": round(symmetry_ratio_180, 4),
            "is_highly_symmetric": bool(symmetry_ratio_180 > 0.965),
        }
    return symmetry_info
