"""Layer 3: Local ROI refinement, CLAHE(V), Adaptive Split-Range HSV, morphology & geometry."""

from __future__ import annotations

import copy
import math
from typing import Tuple

import cv2
import numpy as np

from .config import ROI_HALF, AdaptiveHSVProfile, PipelineConfig
from .types import DetectionCandidate, RefinementResult


def _find_external_contours(mask: np.ndarray) -> list[np.ndarray]:
    """OpenCV 3/4/5-safe external contour extraction."""
    res = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = res[0] if len(res) == 2 else res[1]
    return list(contours)


def preprocess_roi_hsv(
    roi_bgr: np.ndarray,
    gaussian_ksize: int = 5,
    clahe_clip_limit: float = 2.0,
    clahe_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply 5x5 Gaussian blur, BGR->HSV conversion, and CLAHE strictly on the V channel."""
    k = gaussian_ksize if gaussian_ksize % 2 == 1 else gaussian_ksize + 1
    blurred = cv2.GaussianBlur(roi_bgr, (k, k), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip_limit), tileGridSize=clahe_grid_size)
    v_eq = clahe.apply(v_ch)
    return cv2.merge([h_ch, s_ch, v_eq])


def build_split_red_mask(
    hsv_eq: np.ndarray,
    profile: AdaptiveHSVProfile,
    median_ksize: int = 5,
    morph_ksize: int = 7,
    open_iter: int = 1,
    close_iter: int = 3,
) -> np.ndarray:
    """Threshold split-range red hues and apply MedianBlur + Ellipse Open/Close morphology."""
    low_lo = np.array([profile.low_hue_range[0], profile.s_min, profile.v_min], dtype=np.uint8)
    low_hi = np.array([profile.low_hue_range[1], profile.s_max, profile.v_max], dtype=np.uint8)
    high_lo = np.array([profile.high_hue_range[0], profile.s_min, profile.v_min], dtype=np.uint8)
    high_hi = np.array([profile.high_hue_range[1], profile.s_max, profile.v_max], dtype=np.uint8)

    m1 = cv2.inRange(hsv_eq, low_lo, low_hi)
    m2 = cv2.inRange(hsv_eq, high_lo, high_hi)
    mask = cv2.bitwise_or(m1, m2)

    mk = median_ksize if median_ksize % 2 == 1 else median_ksize + 1
    if mk >= 3:
        mask = cv2.medianBlur(mask, mk)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
    if open_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    if close_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    return mask


def compute_geometry_metrics(mask: np.ndarray) -> tuple[float, float, float, float, float, bool]:
    """Compute (red_ratio, circularity, solidity, extent, aspect_ratio, has_contour)."""
    total_pixels = float(mask.shape[0] * mask.shape[1])
    if total_pixels <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, False

    red_ratio = float(np.count_nonzero(mask)) / total_pixels
    contours = _find_external_contours(mask)
    if not contours:
        return red_ratio, 0.0, 0.0, 0.0, 0.0, False

    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    if area <= 0.0:
        return red_ratio, 0.0, 0.0, 0.0, 0.0, False

    perim = float(cv2.arcLength(cnt, True))
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    _, _, bw, bh = cv2.boundingRect(cnt)
    rect_area = float(bw * bh) if bw > 0 and bh > 0 else 1.0

    circularity = (4.0 * math.pi * area / (perim * perim)) if perim > 0.0 else 0.0
    solidity = (area / hull_area) if hull_area > 0.0 else 0.0
    extent = area / rect_area
    axis_ar = (float(bw) / float(bh)) if bh > 0 else 0.0
    (_, _), (mw, mh), _ = cv2.minAreaRect(cnt)
    ow = float(mw) + 1.0
    oh = float(mh) + 1.0
    oriented_ar = (max(ow, oh) / min(ow, oh)) if (ow > 0.0 and oh > 0.0) else axis_ar
    axis_elongation = max(axis_ar, 1.0 / axis_ar) if axis_ar > 0.0 else 0.0
    aspect_ratio = oriented_ar if oriented_ar > axis_elongation else axis_ar
    return red_ratio, circularity, solidity, extent, aspect_ratio, True


def compute_hsv_conf(
    red_ratio: float,
    circularity: float,
    solidity: float,
    extent: float,
    aspect_ratio: float,
) -> float:
    """Compute bounded [0.0, 1.0] HSV confidence combining red area ratio and contour shape."""
    red_score = min(1.0, max(0.0, red_ratio) / 0.45)
    circ_s = min(1.0, max(0.0, circularity) / 0.75)
    sol_s = min(1.0, max(0.0, (solidity - 0.50) / 0.45))
    ext_s = min(1.0, max(0.0, (extent - 0.35) / 0.40))
    ar_s = max(0.0, 1.0 - abs(aspect_ratio - 1.0))
    shape_quality = 0.30 * circ_s + 0.30 * sol_s + 0.20 * ext_s + 0.20 * ar_s
    return float(np.clip(0.45 * red_score + 0.55 * shape_quality, 0.0, 1.0))


def _check_admissibility(
    red_ratio: float,
    circularity: float,
    solidity: float,
    extent: float,
    aspect_ratio: float,
    has_contour: bool,
    config: PipelineConfig,
) -> tuple[bool, str | None]:
    if red_ratio <= config.min_red_ratio:
        return False, f"red_ratio={red_ratio:.3f}<={config.min_red_ratio:.2f}"
    if not has_contour:
        return False, "no_contour"
    if circularity <= config.min_circularity:
        return False, f"circularity={circularity:.3f}<={config.min_circularity:.2f}"
    if solidity <= config.min_solidity:
        return False, f"solidity={solidity:.3f}<={config.min_solidity:.2f}"
    if extent <= config.min_extent:
        return False, f"extent={extent:.3f}<={config.min_extent:.2f}"
    ar_lo, ar_hi = config.aspect_ratio_range
    if not (ar_lo <= aspect_ratio <= ar_hi):
        return False, f"aspect_ratio={aspect_ratio:.2f} not in [{ar_lo}, {ar_hi}]"
    return True, None


def is_annular_or_ribbon_void(
    mask: np.ndarray,
    cnt: np.ndarray | None = None,
) -> bool:
    """Detect hollow annular arcs and uniform-stroke concave ribbons with deep interior voids."""
    if cnt is None:
        contours = _find_external_contours(mask)
        if not contours:
            return False
        cnt = max(contours, key=cv2.contourArea)

    area = float(cv2.contourArea(cnt))
    if area <= 0.0:
        return False
    hull_area = float(cv2.contourArea(cv2.convexHull(cnt)))
    solidity = (area / hull_area) if hull_area > 0.0 else 0.0
    if solidity >= 0.72:
        return False

    bx, by, bw, bh = cv2.boundingRect(cnt)
    if bw <= 0 or bh <= 0:
        return False
    sub = np.zeros((bh, bw), dtype=np.uint8)
    cv2.drawContours(sub, [cnt - np.array([[bx, by]])], -1, 255, -1)
    dist = cv2.distanceTransform(sub, cv2.DIST_L2, 5)
    max_dt = float(dist.max())
    (_, _), (mw, mh), _ = cv2.minAreaRect(cnt)
    minor_span = max(min(float(mw) + 1.0, float(mh) + 1.0), 1.0)
    thick_ratio = (2.0 * max_dt) / minor_span

    max_defect = 0.0
    hull_idx = cv2.convexHull(cnt, returnPoints=False)
    if hull_idx is not None and len(hull_idx) > 3 and len(cnt) > 3:
        try:
            defects = cv2.convexityDefects(cnt, hull_idx)
            if defects is not None and defects.size > 0:
                max_defect = float(np.max(defects.reshape(-1, 4)[:, 3]) / 256.0)
        except cv2.error:
            max_defect = 0.0
    defect_ratio = max_defect / minor_span
    return (thick_ratio < 0.48) or (defect_ratio > 0.46 and thick_ratio < 0.55)


def extract_blob_components(
    sub_fg: np.ndarray,
    min_area: int = 120,
    pinch_ratio: float = 0.82,
) -> list[np.ndarray]:
    """Split touching rounded apple blobs via non-transitive DT saddles & defect pinches.

    Protects uniform-stroke concave ribbons from shattering and prevents interstitial
    overlap centroids in multi-apple clusters from transitively bridging distinct lobes.
    """
    ch, cw = sub_fg.shape[:2]
    if ch <= 0 or cw <= 0:
        return []
    contours = _find_external_contours(sub_fg)
    if not contours:
        return []
    cnt = max(contours, key=cv2.contourArea)
    if is_annular_or_ribbon_void(sub_fg, cnt):
        return [sub_fg]

    dist = cv2.distanceTransform(sub_fg, cv2.DIST_L2, 5)
    dist_s = cv2.GaussianBlur(dist, (3, 3), 0)
    dil = cv2.dilate(dist_s, np.ones((11, 11), np.uint8))
    peaks = (dist_s >= dil - 1e-5) & (dist_s >= 4.0) & (sub_fg > 0)
    n_pk, _, _, pk_cents = cv2.connectedComponentsWithStats(peaks.astype(np.uint8))

    if n_pk > 2:
        raw_cents = pk_cents[1:]
        raw_coords = [
            (
                int(np.clip(round(float(c[0])), 0, cw - 1)),
                int(np.clip(round(float(c[1])), 0, ch - 1)),
            )
            for c in raw_cents
        ]
        raw_heights = [float(dist_s[py, px]) for px, py in raw_coords]
        max_pk_h = max(raw_heights) if raw_heights else 0.0
        valid_pk = [i for i, h_val in enumerate(raw_heights) if h_val >= 0.52 * max_pk_h]
        cents = raw_cents[valid_pk] if len(valid_pk) >= 2 else np.empty((0, 2), dtype=np.float64)
        k_count = len(cents)
    else:
        cents = np.empty((0, 2), dtype=np.float64)
        k_count = 0

    dt_parts: list[np.ndarray] = []
    if k_count >= 2:
        pk_coords = [
            (
                int(np.clip(round(float(c[0])), 0, cw - 1)),
                int(np.clip(round(float(c[1])), 0, ch - 1)),
            )
            for c in cents
        ]
        heights = [float(dist_s[py, px]) for px, py in pk_coords]

        def _line_saddle(i: int, j: int) -> float:
            xi, yi = float(cents[i][0]), float(cents[i][1])
            xj, yj = float(cents[j][0]), float(cents[j][1])
            n_samp = max(16, int(math.ceil(math.hypot(xj - xi, yj - yi))) + 1)
            xs = np.clip(np.round(np.linspace(xi, xj, n_samp)).astype(int), 0, cw - 1)
            ys = np.clip(np.round(np.linspace(yi, yj, n_samp)).astype(int), 0, ch - 1)
            if not np.all(sub_fg[ys, xs] > 0):
                return 0.0
            return float(np.min(dist_s[ys, xs]))

        # Suppress interstitial overlap bridge peaks (e.g. centroid peak inside 3-apple clusters)
        keep_mask = [True] * k_count
        for b in range(k_count):
            unpinched: list[int] = []
            for i in range(k_count):
                if i == b:
                    continue
                d_bi = math.hypot(cents[b][0] - cents[i][0], cents[b][1] - cents[i][1])
                if (
                    d_bi <= 1.30 * max(heights[b], heights[i])
                    and _line_saddle(b, i) >= 0.85 * min(heights[b], heights[i])
                ):
                    unpinched.append(i)
            if len(unpinched) >= 2:
                is_bridge = False
                for ia_idx in range(len(unpinched)):
                    for ib_idx in range(ia_idx + 1, len(unpinched)):
                        ia, ib = unpinched[ia_idx], unpinched[ib_idx]
                        d_ab = math.hypot(
                            cents[ia][0] - cents[ib][0], cents[ia][1] - cents[ib][1]
                        )
                        if (
                            d_ab >= 1.25 * min(heights[ia], heights[ib])
                            and _line_saddle(ia, ib) < 0.86 * min(heights[ia], heights[ib])
                        ):
                            d_ba = math.hypot(
                                cents[b][0] - cents[ia][0], cents[b][1] - cents[ia][1]
                            )
                            d_bb = math.hypot(
                                cents[b][0] - cents[ib][0], cents[b][1] - cents[ib][1]
                            )
                            if d_ab > max(d_ba, d_bb):
                                is_bridge = True
                                break
                    if is_bridge:
                        break
                if is_bridge:
                    keep_mask[b] = False

        kept_indices = [i for i in range(k_count) if keep_mask[i]]
        if len(kept_indices) >= 2:
            m_k = len(kept_indices)
            parent = list(range(m_k))

            def _find(u: int) -> int:
                while parent[u] != u:
                    parent[u] = parent[parent[u]]
                    u = parent[u]
                return u

            s_ratio = [[1.0] * m_k for _ in range(m_k)]
            d_norm = [[0.0] * m_k for _ in range(m_k)]
            pairs: list[tuple[float, int, int]] = []
            for a in range(m_k):
                for b in range(a + 1, m_k):
                    ia, ib = kept_indices[a], kept_indices[b]
                    dij = math.hypot(
                        cents[ia][0] - cents[ib][0], cents[ia][1] - cents[ib][1]
                    )
                    min_h = max(min(heights[ia], heights[ib]), 1e-6)
                    sr = _line_saddle(ia, ib) / min_h
                    s_ratio[a][b] = s_ratio[b][a] = sr
                    d_norm[a][b] = d_norm[b][a] = dij / min_h
                    pairs.append((sr, a, b))

            pairs.sort(reverse=True)
            for sr, a, b in pairs:
                if sr < pinch_ratio and d_norm[a][b] >= 1.15:
                    continue
                ra, rb = _find(a), _find(b)
                if ra == rb:
                    continue
                comp_a = [x for x in range(m_k) if _find(x) == ra]
                comp_b = [x for x in range(m_k) if _find(x) == rb]
                if not any(
                    s_ratio[x][y] < pinch_ratio and d_norm[x][y] >= 1.15
                    for x in comp_a
                    for y in comp_b
                ):
                    parent[rb] = ra

            roots = [_find(u) for u in range(m_k)]
            unique_roots = sorted(set(roots))
            if len(unique_roots) >= 2:
                grp_centers = [
                    np.mean(
                        [cents[kept_indices[u]] for u in range(m_k) if roots[u] == grp],
                        axis=0,
                    )
                    for grp in unique_roots
                ]
                grp_arr = np.asarray(grp_centers, dtype=np.float64)
                ys, xs = np.where(sub_fg > 0)
                pts = np.stack([xs, ys], axis=1)
                d2 = ((pts[:, None, :] - grp_arr[None, :, :]) ** 2).sum(axis=2)
                nearest = np.argmin(d2, axis=1)
                parts: list[np.ndarray] = []
                for gi in range(len(unique_roots)):
                    sel = nearest == gi
                    if int(np.count_nonzero(sel)) < min_area:
                        continue
                    part = np.zeros_like(sub_fg)
                    part[ys[sel], xs[sel]] = 255
                    parts.append(part)
                if len(parts) >= 2:
                    dt_parts = parts

    # Convexity-defect pinch splitting when heavy closing merges DT peaks or misses a lobe
    max_dt = float(dist.max())
    if max_dt >= 8.0:
        (_, _), (mw, mh), _ = cv2.minAreaRect(cnt)
        major = max(float(mw) + 1.0, float(mh) + 1.0)
        area = float(cv2.contourArea(cnt))
        hull_area = float(cv2.contourArea(cv2.convexHull(cnt)))
        solidity = (area / hull_area) if hull_area > 0.0 else 0.0
        if (major / (2.0 * max_dt)) >= 1.35 and solidity >= 0.78:
            hull_idx = cv2.convexHull(cnt, returnPoints=False)
            if hull_idx is not None and len(hull_idx) > 3 and len(cnt) > 3:
                try:
                    defects = cv2.convexityDefects(cnt, hull_idx)
                except cv2.error:
                    defects = None
                if defects is not None and defects.size > 0:
                    d_arr = defects.reshape(-1, 4)
                    min_depth = max(5.5, 0.16 * max_dt)
                    cand_defs = sorted(
                        [
                            (float(d) / 256.0, (int(cnt[f][0][0]), int(cnt[f][0][1])))
                            for _, _, f, d in d_arr
                            if (float(d) / 256.0) >= min_depth
                        ],
                        reverse=True,
                    )
                    kept_defs: list[tuple[int, int]] = []
                    for _, pt in cand_defs:
                        if all(
                            math.hypot(pt[0] - kp[0], pt[1] - kp[1]) >= 0.7 * max_dt
                            for kp in kept_defs
                        ):
                            kept_defs.append(pt)

                    if (
                        len(dt_parts) < 2
                        and len(kept_defs) == 2
                        and math.hypot(
                            kept_defs[0][0] - kept_defs[1][0],
                            kept_defs[0][1] - kept_defs[1][1],
                        )
                        <= 1.85 * max_dt
                    ):
                        cut = sub_fg.copy()
                        cv2.line(cut, kept_defs[0], kept_defs[1], 0, 2)
                        n_c, lbl_c, st_c, _ = cv2.connectedComponentsWithStats(cut)
                        parts = [
                            (lbl_c == cid).astype(np.uint8) * 255
                            for cid in range(1, n_c)
                            if int(st_c[cid, cv2.CC_STAT_AREA]) >= min_area
                        ]
                        if len(parts) >= 2:
                            return parts
                    elif len(kept_defs) >= 3 and len(kept_defs) > len(dt_parts):
                        my, mx = np.unravel_index(int(np.argmax(dist)), dist.shape)
                        center = (int(mx), int(my))
                        cut = sub_fg.copy()
                        for pt in kept_defs:
                            cv2.line(cut, center, pt, 0, 2)
                        n_c, lbl_c, st_c, _ = cv2.connectedComponentsWithStats(cut)
                        parts = [
                            (lbl_c == cid).astype(np.uint8) * 255
                            for cid in range(1, n_c)
                            if int(st_c[cid, cv2.CC_STAT_AREA]) >= min_area
                        ]
                        if len(parts) >= len(kept_defs):
                            return parts

    if len(dt_parts) >= 2:
        return dt_parts
    return [sub_fg]


class AdaptiveHSVCalibrator:
    """Interactive/programmatic ROI HSV sampler (ROI_HALF=100, 20th/80th percentiles)."""

    def __init__(self, initial_profile: AdaptiveHSVProfile | None = None) -> None:
        self._baseline = copy.deepcopy(initial_profile) if initial_profile else AdaptiveHSVProfile()
        self.profile = copy.deepcopy(self._baseline)
        self._accumulated_hsv: list[np.ndarray] = []

    def reset(self) -> AdaptiveHSVProfile:
        """Reset accumulated samples and restore baseline split-range HSV thresholds."""
        self._accumulated_hsv.clear()
        self.profile = copy.deepcopy(self._baseline)
        return self.profile

    def calibrate_from_point(
        self,
        frame_bgr: np.ndarray,
        cx: int,
        cy: int,
        roi_half: int = ROI_HALF,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: tuple[int, int] = (8, 8),
    ) -> Tuple[AdaptiveHSVProfile, tuple[int, int, int, int] | None, list[str]]:
        """Sample a [cx-roi_half:cx+roi_half, cy-roi_half:cy+roi_half] patch and update HSV bounds."""
        h, w = frame_bgr.shape[:2]
        icx = int(cx)
        icy = int(cy)
        if not (0 <= icx < w and 0 <= icy < h):
            return self.profile, None, ["[ROI CLICK] ignored out-of-bounds"]
        x1 = max(0, icx - int(roi_half))
        y1 = max(0, icy - int(roi_half))
        x2 = min(w, icx + int(roi_half))
        y2 = min(h, icy + int(roi_half))
        return self.calibrate_from_rect(
            frame_bgr,
            (x1, y1, x2, y2),
            clahe_clip_limit=clahe_clip_limit,
            clahe_grid_size=clahe_grid_size,
        )

    def calibrate_from_rect(
        self,
        frame_bgr: np.ndarray,
        rect: tuple[int, int, int, int],
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: tuple[int, int] = (8, 8),
    ) -> Tuple[AdaptiveHSVProfile, tuple[int, int, int, int] | None, list[str]]:
        """Sample HSV pixels inside rect, adapt S/V via 20th/80th percentiles and H split ranges."""
        h, w = frame_bgr.shape[:2]
        raw_x1, raw_y1, raw_x2, raw_y2 = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
        if raw_x2 <= raw_x1 or raw_y2 <= raw_y1:
            return self.profile, None, ["[ROI CLICK] ignored out-of-bounds"]
        x1 = max(0, min(w, raw_x1))
        y1 = max(0, min(h, raw_y1))
        x2 = max(0, min(w, raw_x2))
        y2 = max(0, min(h, raw_y2))
        if x2 <= x1 or y2 <= y1:
            return self.profile, None, ["[ROI CLICK] ignored out-of-bounds"]

        patch = frame_bgr[y1:y2, x1:x2]
        hsv_eq = preprocess_roi_hsv(
            patch,
            gaussian_ksize=5,
            clahe_clip_limit=clahe_clip_limit,
            clahe_grid_size=clahe_grid_size,
        )
        hh, ss, vv = cv2.split(hsv_eq)
        # Filter red-leaning pixels within the clicked ROI (excluding neutral background/glare)
        red_sel = ((hh <= 18) | (hh >= 155)) & (ss >= 50) & (vv >= 30)
        if int(np.count_nonzero(red_sel)) >= 25:
            samples = hsv_eq[red_sel]
        else:
            samples = hsv_eq.reshape(-1, 3)

        added_count = int(samples.shape[0])
        self._accumulated_hsv.append(samples)
        all_samples = np.vstack(self._accumulated_hsv)

        h_vals = all_samples[:, 0]
        s_vals = all_samples[:, 1]
        v_vals = all_samples[:, 2]

        s_p20 = int(round(float(np.percentile(s_vals, 20))))
        s_p80 = int(round(float(np.percentile(s_vals, 80))))
        v_p20 = int(round(float(np.percentile(v_vals, 20))))
        v_p80 = int(round(float(np.percentile(v_vals, 80))))

        s_min = int(np.clip(int(round(s_p20 * 0.65)), 40, 145))
        v_min = int(np.clip(int(round(v_p20 * 0.75)), 30, 140))

        low_h = h_vals[h_vals <= 18]
        if low_h.size >= 10:
            l_lo = int(np.clip(np.floor(np.percentile(low_h, 5)), 0, 12))
            l_hi = int(np.clip(np.ceil(np.percentile(low_h, 95)), max(l_lo + 3, 8), 15))
            low_range = (l_lo, l_hi)
        else:
            low_range = self._baseline.low_hue_range

        high_h = h_vals[h_vals >= 155]
        if high_h.size >= 10:
            h_lo = int(np.clip(np.floor(np.percentile(high_h, 5)), 160, 176))
            h_hi = int(np.clip(np.ceil(np.percentile(high_h, 95)), max(h_lo + 2, 170), 179))
            high_range = (h_lo, h_hi)
        else:
            high_range = self._baseline.high_hue_range

        self.profile = AdaptiveHSVProfile(
            low_hue_range=low_range,
            high_hue_range=high_range,
            s_min=s_min,
            s_max=255,
            v_min=v_min,
            v_max=255,
            s_p20=s_p20,
            s_p80=s_p80,
            v_p20=v_p20,
            v_p80=v_p80,
            sampled_pixels=int(all_samples.shape[0]),
        )
        logs = [
            f"[ROI CLICK] added {added_count} hsv pixels",
            self.profile.format_calc_log(),
        ]
        return self.profile, (x1, y1, x2, y2), logs


def _empty_refinement(candidate: DetectionCandidate, reason: str) -> RefinementResult:
    return RefinementResult(
        candidate=candidate,
        passed=False,
        reject_reason=reason,
        red_ratio=0.0,
        circularity=0.0,
        solidity=0.0,
        extent=0.0,
        aspect_ratio=0.0,
        hsv_conf=0.0,
        roi_mask=np.zeros((0, 0), dtype=np.uint8),
    )


def refine_candidate(
    frame_bgr: np.ndarray,
    candidate: DetectionCandidate,
    profile: AdaptiveHSVProfile | None = None,
    config: PipelineConfig | None = None,
) -> RefinementResult:
    """Run Layer 3 ROI preprocessing, split-range HSV, morphology, and geometric gates."""
    cfg = config or PipelineConfig()
    prof = profile or cfg.initial_hsv
    h, w = frame_bgr.shape[:2]

    x1, y1, x2, y2 = candidate.bbox
    if not (
        math.isfinite(float(x1))
        and math.isfinite(float(y1))
        and math.isfinite(float(x2))
        and math.isfinite(float(y2))
    ):
        return _empty_refinement(candidate, "invalid_bbox")

    ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return _empty_refinement(candidate, "invalid_roi")

    cx1 = max(0, min(w, ix1))
    cy1 = max(0, min(h, iy1))
    cx2 = max(0, min(w, ix2))
    cy2 = max(0, min(h, iy2))
    if cx2 <= cx1 or cy2 <= cy1:
        return _empty_refinement(candidate, "invalid_roi")

    roi = frame_bgr[cy1:cy2, cx1:cx2]
    hsv_eq = preprocess_roi_hsv(
        roi,
        gaussian_ksize=cfg.gaussian_ksize,
        clahe_clip_limit=cfg.clahe_clip_limit,
        clahe_grid_size=cfg.clahe_grid_size,
    )
    mask = build_split_red_mask(
        hsv_eq,
        profile=prof,
        median_ksize=cfg.median_ksize,
        morph_ksize=cfg.morph_ksize,
        open_iter=cfg.open_iter,
        close_iter=cfg.close_iter,
    )
    red_ratio, circularity, solidity, extent, aspect_ratio, has_contour = compute_geometry_metrics(mask)
    passed, reject_reason = _check_admissibility(
        red_ratio, circularity, solidity, extent, aspect_ratio, has_contour, cfg
    )
    if passed and is_annular_or_ribbon_void(mask):
        passed = False
        reject_reason = "concave_void_ribbon"
    hsv_conf = compute_hsv_conf(red_ratio, circularity, solidity, extent, aspect_ratio) if passed else 0.0

    return RefinementResult(
        candidate=candidate,
        passed=passed,
        reject_reason=reject_reason,
        red_ratio=red_ratio,
        circularity=circularity,
        solidity=solidity,
        extent=extent,
        aspect_ratio=aspect_ratio,
        hsv_conf=hsv_conf,
        roi_mask=mask,
    )


def detect_in_rect(
    frame_bgr: np.ndarray,
    rect: tuple[int, int, int, int] | None = None,
    profile: AdaptiveHSVProfile | None = None,
    config: PipelineConfig | None = None,
) -> tuple[list[RefinementResult], np.ndarray]:
    """Detect and filter red apple contours directly inside rect (or full frame if rect is None)."""
    cfg = config or PipelineConfig()
    prof = profile or cfg.initial_hsv
    h, w = frame_bgr.shape[:2]

    if rect is None:
        rx1, ry1, rx2, ry2 = 0, 0, w, h
    else:
        raw_x1, raw_y1, raw_x2, raw_y2 = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
        if raw_x2 <= raw_x1 or raw_y2 <= raw_y1:
            return [], np.zeros((h, w), dtype=np.uint8)
        rx1 = max(0, min(w, raw_x1))
        ry1 = max(0, min(h, raw_y1))
        rx2 = max(0, min(w, raw_x2))
        ry2 = max(0, min(h, raw_y2))
        if rx2 <= rx1 or ry2 <= ry1:
            return [], np.zeros((h, w), dtype=np.uint8)

    roi = frame_bgr[ry1:ry2, rx1:rx2]
    hsv_eq = preprocess_roi_hsv(
        roi,
        gaussian_ksize=cfg.gaussian_ksize,
        clahe_clip_limit=cfg.clahe_clip_limit,
        clahe_grid_size=cfg.clahe_grid_size,
    )
    mask = build_split_red_mask(
        hsv_eq,
        profile=prof,
        median_ksize=cfg.median_ksize,
        morph_ksize=cfg.morph_ksize,
        open_iter=cfg.open_iter,
        close_iter=cfg.close_iter,
    )

    contours = _find_external_contours(mask)
    results: list[RefinementResult] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < cfg.min_contour_area:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bw <= 0 or bh <= 0:
            continue

        cnt_mask = np.zeros((bh, bw), dtype=np.uint8)
        cv2.drawContours(cnt_mask, [cnt - np.array([[bx, by]])], -1, 255, -1)
        sub_fg = cv2.bitwise_and(mask[by : by + bh, bx : bx + bw], cnt_mask)
        parts = extract_blob_components(sub_fg, min_area=cfg.min_contour_area)

        for part in parts:
            pys, pxs = np.where(part > 0)
            if pys.size == 0:
                continue
            px1 = int(pxs.min())
            py1 = int(pys.min())
            px2 = int(pxs.max()) + 1
            py2 = int(pys.max()) + 1
            sub_mask = part[py1:py2, px1:px2]

            red_ratio, circularity, solidity, extent, aspect_ratio, has_contour = compute_geometry_metrics(sub_mask)
            passed, _ = _check_admissibility(
                red_ratio, circularity, solidity, extent, aspect_ratio, has_contour, cfg
            )
            if not passed or is_annular_or_ribbon_void(sub_mask):
                continue

            hsv_conf = compute_hsv_conf(red_ratio, circularity, solidity, extent, aspect_ratio)
            cand = DetectionCandidate(
                bbox=(rx1 + bx + px1, ry1 + by + py1, rx1 + bx + px2, ry1 + by + py2),
                yconf=hsv_conf,
                label="red_apple",
                backend="hsv",
            )
            results.append(
                RefinementResult(
                    candidate=cand,
                    passed=True,
                    reject_reason=None,
                    red_ratio=red_ratio,
                    circularity=circularity,
                    solidity=solidity,
                    extent=extent,
                    aspect_ratio=aspect_ratio,
                    hsv_conf=hsv_conf,
                    roi_mask=sub_mask,
                )
            )

    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[ry1:ry2, rx1:rx2] = mask
    return results, full_mask


def segment_full_frame_hsv(
    frame_bgr: np.ndarray,
    profile: AdaptiveHSVProfile | None = None,
    config: PipelineConfig | None = None,
    active_roi: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Compute the binary HSV mask for the Mask (HSV/ROI) visualization panel."""
    cfg = config or PipelineConfig()
    prof = profile or cfg.initial_hsv
    h, w = frame_bgr.shape[:2]

    if active_roi is None:
        rx1, ry1, rx2, ry2 = 0, 0, w, h
    else:
        raw_x1, raw_y1, raw_x2, raw_y2 = (
            int(active_roi[0]),
            int(active_roi[1]),
            int(active_roi[2]),
            int(active_roi[3]),
        )
        if raw_x2 <= raw_x1 or raw_y2 <= raw_y1:
            return np.zeros((h, w), dtype=np.uint8)
        rx1 = max(0, min(w, raw_x1))
        ry1 = max(0, min(h, raw_y1))
        rx2 = max(0, min(w, raw_x2))
        ry2 = max(0, min(h, raw_y2))
        if rx2 <= rx1 or ry2 <= ry1:
            return np.zeros((h, w), dtype=np.uint8)

    roi = frame_bgr[ry1:ry2, rx1:rx2]
    hsv_eq = preprocess_roi_hsv(
        roi,
        gaussian_ksize=cfg.gaussian_ksize,
        clahe_clip_limit=cfg.clahe_clip_limit,
        clahe_grid_size=cfg.clahe_grid_size,
    )
    roi_mask = build_split_red_mask(
        hsv_eq,
        profile=prof,
        median_ksize=cfg.median_ksize,
        morph_ksize=cfg.morph_ksize,
        open_iter=cfg.open_iter,
        close_iter=cfg.close_iter,
    )
    if active_roi is None:
        return roi_mask

    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[ry1:ry2, rx1:rx2] = roi_mask
    return full_mask
