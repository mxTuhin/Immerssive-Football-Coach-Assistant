"""Homographies (image px <-> pitch metres), RANSAC, line-alignment check."""

from __future__ import annotations

import numpy as np

from fmi_schema.pitch import PitchDims


def _normalize(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = pts.mean(0)
    d = np.sqrt(((pts - c) ** 2).sum(1)).mean()
    s = np.sqrt(2) / max(d, 1e-9)
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    ph = np.c_[pts, np.ones(len(pts))] @ T.T
    return ph[:, :2], T


def dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    """Normalised DLT homography mapping src -> dst (N >= 4)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if len(src) < 4:
        return None
    s, Ts = _normalize(src)
    d, Td = _normalize(dst)
    A = []
    for (x, y), (u, v) in zip(s, d):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, Vt = np.linalg.svd(np.asarray(A))
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    if abs(H[2, 2]) < 1e-12:
        return None
    return H / H[2, 2]


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, float).reshape(-1, 2)
    ph = np.c_[pts, np.ones(len(pts))] @ H.T
    w = ph[:, 2:3]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = ph[:, :2] / w
    out[np.abs(w[:, 0]) < 1e-12] = np.nan
    return out


def fit_homography(src: np.ndarray, dst: np.ndarray, thresh: float, iters: int = 300,
                   seed: int = 0) -> tuple[np.ndarray | None, np.ndarray]:
    """RANSAC homography src -> dst; ``thresh`` is in dst units. Returns (H, inlier_mask)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    n = len(src)
    if n < 4:
        return None, np.zeros(n, bool)
    try:
        import cv2  # type: ignore

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, thresh)
        if H is not None:
            return H / H[2, 2], mask.ravel().astype(bool)
    except ImportError:
        pass
    rng = np.random.default_rng(seed)
    best_H, best_in = None, np.zeros(n, bool)
    if n == 4:
        H = dlt(src, dst)
        return H, np.ones(n, bool) if H is not None else np.zeros(n, bool)
    for _ in range(iters):
        idx = rng.choice(n, 4, replace=False)
        H = dlt(src[idx], dst[idx])
        if H is None:
            continue
        err = np.linalg.norm(apply_h(H, src) - dst, axis=1)
        inl = err < thresh
        if inl.sum() > best_in.sum():
            best_H, best_in = H, inl
    if best_H is not None and best_in.sum() >= 4:
        best_H = dlt(src[best_in], dst[best_in])
    return best_H, best_in


def leave_one_out_error(src: np.ndarray, dst: np.ndarray) -> float:
    """Median held-out reprojection error (dst units): each point predicted by an H fitted without it."""
    n = len(src)
    if n < 5:
        return float("nan")
    errs = []
    for i in range(n):
        m = np.arange(n) != i
        H = dlt(src[m], dst[m])
        if H is None:
            continue
        errs.append(float(np.linalg.norm(apply_h(H, src[i:i + 1])[0] - dst[i])))
    return float(np.median(errs)) if errs else float("nan")


def line_mask(gray: np.ndarray, kernel: int = 15, thresh: float = 25.0) -> np.ndarray | None:
    """Bright thin structures (painted lines) via white top-hat. None for textureless frames."""
    if gray.std() < 1e-6:
        return None
    try:
        import cv2  # type: ignore

        k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
        th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
        m = (th > thresh).astype(np.uint8)
        return cv2.dilate(m, np.ones((5, 5), np.uint8)) > 0
    except ImportError:
        # numpy fallback: pixel brighter than its local horizontal + vertical mean
        g = gray.astype(float)
        pad = kernel // 2
        p = np.pad(g, pad, mode="edge")
        cs = p.cumsum(0).cumsum(1)
        H, W = g.shape
        s = cs[kernel:kernel + H, kernel:kernel + W] - cs[:H, kernel:kernel + W] - cs[kernel:kernel + H, :W] + cs[:H, :W]
        local = s / (kernel * kernel)
        return (g - local) > thresh


def line_score(gray: np.ndarray | None, H_img2pitch: np.ndarray, pitch: PitchDims, step: float = 0.5) -> float:
    """Fraction of painted-line samples (projected into the image) that land on line pixels.

    The pitch lines are not the keypoints the homography was fitted on, so this is an independent check.
    Returns NaN when the frame has no usable texture (e.g. synthetic frames).
    """
    if gray is None:
        return float("nan")
    mask = line_mask(gray)
    if mask is None:
        return float("nan")
    try:
        Hinv = np.linalg.inv(H_img2pitch)
    except np.linalg.LinAlgError:
        return 0.0
    pts = apply_h(Hinv, pitch.line_samples(step))
    Hh, Ww = mask.shape
    ok = np.isfinite(pts).all(1) & (pts[:, 0] >= 0) & (pts[:, 0] < Ww) & (pts[:, 1] >= 0) & (pts[:, 1] < Hh)
    if ok.sum() < 20:
        return float("nan")
    u, v = pts[ok, 0].astype(int), pts[ok, 1].astype(int)
    return float(mask[v, u].mean())


def h_to_row(H: np.ndarray) -> dict:
    return {f"h{i}{j}": float(H[i, j]) for i in range(3) for j in range(3)}


def row_to_h(row) -> np.ndarray:
    return np.array([[row[f"h{i}{j}"] for j in range(3)] for i in range(3)], dtype=float)
