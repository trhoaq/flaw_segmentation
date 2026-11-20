from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_binary_mask(path: Path, encoding: str) -> np.ndarray:
    with Image.open(io.BytesIO(path.read_bytes())) as im:
        a = np.asarray(im)
    if encoding == "red_green":
        if a.ndim != 3 or a.shape[2] < 3:
            raise ValueError(f"Expected RGB red/green mask: {path}")
        return ((a[..., 0] >= 128) & (a[..., 0] > a[..., 1]) & (a[..., 0] > a[..., 2])).astype(np.uint8)
    if a.ndim == 3:
        if not np.all(a[..., :3] == a[..., :1]):
            raise ValueError(f"Unexpected non-grayscale mask: {path}")
        a = a[..., 0]
    if encoding == "indexed":
        if not set(np.unique(a)).issubset({0, 1}):
            raise ValueError(f"Unknown indexed class values in {path}: {np.unique(a)}")
        return (a == 1).astype(np.uint8)
    if encoding == "binary_255":
        return (a > 127).astype(np.uint8)
    if encoding == "binary":
        return (a > 0).astype(np.uint8)
    raise ValueError(encoding)


def closest_vertices(outer: np.ndarray, hole: np.ndarray) -> tuple[int, int]:
    best, pair = float("inf"), (0, 0)
    for start in range(0, len(outer), 64):
        distances = ((outer[start:start + 64, None, :] - hole[None, :, :]) ** 2).sum(axis=2)
        i, j = np.unravel_index(distances.argmin(), distances.shape)
        if distances[i, j] < best:
            best, pair = distances[i, j], (start + int(i), int(j))
    return pair


def mask_to_polygons(mask: np.ndarray) -> list[np.ndarray]:
    """Trace 2x pixel footprints; bridge hole rings without filling their interiors.

    The 0.1-pixel inset keeps integer rasterization stable after text serialization.
    A 2x grid gives single-pixel/one-pixel-wide components valid nonzero polygons.
    Connected regions are derived instances because source masks are semantic masks.
    """
    if mask.ndim != 2 or not np.isin(mask, [0, 1]).all():
        raise ValueError("Expected a 2D binary mask")
    h, w = mask.shape
    enlarged = np.repeat(np.repeat(mask.astype(np.uint8), 2, axis=0), 2, axis=1)
    contours, hierarchy = cv2.findContours(enlarged, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    polygons = []
    for i, contour in enumerate(contours):
        if hierarchy[0, i, 3] != -1:
            continue
        points = contour.reshape(-1, 2).astype(np.float64)
        child = int(hierarchy[0, i, 2])
        while child != -1:
            hole = contours[child].reshape(-1, 2).astype(np.float64)
            a, b = closest_vertices(points, hole)
            ring = np.concatenate([hole[b:], hole[:b + 1]])
            points = np.concatenate([points[:a + 1], ring, points[a:]])
            child = int(hierarchy[0, child, 0])
        polygon = (points / 2 + 0.1) / np.array([w, h])
        if len(polygon) < 3 or cv2.contourArea(polygon.astype(np.float32)) <= 0:
            raise ValueError("Nonempty mask produced a degenerate polygon")
        polygons.append(polygon)
    return polygons


def serialize_polygons(polygons: list[np.ndarray]) -> str:
    return "".join("0 " + " ".join(f"{v:.10f}" for v in p.ravel()) + "\n" for p in polygons)


def parse_labels(text: str) -> list[np.ndarray]:
    polygons = []
    for line in text.splitlines():
        values = np.array([float(x) for x in line.split()], dtype=np.float64)
        if len(values) < 7 or len(values) % 2 != 1 or values[0] != 0:
            raise ValueError("Expected class 0 followed by at least three polygon vertices")
        points = values[1:].reshape(-1, 2)
        if not np.isfinite(points).all() or (points < 0).any() or (points > 1).any():
            raise ValueError("Polygon coordinates must be finite and normalized to [0, 1]")
        if cv2.contourArea(points.astype(np.float32)) <= 0:
            raise ValueError("Degenerate polygon")
        polygons.append(points)
    return polygons


def rasterize(polygons: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    result = np.zeros(shape, dtype=np.uint8)
    for p in polygons:
        coordinates = (p * np.array([w, h])).astype(np.int32)
        cv2.fillPoly(result, [coordinates], color=1)
    return result


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.count_nonzero(first | second)
    return float(np.count_nonzero(first & second) / union) if union else 1.0
