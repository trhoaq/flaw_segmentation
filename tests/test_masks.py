import cv2
import numpy as np
import pytest
from PIL import Image
from railflaw_seg.masks import load_binary_mask, mask_iou, mask_to_polygons, parse_labels, rasterize, serialize_polygons


@pytest.mark.parametrize('kind', ['tiny', 'thin', 'holes', 'disconnected', 'border', 'empty'])
def test_exact_mask_roundtrip(kind):
    m = np.zeros((24, 29), dtype=np.uint8)
    if kind == 'tiny': m[8, 9] = 1
    elif kind == 'thin': np.fill_diagonal(m, 1)
    elif kind == 'holes':
        m[1:23, 1:28] = 1
        m[4:8, 5:12] = 0
        m[12:19, 17:24] = 0
    elif kind == 'disconnected':
        m[2:5, 2:5] = 1
        m[12:20, 18:25] = 1
    elif kind == 'border': m[:] = 1
    p = parse_labels(serialize_polygons(mask_to_polygons(m)))
    assert mask_iou(m, rasterize(p, m.shape)) == 1.0


def test_binary_noise_and_palette_and_color(tmp_path):
    array = np.array([[0, 5, 120], [128, 250, 255]], dtype=np.uint8)
    path = tmp_path / 'binary.png'
    Image.fromarray(array).save(path)
    assert load_binary_mask(path, 'binary_255').tolist() == [[0, 0, 0], [1, 1, 1]]
    color = np.array([[[0, 255, 0], [255, 0, 0]]], dtype=np.uint8)
    Image.fromarray(color).save(path)
    assert load_binary_mask(path, 'red_green').tolist() == [[0, 1]]
    palette = Image.fromarray(np.array([[0, 1]], dtype=np.uint8)).convert('P')
    palette.save(path)
    assert load_binary_mask(path, 'indexed').tolist() == [[0, 1]]


@pytest.mark.parametrize('text', ['0 .1 .1 .2 .2', '1 .1 .1 .2 .1 .2 .2', '0 -1 0 1 0 1 1', '0 nan 0 1 0 1 1'])
def test_reject_invalid_or_detection_labels(text):
    with pytest.raises(ValueError): parse_labels(text)


def test_seeded_random_binary_masks_roundtrip():
    rng = np.random.default_rng(42)
    for _ in range(15):
        m = (rng.random((20, 21)) > .6).astype(np.uint8)
        result = rasterize(parse_labels(serialize_polygons(mask_to_polygons(m))), m.shape)
        assert mask_iou(m, result) == 1.0
