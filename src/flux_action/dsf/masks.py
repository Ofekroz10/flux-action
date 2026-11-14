"""Pixel masks -> latent token masks, on FLUX's DROID canvas. Pure numpy, no model needed.

The canvas (``processing.packing.compose_canvas``, layout ``droid``): the wrist camera (360x640) on top at full size;
the left (``ext1``) and right (``ext2``) cameras shrunk to 180x320 (bilinear, align_corners=False) side by side below
-> a 540x640 composite, reflect-padded to 544x736 -> VAE /32 -> a 17x20 token grid per latent frame (the padded cells
are cropped away). A token is hidden if ANY pixel of it is hidden, in any of its latent frame's 4 frames.

Three facts make the step exact:
  1. the 2x bilinear shrink (no antialiasing) is the mean of each 2x2 block, so a shrunk pixel is hidden if any of its
     4 source pixels is;
  2. the crop augmentation (``sample_augmentation``: a crop resized back with bilinear interpolation) moves a hidden
     pixel to every output pixel whose bilinear support contains it;
  3. only the 540x640 content counts, never the reflect padding.
Same rules and numbers as ``dsf_flux`` (``geometry.py``, ``window.py``); ``tests/test_dsf.py`` checks them.
"""

from __future__ import annotations

import numpy as np

CAMERA_HW = (360, 640)
HALF_HW = (180, 320)
COMPOSITE_HW = (540, 640)
CELL = 32
LATENT_HW = (17, 20)
BLOCK_FRAMES = 4  # one latent frame = 4 frames at 15 Hz
PREDICTED_BLOCKS = 8  # latent frames 1..8 of a 33-frame window
CHUNK = PREDICTED_BLOCKS * BLOCK_FRAMES  # 32 action tokens
HIDDEN_CAMERAS = ("ext1", "ext2")  # ext1 = images.left, ext2 = images.right; the wrist is never hidden
PANEL = {"ext1": (0, 360, 320, 540), "ext2": (320, 360, 640, 540)}  # [x0, y0, x1, y1) in the composite
RING_CELL_PROBABILITY = 0.5
PACKED_BYTES = CAMERA_HW[0] * CAMERA_HW[1] // 8
FRACTION_EPS = 1e-6  # a bilinear weight below this is float rounding: the pixel does not count


def unpack_camera_mask(packed: np.ndarray) -> np.ndarray:
    """``np.packbits`` of a 360x640 mask -> the (360, 640) bool mask."""
    return np.unpackbits(np.asarray(packed, np.uint8))[: CAMERA_HW[0] * CAMERA_HW[1]].reshape(CAMERA_HW).astype(bool)


def _support(out_size: int, in_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bilinear resize in_size -> out_size (align_corners=False): per output index the first and second source index
    and whether the second has a nonzero weight."""
    source = np.maximum((np.arange(out_size) + 0.5) * in_size / out_size - 0.5, 0.0)
    first = np.minimum(np.floor(source).astype(int), in_size - 1)
    second = np.minimum(first + 1, in_size - 1)
    return first, second, (source - first) > FRACTION_EPS


def augment_support(mask: np.ndarray, augmentation: dict | None) -> np.ndarray:
    """The 360x640 mask after the window's crop augmentation (crop, then bilinear resize back): an output pixel is
    hidden if any source pixel with nonzero bilinear weight is. Colour jitter does not move pixels."""
    if augmentation is None:
        return mask
    top, left = augmentation["crop_top"], augmentation["crop_left"]
    height, width = augmentation["crop_height"], augmentation["crop_width"]
    cropped = mask[top : top + height, left : left + width]
    rows_first, rows_second, rows_extra = _support(CAMERA_HW[0], height)
    resized = cropped[rows_first] | (cropped[rows_second] & rows_extra[:, None])
    columns_first, columns_second, columns_extra = _support(CAMERA_HW[1], width)
    return resized[:, columns_first] | (resized[:, columns_second] & columns_extra[None, :])


def camera_cells(camera_mask: np.ndarray, camera: str) -> np.ndarray:
    """A 360x640 exterior-camera mask -> the (17, 20) cells it hides on the canvas."""
    if camera not in PANEL:
        raise ValueError(f"only exterior cameras are hidden, not {camera!r}")
    half = camera_mask.reshape(HALF_HW[0], 2, HALF_HW[1], 2).any(axis=(1, 3))
    left, top, right, bottom = PANEL[camera]
    canvas = np.zeros((LATENT_HW[0] * CELL, LATENT_HW[1] * CELL), bool)
    canvas[top:bottom, left:right] = half
    return canvas.reshape(LATENT_HW[0], CELL, LATENT_HW[1], CELL).any(axis=(1, 3))


def panel_cells(camera: str) -> np.ndarray:
    """(17, 20) True where the camera's panel touches a cell: the cells a ring may use."""
    left, top, right, bottom = PANEL[camera]
    inside = np.zeros(LATENT_HW, bool)
    inside[top // CELL : -(-bottom // CELL), left // CELL : -(-right // CELL)] = True
    return inside


def ring_field(episode_index: int, start: int, camera: str) -> np.ndarray:
    """The random ring's coin flips of one window: a 17x20 field uniform in [0, 1), seeded by (Cosmos episode_index,
    window start, hidden camera number 1/2). Its own generator, no global RNG (the same as ``dsf_flux``)."""
    number = HIDDEN_CAMERAS.index(camera) + 1
    return np.random.default_rng([int(episode_index), int(start), number]).random(LATENT_HW)


def add_random_ring(cells: np.ndarray, field: np.ndarray, camera: str, probability: float) -> np.ndarray:
    """The exact cells plus each 8-neighbour ring cell (inside the camera's panel, not exact) where field <
    probability. A block without exact cells gets no ring."""
    padded = np.pad(cells, 1)
    grown = np.zeros_like(cells)
    for row in range(3):
        for column in range(3):
            grown |= padded[row : row + cells.shape[0], column : column + cells.shape[1]]
    ring = grown & panel_cells(camera) & ~cells
    return cells | (ring & (field < probability))


def window_token_masks(
    union: np.ndarray,
    block_high: np.ndarray,
    camera: str,
    *,
    augmentation: dict | None = None,
    random_ring: bool = False,
    episode_index: int = 0,
    start: int = 0,
    probability: float = RING_CELL_PROBABILITY,
) -> tuple[np.ndarray, np.ndarray]:
    """A window's masks: ``union`` (8, PACKED_BYTES) the hidden camera's packed pixel mask per predicted latent frame
    (the OR of its 4 frames), ``block_high`` (8,) which of them hold one. Returns ``video`` (8, 17, 20) bool (token
    order t, h, w) and ``action`` (32,) bool: a latent frame's 4 actions are high if it hides at least one token.
    The cells follow ``augmentation``; the random ring (one field per window, shared by its 8 latent frames) is drawn
    after the crop and never changes which actions are high."""
    video = np.zeros((PREDICTED_BLOCKS, *LATENT_HW), bool)
    action = np.zeros(CHUNK, bool)
    field = ring_field(episode_index, start, camera) if random_ring else None
    for j in range(PREDICTED_BLOCKS):
        if not block_high[j]:
            continue
        cells = camera_cells(augment_support(unpack_camera_mask(union[j]), augmentation), camera)
        video[j] = add_random_ring(cells, field, camera, probability) if random_ring else cells
        action[BLOCK_FRAMES * j : BLOCK_FRAMES * (j + 1)] = cells.any()
    return video, action
