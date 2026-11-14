"""The trainer's dataset plug for DSF: windows of a DROID index, each with one hidden exterior camera's pixel masks.

    TrainConfig.dataset = "flux_action.dsf.dataset:build_train"     (validation: build_validation, used by the trainer)
    TrainConfig.dsf = {"masks_root": <dsf_flux run folder>, "run": "k0_ringrandom", "split": <split.json>}

A DSF unit is (episode, hidden camera): the camera ext1 (``images.left``) or ext2 (``images.right``) whose robot and
handled object are hidden, the other one being the clean reference. Its windows are the starts FLUX may draw (the
index's valid starts) that are multiples of 4, so every latent frame is one ``dsf_flux`` block, and whose 8 blocks hide
at least one cell (exact cells, before augmentation). Every DSF sample is one of these windows; windows without a hidden
cell are not used. An epoch visits every unit ``visits_per_epoch`` times and draws one of its windows per visit,
seeded by (run seed, epoch, episode, camera, visit), like the index loader draws one window per episode.

The masks: ``<masks_root>/ep_<episode_index>/blocks_<run>.npz`` (``dsf_flux/plan.py``). Per window the dataset attaches
``dsf_union`` (8, 28800) uint8 (the hidden camera's packed 360x640 mask per latent frame, zeros when not high),
``dsf_block_high`` (8,), ``dsf_hidden`` (0 ext1, 1 ext2), ``dsf_random_ring``, ``dsf_episode_index``, ``dsf_start``;
the policy turns them into token masks after the window's own crop augmentation (``flux_action.dsf.masks``).

The split, by episode (``python -m flux_action.dsf.dataset split <masks_root> --validation 0.1 --seed 2026``), is a
JSON ``{"train": [...], "validation": [...]}`` of Cosmos3-DROID episode indices.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from random import Random
from typing import Any

import numpy as np
import torch

from ..data.droid.episodes import select_caption
from ..data.droid.index import ROWS_FILENAME
from ..training.data import WindowDataset, load_manifest
from . import masks

SPLITS = ("train", "validation")


def blocks_path(masks_root: Path, episode_index: int, run: str) -> Path:
    return Path(masks_root) / f"ep_{episode_index}" / f"blocks_{run}.npz"


def episodes_with_masks(masks_root: Path, run: str) -> list[int]:
    """The Cosmos3-DROID episode indices that have a ``blocks_<run>.npz``."""
    found = Path(masks_root).glob(f"ep_*/blocks_{run}.npz")
    return sorted(int(path.parent.name.removeprefix("ep_")) for path in found)


def write_split(masks_root: Path, output: Path, run: str, validation: float = 0.1, seed: int = 2026) -> dict:
    """A seeded split of the episodes with masks, by episode."""
    episodes = episodes_with_masks(masks_root, run)
    shuffled = episodes[:]
    random.Random(seed).shuffle(shuffled)
    count = round(len(episodes) * validation)
    split = {"seed": seed, "run": run, "validation": sorted(shuffled[:count]), "train": sorted(shuffled[count:])}
    Path(output).write_text(json.dumps(split))
    return split


class DSFWindowDataset(WindowDataset):
    """A ``WindowDataset`` whose positions are DSF units and whose windows carry the hidden camera's pixel masks."""

    def __init__(
        self,
        manifest: dict,
        source_root,
        rows_path,
        *,
        masks_root,
        run: str,
        episodes: set[int],
        visits_per_epoch: int = 1,
        **loader: Any,
    ):
        self.masks_root, self.run = Path(masks_root), run
        chunk = int(manifest["chunk_size"])
        if chunk != masks.CHUNK:
            raise ValueError(f"DSF windows hold {masks.CHUNK} actions, the index {chunk}")
        kept, self.units, self.skipped = [], [], {}
        probe = WindowDataset.__new__(WindowDataset)  # only for valid_starts on the kept episodes
        probe.n_obs_steps, probe.chunk_size = None, chunk
        for episode in manifest["episodes"]:
            index = int(episode["episode_index"])
            if index not in episodes:
                continue
            path = blocks_path(self.masks_root, index, run)
            if not path.exists():
                self.skipped[index] = "no masks"
                continue
            with np.load(path) as blocks:
                block_high, cells = blocks["block_high"], {c: blocks[f"cells_{c}"] for c in masks.HIDDEN_CAMERAS}
            expected_blocks = (int(episode["n_frames"]) - 1) // masks.BLOCK_FRAMES
            if len(block_high) != expected_blocks:
                raise ValueError(
                    f"episode {index}: the masks have {len(block_high)} blocks, the index episode {expected_blocks} "
                    "(a different Cosmos3-DROID release?)"
                )
            probe.episodes = [episode]
            starts = [s for s in probe.valid_starts(0) if s % masks.BLOCK_FRAMES == 0]
            position = len(kept)
            units = []
            for hidden, camera in enumerate(masks.HIDDEN_CAMERAS):
                noisy = [
                    s
                    for s in starts
                    if s // masks.BLOCK_FRAMES + masks.PREDICTED_BLOCKS <= len(block_high)
                    and cells[camera][s // masks.BLOCK_FRAMES : s // masks.BLOCK_FRAMES + masks.PREDICTED_BLOCKS].any()
                ]
                if noisy:
                    units.append((position, hidden, noisy))
            if units:
                kept.append(episode)
                self.units.extend(units)
            else:
                self.skipped[index] = "no window with a hidden cell"
        self.n_windows = sum(len(unit[2]) for unit in self.units)
        super().__init__(
            {**manifest, "episodes": kept}, source_root, rows_path, split=None, visits_per_epoch=visits_per_epoch,
            **loader,
        )

    @property
    def n_positions(self) -> int:
        """Unit visits per epoch: position ``p`` is unit ``p % n_units``, visit ``p // n_units``."""
        return len(self.units) * self.visits_per_epoch

    def window(self, position: int) -> dict[str, Any]:
        episode_position, hidden, starts = self.units[position % len(self.units)]
        visit = position // len(self.units)
        episode = self.episodes[episode_position]
        index = int(episode["episode_index"])
        rng = Random(f"dsf:{self.seed}:{self.epoch}:{index}:{hidden}:{visit}")
        start = starts[rng.randrange(len(starts))]
        task = select_caption(episode["caption"], rng)
        item = self.window_at(episode_position, start, task=task, window_seed=rng.getrandbits(63))
        item.update(self.window_masks(index, hidden, start))
        return item

    def window_masks(self, episode_index: int, hidden: int, start: int) -> dict[str, Any]:
        """The DSF fields of a window (see the module docstring)."""
        camera = masks.HIDDEN_CAMERAS[hidden]
        union = np.zeros((masks.PREDICTED_BLOCKS, masks.PACKED_BYTES), np.uint8)
        block_high = np.zeros(masks.PREDICTED_BLOCKS, bool)
        with np.load(blocks_path(self.masks_root, episode_index, self.run)) as blocks:
            high_index = blocks["high_index"]
            packed = blocks[f"union_{camera}"]
            random_ring = bool(blocks["random_ring"]) if "random_ring" in blocks.files else False
            if random_ring and int(blocks["ring_tokens"]):
                raise ValueError(f"episode {episode_index}: a random ring and a uniform ring are separate modes")
            if not random_ring and int(blocks["ring_tokens"]):
                raise ValueError(f"episode {episode_index}: uniform rings (ring_tokens) are not supported here")
        for j in range(masks.PREDICTED_BLOCKS):
            block = start // masks.BLOCK_FRAMES + j
            where = int(np.searchsorted(high_index, block))
            if where < len(high_index) and high_index[where] == block:
                union[j] = packed[where]
                block_high[j] = True
        return {
            "dsf_union": torch.from_numpy(union),
            "dsf_block_high": torch.from_numpy(block_high),
            "dsf_hidden": hidden,
            "dsf_random_ring": random_ring,
            "dsf_episode_index": episode_index,
            "dsf_start": start,
        }


def _build(config, split: str, **loader: Any) -> DSFWindowDataset:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    settings = config.dsf or {}
    for key in ("masks_root", "run", "split"):
        if key not in settings:
            raise ValueError(f"TrainConfig.dsf needs {key!r} for the DSF dataset")
    episodes = set(json.loads(Path(settings["split"]).read_text())[split])
    index = Path(config.index_dir)
    loader.pop("n_obs_steps", None)
    return DSFWindowDataset(
        load_manifest(index / "manifest.json"),
        config.source_root,
        index / ROWS_FILENAME,
        masks_root=settings["masks_root"],
        run=settings["run"],
        episodes=episodes,
        visits_per_epoch=config.visits_per_epoch,
        decoder=config.decoder,
        frame_hw=config.frame_hw,
        **loader,
    )


def build_train(config, **loader: Any) -> DSFWindowDataset:
    """``TrainConfig.dataset = "flux_action.dsf.dataset:build_train"``."""
    return _build(config, "train", **loader)


def build_validation(config, **loader: Any) -> DSFWindowDataset:
    """The held-out windows, for the trainer's validation (``TrainConfig.validation_every``)."""
    return _build(config, "validation", **loader)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    split = commands.add_parser("split", help="a seeded train / validation split by episode")
    split.add_argument("masks_root", type=Path, help="the dsf_flux run folder (ep_<E>/blocks_<run>.npz)")
    split.add_argument("--output", type=Path, help="default: <masks_root>/split.json")
    split.add_argument("--run", default="k0_ringrandom")
    split.add_argument("--validation", type=float, default=0.1)
    split.add_argument("--seed", type=int, default=2026)
    arguments = parser.parse_args()
    output = arguments.output or arguments.masks_root / "split.json"
    result = write_split(arguments.masks_root, output, arguments.run, arguments.validation, arguments.seed)
    print(f"{len(result['train'])} train / {len(result['validation'])} validation episodes -> {output}")


if __name__ == "__main__":
    main()
