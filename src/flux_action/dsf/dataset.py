"""The trainer's dataset plug for DSF: windows of a DROID index, each with one hidden exterior camera's pixel masks.

    TrainConfig.dataset = "flux_action.dsf.dataset:build_train"     (validation: build_validation, used by the trainer)
    TrainConfig.dsf = {"masks_root": <dsf_flux run folder>, "run": "k0_ringrandom", "split": <split.json>}

A DSF unit is (episode, hidden camera): the camera ext1 (``images.left``) or ext2 (``images.right``) whose robot and
handled object are hidden, the other one being the clean reference. Its windows are the starts FLUX may draw (the
index's valid starts) that are multiples of 4, so every latent frame is one ``dsf_flux`` block, whose 8 blocks hide
at least one cell (exact cells, before augmentation), and whose future holds a gripper close: the commanded gripper
crosses ``CLOSE_THRESHOLD`` upward between action steps ``CLOSE_STEPS`` (8-27 of 32: 0.53-1.8 s at 15 Hz), so the
approach before the grasp and its outcome after it are both in the window. Every DSF sample is one of these windows,
except ordinary ones. With ``whole_window`` a window must also have a mask for all 8 of its latent frames (masks made
for whole-window hiding: the robot hidden in that camera through the whole predicted future, so nothing in the same
camera shows it a moment before or after), and its 32 actions are all at t_high (``dsf_all_actions``): clean absolute
joint positions would let the model draw the hidden robot from the joints instead of from the other camera.

``ordinary_fraction`` (training only) turns that share of the visits into ordinary windows: any valid start of the
unit's episode, nothing hidden (``dsf_ordinary``), one noise level from the policy's own distribution. They keep the
regular video and action prediction trained next to the DSF windows. An epoch visits every unit ``visits_per_epoch`` times and draws one of its windows per visit,
seeded by (run seed, epoch, episode, camera, visit), like the index loader draws one window per episode.

The masks: ``<masks_root>/ep_<episode_index>/blocks_<run>.npz`` (``dsf_flux/plan.py``). Per window the dataset attaches
``dsf_union`` (8, 28800) uint8 (the hidden camera's packed 360x640 mask per latent frame, zeros when not high),
``dsf_block_high`` (8,), ``dsf_hidden`` (0 ext1, 1 ext2), ``dsf_random_ring``, ``dsf_episode_index``, ``dsf_start``;
the policy turns them into token masks after the window's own crop augmentation (``flux_action.dsf.masks``).

``python -m flux_action.dsf.dataset count <source_root> <keep_ranges.json>`` counts these windows (close rule only) on
the Cosmos3-DROID data and meta files, without videos or masks.

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
GRIPPER_ACTION_COLUMN = 15  # rows.f32.npy: state (7 joints, gripper), then action (7 joints, gripper)
CLOSE_THRESHOLD = 0.5  # commanded gripper: 0 open, 1 closed
CLOSE_STEPS = (8, 27)  # a close at action step k of the window (inclusive range) keeps it


def close_steps(gripper: np.ndarray) -> np.ndarray:
    """Frames ``i`` of an episode where the commanded gripper closes: ``gripper[i - 1] < 0.5 <= gripper[i]``."""
    gripper = np.asarray(gripper, np.float32)
    return np.flatnonzero((gripper[:-1] < CLOSE_THRESHOLD) & (gripper[1:] >= CLOSE_THRESHOLD)) + 1


def close_starts(gripper: np.ndarray, starts: list[int]) -> list[int]:
    """The ``starts`` whose window closes the gripper at action step ``CLOSE_STEPS[0]..CLOSE_STEPS[1]``."""
    closes = close_steps(gripper)
    first, last = CLOSE_STEPS
    return [s for s in starts if ((closes >= s + first) & (closes <= s + last)).any()]


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
        whole_window: bool = False,
        ordinary_fraction: float = 0.0,
        **loader: Any,
    ):
        if not 0 <= ordinary_fraction < 1:
            raise ValueError("ordinary_fraction must be in [0, 1)")
        self.masks_root, self.run = Path(masks_root), run
        self.whole_window, self.ordinary_fraction = bool(whole_window), float(ordinary_fraction)
        chunk = int(manifest["chunk_size"])
        if chunk != masks.CHUNK:
            raise ValueError(f"DSF windows hold {masks.CHUNK} actions, the index {chunk}")
        kept, self.units, self.skipped = [], [], {}
        rows = np.load(rows_path, mmap_mode="r")
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
            first_row = int(episode["from_index"])
            closing = set(close_starts(rows[first_row : first_row + int(episode["n_frames"]), GRIPPER_ACTION_COLUMN], starts))
            position = len(kept)
            units, any_hidden = [], False
            for hidden, camera in enumerate(masks.HIDDEN_CAMERAS):
                noisy = [
                    s
                    for s in starts
                    if s // masks.BLOCK_FRAMES + masks.PREDICTED_BLOCKS <= len(block_high)
                    and cells[camera][s // masks.BLOCK_FRAMES : s // masks.BLOCK_FRAMES + masks.PREDICTED_BLOCKS].any()
                    and (
                        not self.whole_window
                        or block_high[s // masks.BLOCK_FRAMES : s // masks.BLOCK_FRAMES + masks.PREDICTED_BLOCKS].all()
                    )
                ]
                any_hidden = any_hidden or bool(noisy)
                chosen = [s for s in noisy if s in closing]
                if chosen:
                    units.append((position, hidden, chosen))
            if units:
                kept.append(episode)
                self.units.extend(units)
            else:
                self.skipped[index] = "no gripper close in a hidden window" if any_hidden else "no window with a hidden cell"
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
        ordinary = self.ordinary_fraction > 0 and rng.random() < self.ordinary_fraction
        if ordinary:
            every = self.valid_starts(episode_position)
            start = every[rng.randrange(len(every))]
        else:
            start = starts[rng.randrange(len(starts))]
        task = select_caption(episode["caption"], rng)
        item = self.window_at(episode_position, start, task=task, window_seed=rng.getrandbits(63))
        item.update(self.ordinary_masks(index, hidden, start) if ordinary else self.window_masks(index, hidden, start))
        return item

    def ordinary_masks(self, episode_index: int, hidden: int, start: int) -> dict[str, Any]:
        """The DSF fields of an ordinary window: nothing hidden."""
        return {
            "dsf_union": torch.zeros(masks.PREDICTED_BLOCKS, masks.PACKED_BYTES, dtype=torch.uint8),
            "dsf_block_high": torch.zeros(masks.PREDICTED_BLOCKS, dtype=torch.bool),
            "dsf_hidden": hidden,
            "dsf_random_ring": False,
            "dsf_episode_index": episode_index,
            "dsf_start": start,
            "dsf_all_actions": False,
            "dsf_ordinary": True,
        }

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
            "dsf_all_actions": self.whole_window,
            "dsf_ordinary": False,
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
    unknown = set(settings) - {"masks_root", "run", "split", "whole_window", "ordinary_fraction"}
    if unknown:
        raise ValueError(f"unknown TrainConfig.dsf settings {sorted(unknown)}")
    loader.pop("n_obs_steps", None)
    return DSFWindowDataset(
        load_manifest(index / "manifest.json"),
        config.source_root,
        index / ROWS_FILENAME,
        masks_root=settings["masks_root"],
        run=settings["run"],
        episodes=episodes,
        visits_per_epoch=config.visits_per_epoch,
        whole_window=bool(settings.get("whole_window", False)),
        # validation always measures the DSF windows themselves
        ordinary_fraction=float(settings.get("ordinary_fraction", 0.0)) if split == "train" else 0.0,
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


def count_close_windows(source_root, filter_path, episodes: set[int] | None = None) -> dict:
    """Windows the close rule keeps on Cosmos3-DROID (data + meta files only): starts that are multiples of 4 among
    the valid starts, with a gripper close at action step ``CLOSE_STEPS``. Masks are not consulted."""
    import pyarrow.parquet as pq

    from ..data.droid.index import build_manifest
    from ..data.windows import valid_window_spans

    root = Path(source_root)
    manifest = build_manifest(root, filter_path, chunk_size=masks.CHUNK)
    by_file: dict[str, list[dict]] = {}
    for episode in manifest["episodes"]:
        if episodes is None or int(episode["episode_index"]) in episodes:
            by_file.setdefault(episode["data_file"], []).append(episode)
    totals = {"episodes": 0, "episodes_with_close_window": 0, "closes": 0, "windows": 0, "close_windows": 0}
    for data_file, group in sorted(by_file.items()):
        table = pq.read_table(root / data_file, columns=["index", "action.gripper_position"])
        index = table["index"].to_numpy()
        gripper = table["action.gripper_position"].combine_chunks()
        gripper = (gripper.flatten() if hasattr(gripper, "flatten") else gripper).to_numpy()  # [g] or g per row
        position = {int(i): p for p, i in enumerate(index)}
        for episode in group:
            n, first = int(episode["n_frames"]), position[int(episode["from_index"])]
            grip = gripper[first : first + n]
            spans = valid_window_spans(n, episode["valid_ranges"], masks.CHUNK)
            starts = [s for f, c in spans for s in range(f, f + c) if s % masks.BLOCK_FRAMES == 0]
            kept = close_starts(grip, starts)
            totals["episodes"] += 1
            totals["episodes_with_close_window"] += bool(kept)
            totals["closes"] += len(close_steps(grip))
            totals["windows"] += len(starts)
            totals["close_windows"] += len(kept)
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    split = commands.add_parser("split", help="a seeded train / validation split by episode")
    split.add_argument("masks_root", type=Path, help="the dsf_flux run folder (ep_<E>/blocks_<run>.npz)")
    split.add_argument("--output", type=Path, help="default: <masks_root>/split.json")
    split.add_argument("--run", default="k0_ringrandom")
    split.add_argument("--validation", type=float, default=0.1)
    split.add_argument("--seed", type=int, default=2026)
    count = commands.add_parser("count", help="windows the gripper-close rule keeps (no videos or masks needed)")
    count.add_argument("source_root", type=Path, help="Cosmos3-DROID root with success/data and success/meta")
    count.add_argument("filter", type=Path, help="keep_ranges_1_0_1.json")
    count.add_argument("--episodes", type=Path, help="a JSON list of episode indices, or a split.json (train+validation)")
    arguments = parser.parse_args()
    if arguments.command == "count":
        chosen = None
        if arguments.episodes:
            listed = json.loads(arguments.episodes.read_text())
            chosen = set(listed["train"] + listed["validation"]) if isinstance(listed, dict) else set(listed)
        print(json.dumps(count_close_windows(arguments.source_root, arguments.filter, chosen), indent=1))
        return
    output = arguments.output or arguments.masks_root / "split.json"
    result = write_split(arguments.masks_root, output, arguments.run, arguments.validation, arguments.seed)
    print(f"{len(result['train'])} train / {len(result['validation'])} validation episodes -> {output}")


if __name__ == "__main__":
    main()
