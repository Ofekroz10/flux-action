"""DSF: pixel masks -> tokens on FLUX's canvas, two noise levels in one forward, the dataset plug, validation logs."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from conftest import TINY_DIT, FakeVideoVAE, tiny_config
from synthetic_droid import build_compact_dataset
from torch.utils.data import IterableDataset

from flux_action.data.droid.index import ROWS_FILENAME, build_manifest, build_rows, write_manifest
from flux_action.dsf import masks, noise
from flux_action.dsf.dataset import (
    DSFWindowDataset,
    close_starts,
    close_steps,
    count_close_windows,
    write_split,
)
from flux_action.models.text_encoder import MockTextEncoder
from flux_action.policy import FluxActionPolicy
from flux_action.processing import packing
from flux_action.training.data import collate_windows
from flux_action.training.trainer import DSF_METRICS, TrainConfig, Trainer

NEUTRAL = {"color_fn_order": [0, 1, 2, 3], "brightness_factor": 1.0, "contrast_factor": 1.0,
           "saturation_factor": 1.0, "hue_factor": 0.0}


def droid_policy(**overrides):
    config = tiny_config(
        camera_layout="droid",
        camera_keys=("images.wrist", "images.left", "images.right"),
        canvas_hw=(544, 736),
        action_dim=8,
        gripper_flip_dims=(-1,),
        **overrides,
    )
    return FluxActionPolicy(config, video_vae=FakeVideoVAE(), text_encoder=MockTextEncoder(32))


def patch(top, left, bottom, right):
    mask = np.zeros(masks.CAMERA_HW, bool)
    mask[top:bottom, left:right] = True
    return mask


def canvas_cells(camera_mask, camera, augmentation):
    """The cells FLUX's own canvas code reaches from the hidden pixels: the camera image is 1 there and 0 elsewhere,
    the other cameras 0; a cell counts if any of its content pixels comes out above black."""
    cams = torch.zeros(3, 1, 3, *masks.CAMERA_HW)
    cams[1 + masks.HIDDEN_CAMERAS.index(camera), 0, :] = torch.from_numpy(camera_mask).float()
    x = cams.reshape(-1, 3, *masks.CAMERA_HW)
    if augmentation is not None:
        x = packing.apply_augmentation(x, augmentation)
    canvas = packing.compose_canvas(x.reshape(3, 1, 3, *masks.CAMERA_HW), "droid")  # (3, 1, 544, 736)
    content = (canvas[0, 0] > -1 + 1e-6)[: masks.COMPOSITE_HW[0], : masks.COMPOSITE_HW[1]].numpy()
    padded = np.zeros((masks.LATENT_HW[0] * masks.CELL, masks.LATENT_HW[1] * masks.CELL), bool)
    padded[: masks.COMPOSITE_HW[0], : masks.COMPOSITE_HW[1]] = content
    return padded.reshape(masks.LATENT_HW[0], masks.CELL, masks.LATENT_HW[1], masks.CELL).any(axis=(1, 3))


@pytest.mark.parametrize("camera", masks.HIDDEN_CAMERAS)
def test_cells_are_the_canvas_tokens_flux_builds_with_and_without_the_crop(camera):
    rng = np.random.default_rng(0)
    for trial in range(6):
        mask = rng.random(masks.CAMERA_HW) > 0.9995
        mask |= patch(*sorted(rng.integers(0, 360, 2)), *sorted(rng.integers(0, 640, 2)))
        augmentation = None
        if trial % 2:
            augmentation = {**packing.sample_augmentation(torch.Generator().manual_seed(trial)), **NEUTRAL}
        expected = canvas_cells(mask, camera, augmentation)
        got = masks.camera_cells(masks.augment_support(mask, augmentation), camera)
        np.testing.assert_array_equal(got, expected)
    # the wrist rows: the exteriors start at composite row 360, so camera row 0 hides cell row 11 only
    cells = masks.camera_cells(patch(0, 0, 1, 1), camera)
    assert list(zip(*np.nonzero(cells))) == [(11, 0 if camera == "ext1" else 10)]


def window_union(blocks: dict[int, np.ndarray]):
    union = np.zeros((masks.PREDICTED_BLOCKS, masks.PACKED_BYTES), np.uint8)
    high = np.zeros(masks.PREDICTED_BLOCKS, bool)
    for j, mask in blocks.items():
        union[j], high[j] = np.packbits(mask), True
    return union, high


def test_window_token_masks_actions_and_random_ring():
    union, high = window_union({2: patch(100, 200, 160, 260), 5: np.zeros(masks.CAMERA_HW, bool)})
    video, action = masks.window_token_masks(union, high, "ext1")
    assert video.shape == (8, 17, 20) and action.shape == (32,)
    assert list(np.nonzero(video.any(axis=(1, 2)))[0]) == [2]          # block 5 is high but sees nothing
    assert list(np.nonzero(action)[0]) == list(range(8, 12))
    assert not video[:, :, 10:].any()                                    # ext1 is the left half
    rings = []
    for start in range(0, 400, 4):
        ring, ring_action = masks.window_token_masks(union, high, "ext1", random_ring=True, episode_index=7, start=start)
        exact, grown = video[2], masks.add_random_ring(video[2], np.zeros(masks.LATENT_HW), "ext1", 1.0)
        assert (ring[2] >= exact).all() and (ring[2] <= grown).all()
        np.testing.assert_array_equal(ring_action, action)
        assert not ring[[0, 1, 3, 4, 5, 6, 7]].any()                    # no ring from nothing
        rings.append(ring[2])
    again, _ = masks.window_token_masks(union, high, "ext1", random_ring=True, episode_index=7, start=0)
    np.testing.assert_array_equal(again, masks.window_token_masks(union, high, "ext1", random_ring=True,
                                                                  episode_index=7, start=0)[0])
    added = np.mean([(r & ~video[2]).sum() for r in rings]) / (grown & ~video[2]).sum()
    assert 0.4 < added < 0.6 and len({r.tobytes() for r in rings}) > 50


def test_levels_and_per_token_noise():
    low, high = noise.draw_levels(1000, "uniform", width=0.75, shift=42.0, generator=torch.Generator().manual_seed(0))
    assert (low <= high).all() and (high <= 1).all() and 0.4 < low.mean() < 0.6
    recipe, _ = noise.draw_levels(1000, "recipe", width=0.75, shift=42.0, generator=torch.Generator().manual_seed(0))
    assert recipe.median() > 0.9
    video_t, action_t = noise.token_timesteps(low[:2], high[:2], torch.tensor([[1, 0, 0]]).bool().repeat(2, 1),
                                              torch.tensor([[0, 1]]).bool().repeat(2, 1))
    assert video_t[:, 0].tolist() == high[:2].tolist() and video_t[:, 1].tolist() == low[:2].tolist()
    x0 = torch.zeros(2, 3, 4)
    x_t, _ = packing.add_noise(x0 + 1, video_t, torch.Generator().manual_seed(1))
    eps = torch.randn((2, 3, 4), generator=torch.Generator().manual_seed(1))
    torch.testing.assert_close(x_t, video_t[..., None] * eps + (1 - video_t[..., None]))


def dsf_batch(hidden_block: int = 2, seed: int = 0):
    """Two DROID windows: window 0 hides a patch of ext1 in latent frame ``hidden_block``; window 1 hides nothing."""
    g = torch.Generator().manual_seed(seed)
    batch = {f"images.{k}": torch.randint(0, 255, (2, 33, 3, 360, 640), dtype=torch.uint8, generator=g)
             for k in ("wrist", "left", "right")}
    union0, high0 = window_union({hidden_block: patch(100, 200, 200, 300)})
    union1, high1 = window_union({})
    batch.update(
        state=torch.rand(2, 8, generator=g),
        action=torch.rand(2, 32, 8, generator=g),
        task=["put the cup in the sink", "open the drawer"],
        window_seed=torch.tensor([11, 12]),
        dsf_union=torch.from_numpy(np.stack([union0, union1])),
        dsf_block_high=torch.from_numpy(np.stack([high0, high1])),
        dsf_hidden=torch.tensor([0, 0]),
        dsf_random_ring=torch.tensor([False, False]),
        dsf_episode_index=torch.tensor([3, 4]),
        dsf_start=torch.tensor([40, 80]),
    )
    return batch


def test_forward_noises_the_hidden_tokens_and_their_actions_at_t_high(monkeypatch):
    policy = droid_policy(augment=True, caption_dropout=0.0).train()
    seen, crops = {}, []
    build = packing.build_forward_kwargs
    sample = packing.sample_augmentation

    def recording_build(video, action, text, t, modality, generator=None, **kwargs):
        out = build(video, action, text, t, modality, generator, **kwargs)
        seen.update(out[0])
        return out

    def recording_sample(*args, **kwargs):
        crops.append(sample(*args, **kwargs))
        return crops[-1]

    monkeypatch.setattr(packing, "build_forward_kwargs", recording_build)
    monkeypatch.setattr(packing, "sample_augmentation", recording_sample)
    torch.manual_seed(5)
    batch = dsf_batch()
    loss, info = policy(batch)
    loss.backward()
    assert torch.isfinite(loss) and set(DSF_METRICS) <= set(info)
    # the masks the policy used follow window 0's own crop
    expected, expected_actions = masks.window_token_masks(
        batch["dsf_union"][0].numpy(), batch["dsf_block_high"][0].numpy(), "ext1", augmentation=crops[0]
    )
    video_t, action_t = seen["x_video_timesteps"], seen[f"x_{policy.modality}_timesteps"]
    assert video_t.shape == (2, 8 * 17 * 20) and action_t.shape == (2, 32)
    levels = video_t[0].unique()
    assert len(levels) == 2                                             # t_low and t_high, nothing else
    low, high = levels.min(), levels.max()
    np.testing.assert_array_equal((video_t[0] == high).numpy(), expected.reshape(-1))
    np.testing.assert_array_equal((action_t[0] == high).numpy(), expected_actions)
    assert expected_actions[8:12].all() and not expected_actions[:8].any()
    assert len(video_t[1].unique()) == 1 and len(action_t[1].unique()) == 1   # window 1: one level everywhere
    assert (seen["x_video_cond_timesteps"] == 0).all() and (seen[f"x_{policy.modality}_cond_timesteps"] == 0).all()
    assert float(low) <= float(high)
    assert 0 < float(info["dsf_video_hidden_share"]) < 0.1


def test_fixed_timestep_noises_every_token_at_one_level(monkeypatch):
    policy = droid_policy().eval()
    seen = {}
    build = packing.build_forward_kwargs

    def recording_build(*args, **kwargs):
        out = build(*args, **kwargs)
        seen.update(out[0])
        return out

    monkeypatch.setattr(packing, "build_forward_kwargs", recording_build)
    with torch.no_grad():
        loss, _ = policy(dsf_batch(), fixed_timestep=0.5)
    assert torch.isfinite(loss)
    assert (seen["x_video_timesteps"] == 0.5).all() and (seen[f"x_{policy.modality}_timesteps"] == 0.5).all()


def write_blocks(root, episode_index, n_frames, high: dict[int, tuple], random_ring=True):
    """A dsf_flux blocks_k0_ringrandom.npz: block -> ext1 patch (or None for 'high but nothing seen')."""
    blocks_n = (n_frames - 1) // masks.BLOCK_FRAMES
    order = sorted(high)
    union = {c: [] for c in masks.HIDDEN_CAMERAS}
    cells = {c: np.zeros((blocks_n, *masks.LATENT_HW), bool) for c in masks.HIDDEN_CAMERAS}
    for block in order:
        mask = patch(*high[block]) if high[block] is not None else np.zeros(masks.CAMERA_HW, bool)
        union["ext1"].append(np.packbits(mask))
        union["ext2"].append(np.packbits(np.zeros(masks.CAMERA_HW, bool)))
        cells["ext1"][block] = masks.camera_cells(mask, "ext1")
    block_high = np.zeros(blocks_n, bool)
    block_high[order] = True
    folder = root / f"ep_{episode_index}"
    folder.mkdir(parents=True)
    np.savez_compressed(
        folder / "blocks_k0_ringrandom.npz", block_high=block_high, high_index=np.array(order, np.int32),
        **{f"union_{c}": np.stack(union[c]) for c in masks.HIDDEN_CAMERAS},
        **{f"cells_{c}": cells[c] for c in masks.HIDDEN_CAMERAS},
        ring_tokens=np.array(0), random_ring=np.array(random_ring), episode_index=np.array(episode_index),
        k_pixels=np.array(0),
    )


@pytest.fixture(scope="module")
def dsf_index(tmp_path_factory):
    root = tmp_path_factory.mktemp("dsf")
    episodes = [{"n_frames": 121, "caption": f"task {e}", "keep": [[0, 121]]} for e in range(4)]
    closes = {0: 60, 2: 60}  # the commanded gripper closes at this frame; episodes 1 and 3 never close
    for e, frame in closes.items():
        episodes[e]["gripper"] = [0.0] * frame + [1.0] * (121 - frame)
    filter_path = build_compact_dataset(root / "source", episodes)
    manifest = build_manifest(root / "source", filter_path)
    (root / "index").mkdir()
    build_rows(root / "source", manifest, root / "index" / ROWS_FILENAME)
    write_manifest(manifest, root / "index" / "manifest.json")
    write_blocks(root / "masks", 0, 121, {10: (100, 200, 150, 260), 11: (100, 210, 150, 270)})
    write_blocks(root / "masks", 1, 121, {3: (0, 0, 40, 40)})
    write_blocks(root / "masks", 2, 121, {20: None})                    # high, but the robot is out of view
    # episode 3 has no masks
    return root, json.loads((root / "index" / "manifest.json").read_text())


def dataset(root, manifest, episodes, **kwargs):
    return DSFWindowDataset(
        manifest, root / "source", root / "index" / ROWS_FILENAME, masks_root=root / "masks", run="k0_ringrandom",
        episodes=episodes, seed=1, frame_hw=(32, 64), decoder="pyav", **kwargs,
    )


def test_dsf_dataset_draws_noisy_windows_with_their_masks(dsf_index):
    root, manifest = dsf_index
    data = dataset(root, manifest, {0, 1, 2, 3})
    assert sorted(int(e["episode_index"]) for e in data.episodes) == [0]
    assert data.skipped == {
        1: "no gripper close in a hidden window",  # block 3 hidden, but the gripper never closes
        2: "no window with a hidden cell",  # high, but nothing seen
        3: "no masks",
    }
    assert [(data.episodes[p]["episode_index"], hidden) for p, hidden, _ in data.units] == [(0, 0)]
    # blocks 10-11 in the window (starts 12..44) AND the close at frame 60 at action step 8..27 (starts 33..52)
    assert data.units[0][2] == [36, 40, 44]
    for position in range(data.n_positions):
        item = data.window(position)
        unit = data.units[position % len(data.units)]
        assert item["start"] in unit[2] and item["dsf_start"] == item["start"]
        first = item["start"] // 4
        high = item["dsf_block_high"].numpy()
        expected = [first + j in {10, 11} for j in range(8)]
        assert high.tolist() == expected and item["images.left"].shape == (33, 3, 32, 64)
        j = expected.index(True)
        assert masks.unpack_camera_mask(item["dsf_union"][j].numpy()).any()
    again = dataset(root, manifest, {0, 1, 2, 3})
    assert [again.window(p)["start"] for p in range(2)] == [data.window(p)["start"] for p in range(2)]
    only = dataset(root, manifest, {0})
    assert [int(e["episode_index"]) for e in only.episodes] == [0]


def test_close_rule_keeps_windows_closing_the_gripper_at_steps_8_to_27():
    gripper = np.array([0.2] * 40 + [0.9] * 20 + [0.1] * 10 + [0.6] * 10, np.float32)  # closes at 40 and 70
    assert close_steps(gripper).tolist() == [40, 70]
    assert close_steps(np.full(50, 0.9)).tolist() == []  # closed from the start: no close event
    starts = list(range(0, 49, 4))
    # 40: starts 13..32; 70: starts 43..62 (only <= 48 exist here)
    assert close_starts(gripper, starts) == [16, 20, 24, 28, 32, 44, 48]


def test_count_close_windows_on_the_data_files(dsf_index):
    root, _ = dsf_index
    totals = count_close_windows(root / "source", root / "source" / "keep_ranges.json")
    # 4 episodes x 23 starts (0..88 step 4); closes at frame 60 in episodes 0 and 2: starts 36..52 -> 5 each
    assert totals == {
        "episodes": 4, "episodes_with_close_window": 2, "closes": 2, "windows": 92, "close_windows": 10,
    }


def test_split_and_length_check(dsf_index, tmp_path):
    root, manifest = dsf_index
    split = write_split(root / "masks", tmp_path / "split.json", "k0_ringrandom", validation=0.34, seed=1)
    assert sorted(split["train"] + split["validation"]) == [0, 1, 2] and len(split["validation"]) == 1
    wrong = json.loads(json.dumps(manifest))
    wrong["episodes"][0]["n_frames"] = 77
    with pytest.raises(ValueError, match="different Cosmos3-DROID release"):
        dataset(root, wrong, {0})


class DSFWindows(IterableDataset):
    """Synthetic DSF windows at the real camera size for a tiny trainer run (train and validation)."""

    def __init__(self, seed, epoch, rank, world_size, num_workers, windows_per_rank, skip_batches,
                 grad_accumulation, **_):
        self.seed, self.epoch, self.windows_per_rank, self.skip = seed, epoch, windows_per_rank, skip_batches
        self.rank, self.num_workers = rank, num_workers
        self.batches_per_rank, self.n_positions, self.n_windows = 2 * grad_accumulation, 4, 9
        self.episodes = [{}, {}]

    def __iter__(self):
        for b in range(self.skip, self.batches_per_rank):
            batch = dsf_batch(hidden_block=(b + self.epoch) % 8, seed=self.seed * 100 + self.epoch * 10 + b)
            for w in range(self.windows_per_rank):
                yield {k: (v[w % 2] if not isinstance(v, list) else v[w % 2]) for k, v in batch.items()}


def build_windows(config, **kwargs):
    return DSFWindows(**kwargs)


def test_trainer_logs_progress_dsf_metrics_and_validation(tmp_path):
    output = tmp_path / "run"
    config = TrainConfig(
        dataset=f"{__name__}:build_windows", validation_dataset=f"{__name__}:build_windows",
        source_root="unused", index_dir="unused", output_dir=str(output),
        policy=dict(camera_layout="droid", camera_keys=["images.wrist", "images.left", "images.right"],
                    canvas_hw=[544, 736], action_dim=8, gripper_flip_dims=[-1], dit_config=TINY_DIT, augment=False),
        dsf={"note": "synthetic"}, steps=2, cooldown_start=None, frozen_steps=0, trunk_warmup_steps=1,
        heads_warmup_steps=1, windows_per_rank=2, num_workers=1, in_process_loader=True, param_dtype="float32",
        compute_dtype="float32", checkpoint_every=100, log_every=1, ema_sigma_rels=(0.10,),
        validation_every=1, validation_batches=1, validation_timesteps=(0.5,),
        validation_dsf_levels=[[0.15, 0.95]],
    )
    Trainer(config, policy_factory=lambda c: FluxActionPolicy(c, video_vae=FakeVideoVAE(),
                                                              text_encoder=MockTextEncoder(32))).run()
    lines = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    updates = [r for r in lines if "loss" in r and "event" not in r]
    validations = [r for r in lines if r.get("event") == "validation"]
    assert [r["step"] for r in updates] == [1, 2]
    assert updates[-1]["progress"] == 1.0 and updates[0]["windows_seen"] == 2 and updates[0]["dataset_windows"] == 9
    assert all(np.isfinite(updates[0][k]) for k in DSF_METRICS) and updates[0]["dsf_t_low"] <= updates[0]["dsf_t_high"]
    assert [v["step"] for v in validations] == [0, 1, 2]                 # baseline, then every update
    assert {"val_loss", "val_dsf_video_hidden_mse", "val_dsf_action_hidden_mse", "val_loss_t0.5"} <= set(validations[0])
    pair = {f"val_dsf_tl0.15_th0.95_{k}" for k in ("loss", "video_hidden_mse", "action_hidden_mse")}
    assert pair <= set(validations[0]) and all(np.isfinite(validations[0][k]) for k in pair)
    epoch = next(r for r in lines if r.get("event") == "epoch")
    assert epoch["dataset_episodes"] == 2 and epoch["dataset_positions"] == 4


def test_strong_levels_ordinary_windows_and_all_actions_high():
    generator = torch.Generator().manual_seed(0)
    ordinary = torch.zeros(4000, dtype=torch.bool)
    ordinary[:1000] = True
    low, high = noise.draw_levels(4000, "uniform", width=0.75, shift=42.0, generator=generator, strong_fraction=0.7,
                                  ordinary=ordinary)
    assert (low[:1000] == high[:1000]).all() and low[:1000].median() > 0.9      # one level, the policy's own
    dsf_low, dsf_high = low[1000:], high[1000:]
    strong = (dsf_low >= 0.1) & (dsf_low <= 0.2) & (dsf_high >= 0.9) & (dsf_high <= 0.99)
    assert 0.65 < strong.float().mean() < 0.8 and (dsf_low <= dsf_high).all()
    union, block_high = window_union({2: patch(100, 200, 160, 260)})
    _, action = masks.window_token_masks(union, block_high, "ext1", all_actions_high=True)
    assert action.all()                                                          # the joints would be a shortcut
    _, nothing = masks.window_token_masks(*window_union({2: np.zeros(masks.CAMERA_HW, bool)}), "ext1",
                                          all_actions_high=True)
    assert not nothing.any()                                                     # nothing hidden, nothing high


def test_whole_window_and_ordinary_windows(dsf_index, tmp_path):
    root, manifest = dsf_index
    masks_root = tmp_path / "masks"
    every_block = {b: (100, 200 + 4 * b, 150, 260 + 4 * b) for b in range(30)}   # robot masks for every latent frame
    write_blocks(masks_root, 0, 121, every_block)
    write_blocks(masks_root, 1, 121, {10: (100, 200, 150, 260), 11: (100, 210, 150, 270)})  # around an event only
    for e in (2, 3):
        write_blocks(masks_root, e, 121, every_block)
    whole = DSFWindowDataset(manifest, root / "source", root / "index" / ROWS_FILENAME, masks_root=masks_root,
                             run="k0_ringrandom", episodes={0, 1}, seed=1, frame_hw=(32, 64), decoder="pyav",
                             whole_window=True, ordinary_fraction=0.5, visits_per_epoch=40)
    assert [int(e["episode_index"]) for e in whole.episodes] == [0]
    assert whole.skipped == {1: "no window with a hidden cell"}                  # partial masks: no whole window
    assert whole.units[0][2] == [36, 40, 44, 48, 52]                              # the close at frame 60
    kinds = []
    for position in range(whole.n_positions):
        item = whole.window(position)
        kinds.append(item["dsf_ordinary"])
        if item["dsf_ordinary"]:
            assert not item["dsf_block_high"].any() and not item["dsf_union"].any() and not item["dsf_all_actions"]
            assert item["start"] in whole.valid_starts(0)
        else:
            assert item["start"] in whole.units[0][2] and item["dsf_block_high"].all() and item["dsf_all_actions"]
    assert 0.3 < np.mean(kinds) < 0.7
    policy = droid_policy()
    chosen = [kinds.index(True), kinds.index(True, kinds.index(True) + 1), kinds.index(False), kinds.index(False, kinds.index(False) + 1)]
    batch = collate_windows([whole.window(p) for p in chosen])
    prepared = policy.prepare(batch)
    ordinary = torch.tensor([bool(o) for o in batch["dsf_ordinary"]])
    assert prepared.dsf_ordinary.tolist() == ordinary.tolist()
    assert prepared.dsf_action_high.all(1).tolist() == (~ordinary).tolist()
    assert not prepared.dsf_video_high[ordinary].any()
