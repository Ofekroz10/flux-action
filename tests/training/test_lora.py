"""LoRA on the DiT: which layers get adapters, identity at start, merge, adapter-only checkpoints, resume, export."""

from __future__ import annotations

import copy
import json

import pytest
import torch
from conftest import TINY_DIT, FakeVideoVAE, make_policy, tiny_config
from synthetic_droid import FRAME_HW, build_compact_dataset
from torch.distributed.checkpoint import FileSystemReader

from flux_action.data.droid.index import ROWS_FILENAME, build_manifest, build_rows, write_manifest
from flux_action.models.text_encoder import MockTextEncoder
from flux_action.policy import FluxActionPolicy
from flux_action.training.checkpoint import export_policy
from flux_action.training.lora import LoRALinear, apply_lora, lora_settings, merge_lora
from flux_action.training.trainer import TrainConfig, Trainer

EPISODES = [
    {"n_frames": 40, "caption": "pick up the cup", "keep": [[0, 40]]},
    {"n_frames": 44, "caption": "open the box", "keep": [[0, 44]]},
]
POLICY = dict(
    camera_layout="single",
    camera_keys=["images.wrist"],
    canvas_hw=[64, 96],
    action_dim=8,
    dit_config=TINY_DIT,
    augment=False,
)


def test_adapters_only_on_the_shared_blocks_and_the_video_mode_blocks():
    dit = make_policy(tiny_config(action_dim=8)).dit
    streams = sorted(dit.content_mode_blocks)
    assert {"video", "video_cond"} < set(streams) and len(
        streams
    ) > 2  # the action streams have mode blocks too
    wrapped = apply_lora(dit, rank=4, alpha=8)
    blocks = {name.rsplit(".", 1)[0] for name in wrapped}
    assert blocks == {"single_blocks.0", "content_mode_blocks.video.0", "content_mode_blocks.video_cond.0"}
    assert {name.rsplit(".", 1)[1] for name in wrapped} == {
        "q_proj",
        "k_proj",
        "v_proj",
        "attn_out",
        "mlp_in",
        "mlp_out",
    }
    trainable = [name for name, p in dit.named_parameters() if p.requires_grad]
    assert trainable and all(".lora_A" in name or ".lora_B" in name for name in trainable)
    assert {p.dtype for name, p in dit.named_parameters() if p.requires_grad} == {torch.float32}
    # frozen: the action streams' mode blocks, their heads, the text path, the modulations, time_in
    for stream in streams:
        if stream not in ("video", "video_cond"):
            assert not any(isinstance(m, LoRALinear) for m in dit.content_mode_blocks[stream].modules())
    assert not any(isinstance(m, LoRALinear) for m in dit.txt_mode_blocks.modules())
    for frozen in (dit.emb_in, dit.final_layer, dit.single_stream_modulations, dit.time_in, dit.txt_in):
        assert not any(p.requires_grad for p in frozen.parameters())


def test_lora_starts_as_the_base_and_merges_exactly():
    torch.manual_seed(0)
    base = torch.nn.Linear(16, 24)
    layer = LoRALinear(copy.deepcopy(base), rank=4, alpha=8)
    x = torch.randn(3, 16)
    torch.testing.assert_close(layer(x), base(x), rtol=0, atol=0)  # B = 0
    with torch.no_grad():
        layer.lora_B.normal_()
    merged = layer.merged()
    assert isinstance(merged, torch.nn.Linear)
    torch.testing.assert_close(merged(x), layer(x), rtol=1e-5, atol=1e-5)
    expected = base.weight + 2.0 * layer.lora_B @ layer.lora_A
    torch.testing.assert_close(merged.weight, expected.detach(), rtol=1e-6, atol=1e-6)


def test_settings_defaults_and_errors():
    assert lora_settings({"rank": 512}) == {
        "rank": 512,
        "alpha": 512.0,
        "mode_streams": ["video", "video_cond"],
    }
    assert lora_settings(None) is None
    with pytest.raises(ValueError, match="unknown lora settings"):
        lora_settings({"rank": 4, "targets": "all"})


@pytest.fixture(scope="module")
def indexed(tmp_path_factory):
    root = tmp_path_factory.mktemp("lora")
    filter_path = build_compact_dataset(root / "source", EPISODES)
    manifest = build_manifest(root / "source", filter_path)
    (root / "index").mkdir()
    build_rows(root / "source", manifest, root / "index" / ROWS_FILENAME)
    write_manifest(manifest, root / "index" / "manifest.json")
    # the "released policy" to fine-tune: a tiny policy export, trunk_weights of the LoRA runs
    policy = FluxActionPolicy(
        tiny_config(**{k: v for k, v in POLICY.items() if k != "camera_keys"}, camera_keys=("images.wrist",)),
        video_vae=FakeVideoVAE(),
        text_encoder=MockTextEncoder(32),
    )
    policy.save_pretrained(root / "trunk")
    return root


def lora_config(root, output, **overrides):
    values = dict(
        source_root=str(root / "source"),
        index_dir=str(root / "index"),
        output_dir=str(output),
        policy={**POLICY, "trunk_weights": str(root / "trunk" / "model.safetensors"), "optimizer_lr": 1e-2},
        lora={"rank": 4, "alpha": 8},
        steps=4,
        cooldown_start=None,
        frozen_steps=0,
        trunk_warmup_steps=1,
        heads_warmup_steps=1,
        windows_per_rank=2,
        num_workers=1,
        in_process_loader=True,
        decoder="pyav",
        frame_hw=FRAME_HW,
        param_dtype="float32",
        compute_dtype="float32",
        activation_checkpointing=False,
        checkpoint_every=2,
        log_every=1,
        ema_sigma_rels=(0.10,),
        reseed_on_resume=False,
    )
    values.update(overrides)
    return TrainConfig(**values)


def tiny_policy(config):
    return FluxActionPolicy(config, video_vae=FakeVideoVAE(), text_encoder=MockTextEncoder(32))


def updates(output):
    lines = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    return [(r["step"], r["loss"]) for r in lines if "loss" in r and "event" not in r], lines


def test_lora_run_saves_adapters_only_resumes_exactly_and_exports_merged(indexed, tmp_path):
    full = Trainer(lora_config(indexed, tmp_path / "full"), policy_factory=tiny_policy)
    full.run()
    records, lines = updates(tmp_path / "full")
    assert [step for step, _ in records] == [1, 2, 3, 4]
    lora_event = next(r for r in lines if r.get("event") == "lora")
    assert lora_event["layers"] == 18 and lora_event["rank"] == 4  # 3 blocks x 6 linear layers
    frozen_event = next(r for r in lines if r.get("event") == "frozen")
    assert 0 < frozen_event["trainable_numel"] < frozen_event["frozen_numel"]
    assert len(full.optimizer.param_groups) == 1  # the heads are frozen: one group, the adapters
    saved = FileSystemReader(str(tmp_path / "full" / "step-4" / "model")).read_metadata().state_dict_metadata
    assert saved and all(".lora_" in key for key in saved)

    # resume from step 2 continues exactly like the uninterrupted run
    Trainer(lora_config(indexed, tmp_path / "part", steps=2), policy_factory=tiny_policy).run()
    Trainer(lora_config(indexed, tmp_path / "part"), policy_factory=tiny_policy).run()
    assert updates(tmp_path / "part")[0] == records

    # the export is the trunk with the trained adapters merged in: a plain policy
    export = export_policy(tmp_path / "full" / "step-4", tmp_path / "export", profile="model")
    assert export["lora_layers_merged"] == 18
    restored = FluxActionPolicy.from_pretrained(
        tmp_path / "export", video_vae=FakeVideoVAE(), text_encoder=MockTextEncoder(32)
    )
    reference = copy.deepcopy(full.policy.dit)
    merge_lora(reference)
    expected = reference.state_dict()
    exported = restored.dit.state_dict()
    assert set(exported) == set(expected)
    for name, tensor in exported.items():
        torch.testing.assert_close(tensor, expected[name], rtol=1e-6, atol=1e-6)
    trunk = FluxActionPolicy.from_pretrained(
        indexed / "trunk", video_vae=FakeVideoVAE(), text_encoder=MockTextEncoder(32)
    ).dit.state_dict()
    changed = {name for name in exported if not torch.equal(exported[name], trunk[name])}
    assert changed and all(
        name.startswith(("single_blocks.", "content_mode_blocks.video.", "content_mode_blocks.video_cond."))
        for name in changed
    )
    ema = export_policy(tmp_path / "full" / "step-4", tmp_path / "export_ema", profile="ema_0p10")
    assert ema["lora_layers_merged"] == 18
