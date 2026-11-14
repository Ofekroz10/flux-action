"""Look at a DSF window the way training builds it, through FLUX's own video VAE.

    uv run python examples/dsf/visual_check.py --episode 54549 --masks-root <dsf_flux out or pilot folder> \
        --source-root outputs/dsf-check/source --output outputs/dsf-check

1. Fetch the episode's Cosmos3-DROID files (episode table, data file, the 3 camera videos) and index it with the
   trainer's own index code (``build_manifest`` / ``build_rows``).
2. Take a DSF window through the trainer's dataset plug (``DSFWindowDataset``): the frames, and the hidden camera's
   pixel masks of its 8 latent frames.
3. Build the canvas exactly as training does (``packing.materialize_video``: the window's 95 % crop augmentation drawn
   from its seed, then the DROID composite), and the token masks from the same crop (``dsf.masks``).
4. Encode with the video VAE; replace every high-noise latent token by the latent of a magenta canvas; decode.

Output: ``<output>/ep<E>_<camera>_s<start>.mp4`` (5 fps; left: the canvas with the hidden cells outlined in red;
middle: decoded as is; right: decoded with the high-noise tokens magenta) and a ``.json`` report.
``--fake-vae`` runs everything with the test fixture's shape-only VAE (no GPU, no NATTEN): a dry run of the plumbing.
The real VAE needs an NVIDIA GPU (NATTEN) and access to ``black-forest-labs/flux-3-action-base``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from flux_action.data.droid.index import ROWS_FILENAME, build_manifest, build_rows, write_manifest
from flux_action.dsf import masks
from flux_action.dsf.dataset import DSFWindowDataset
from flux_action.processing import packing

COSMOS = "nvidia/Cosmos3-DROID"
VIDEO_VAE = "black-forest-labs/flux-3-action-base:video_vae.safetensors@62878e2925e59b7a89ec14463ce89932624c490d"
CAMERA_FEATURES = ("wrist_image_left", "exterior_image_1_left", "exterior_image_2_left")
MAGENTA = (1.0, -1.0, 1.0)  # RGB in the canvas range [-1, 1]


def fetch_episode(source_root: Path, episode_index: int) -> None:
    """The episode table, the data file and the three video files that hold ``episode_index`` (Hugging Face cache
    is not used: files go to ``source_root/success/...``, the layout the index reads)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    def get(name: str) -> Path:
        return Path(hf_hub_download(COSMOS, f"success/{name}", repo_type="dataset", local_dir=source_root))

    get("meta/info.json")
    get("meta/tasks.parquet")
    for number in range(64):
        name = f"meta/episodes/chunk-000/file-{number:03d}.parquet"
        rows = pq.read_table(get(name), filters=[("episode_index", "=", episode_index)]).to_pylist()
        if rows:
            break
    else:
        raise SystemExit(f"episode {episode_index} is not in Cosmos3-DROID success")
    row = rows[0]
    get(f"data/chunk-{row['data/chunk_index']:03d}/file-{row['data/file_index']:03d}.parquet")
    for camera in CAMERA_FEATURES:
        key = f"videos/observation.image.{camera}"
        get(f"videos/observation.image.{camera}/chunk-{row[f'{key}/chunk_index']:03d}/file-{row[f'{key}/file_index']:03d}.mp4")


def index_episode(source_root: Path, index_dir: Path, filter_path: Path) -> dict:
    manifest = build_manifest(source_root, filter_path, only_available=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    build_rows(source_root, manifest, index_dir / ROWS_FILENAME)
    write_manifest(manifest, index_dir / "manifest.json")
    return json.loads((index_dir / "manifest.json").read_text())


def load_vae(fake: bool, spec: str, device: torch.device):
    if fake:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
        from conftest import FakeVideoVAE

        class FakeWithDecode(FakeVideoVAE):
            def decode(self, z):  # shape-only: each latent frame back to 4 frames of 32x32 blocks
                b, _, t, h, w = z.shape
                rgb = torch.einsum("bcthw,cd->bdthw", z.float(), self.proj.float() * 40).tanh()
                frames = rgb[:, :, :1].repeat(1, 1, 1, 1, 1)
                rest = rgb[:, :, 1:].repeat_interleave(4, dim=2)
                video = torch.cat([frames, rest], dim=2)
                return video.repeat_interleave(32, 3).repeat_interleave(32, 4)

        return FakeWithDecode().to(device)
    from flux_action.models.video_vae import load_video_vae

    return load_video_vae(spec)


def stage(name: str) -> None:
    """A progress line with the process's peak memory (the CPU run is memory-bound)."""
    import resource
    import sys

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**30 if sys.platform == "darwin" else 2**20)
    print(f"[visual_check] {name} (peak memory {peak:.1f} GB)", flush=True)


def to_uint8(video: torch.Tensor) -> np.ndarray:
    """(3, T, H, W) in [-1, 1] -> (T, H, W, 3) uint8."""
    return ((video.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 3, 0).cpu().numpy()


def outline(frames: np.ndarray, video_hidden: np.ndarray) -> np.ndarray:
    """Red 2-pixel outlines of the hidden cells on every frame of their latent frame (frame 0 is the context)."""
    out = frames.copy()
    for j in range(masks.PREDICTED_BLOCKS):
        for row, column in zip(*np.nonzero(video_hidden[j])):
            y0, x0 = row * masks.CELL, column * masks.CELL
            for f in range(1 + masks.BLOCK_FRAMES * j, 1 + masks.BLOCK_FRAMES * (j + 1)):
                cell = out[f, y0 : y0 + masks.CELL, x0 : x0 + masks.CELL]
                cell[:2], cell[-2:], cell[:, :2], cell[:, -2:] = 255 * np.array([1, 0, 0], np.uint8), \
                    255 * np.array([1, 0, 0], np.uint8), 255 * np.array([1, 0, 0], np.uint8), \
                    255 * np.array([1, 0, 0], np.uint8)
    return out


def write_mp4(frames: np.ndarray, path: Path, fps: int = 5) -> None:
    t, h, w, _ = frames.shape
    command = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
               "-i", "-", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "18", str(path)]
    subprocess.run(command, input=frames.tobytes(), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episode", type=int, required=True, help="Cosmos3-DROID episode_index")
    parser.add_argument("--masks-root", type=Path, required=True, help="folder with ep_<E>/blocks_<run>.npz (dsf_flux)")
    parser.add_argument("--run", default="k0_ringrandom")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--filter", type=Path, help="keep_ranges_1_0_1.json (default: downloaded from KarlP/droid)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera", choices=masks.HIDDEN_CAMERAS, help="hidden camera (default: the first with masks)")
    parser.add_argument("--start", type=int, help="window start (default: the noisy start with the most hidden tokens)")
    parser.add_argument("--no-augment", action="store_true", help="skip the crop augmentation (training uses it)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the augmentation draw")
    parser.add_argument("--vae", default=VIDEO_VAE)
    parser.add_argument("--fake-vae", action="store_true", help="the shape-only test VAE (dry run, CPU)")
    arguments = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fetch_episode(arguments.source_root, arguments.episode)
    filter_path = arguments.filter
    if filter_path is None:
        from huggingface_hub import hf_hub_download

        filter_path = Path(hf_hub_download("KarlP/droid", "keep_ranges_1_0_1.json", local_dir=arguments.source_root))
    manifest = index_episode(arguments.source_root, arguments.output / "index", filter_path)
    dataset = DSFWindowDataset(
        manifest, arguments.source_root, arguments.output / "index" / ROWS_FILENAME, masks_root=arguments.masks_root,
        run=arguments.run, episodes={arguments.episode}, seed=arguments.seed, decoder="ffmpeg",
    )
    units = [u for u in dataset.units if arguments.camera is None or masks.HIDDEN_CAMERAS[u[1]] == arguments.camera]
    if not units:
        raise SystemExit(f"episode {arguments.episode} has no noisy window for {arguments.camera or 'any camera'}")
    position, hidden, starts = units[0]
    camera = masks.HIDDEN_CAMERAS[hidden]
    augmentation = None
    if not arguments.no_augment:
        augmentation = packing.sample_augmentation(torch.Generator().manual_seed(arguments.seed))

    def token_masks(start):
        fields = dataset.window_masks(arguments.episode, hidden, start)
        return fields, masks.window_token_masks(
            fields["dsf_union"].numpy(), fields["dsf_block_high"].numpy(), camera, augmentation=augmentation,
            random_ring=bool(fields["dsf_random_ring"]), episode_index=arguments.episode, start=start,
        )

    start = arguments.start
    if start is None:
        start = max(starts, key=lambda s: token_masks(s)[1][0].sum())
    fields, (video_hidden, action_high) = token_masks(start)
    item = dataset.window_at(position, start)
    cams = torch.stack([item[f"images.{key}"] for key in ("wrist", "left", "right")])  # (3, 33, 3, 360, 640)
    canvas = packing.materialize_video(cams, augmentation, device, layout="droid")  # (3, 33, 544, 736)

    stage("canvas built")
    vae = load_vae(arguments.fake_vae, arguments.vae, device)
    stage("VAE loaded")
    # bf16 autocast on the CPU as well: the 45-frame encode in fp32 does not fit a 48 GB laptop
    with torch.autocast(device.type, dtype=torch.bfloat16):
        latents = packing.encode_videos(vae, canvas[None], masks.LATENT_HW)  # (1, 96, 9, 17, 20), as training
        stage("window encoded")
        # the magenta reference: one predicted latent frame of a constant magenta clip (5 frames, no chunk padding)
        magenta = torch.tensor(MAGENTA, device=device).view(1, 3, 1, 1, 1).expand(1, 3, 5, *canvas.shape[-2:])
        magenta_latents = vae.encode_task(magenta.contiguous().to(torch.bfloat16))[..., : masks.LATENT_HW[0], : masks.LATENT_HW[1]]
    stage("magenta encoded")
    hidden_tokens = torch.from_numpy(video_hidden).to(device)  # (8, 17, 20)
    replaced = latents.clone()
    magenta_frame = magenta_latents[:, :, 1:2].to(latents.dtype).expand_as(latents[:, :, 1:])
    replaced[:, :, 1:] = torch.where(hidden_tokens[None, None], magenta_frame, latents[:, :, 1:])
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
        decoded = vae.decode(latents.to(torch.bfloat16))
        stage("decoded")
        decoded_magenta = vae.decode(replaced.to(torch.bfloat16))

    h, w = masks.COMPOSITE_HW
    left = outline(to_uint8(canvas)[:, :h, :w], video_hidden)
    middle = to_uint8(decoded[0])[:, :h, :w]
    right = to_uint8(decoded_magenta[0])[:, :h, :w]
    frames = min(len(left), len(middle), len(right))
    gap = np.full((frames, h, 8, 3), 255, np.uint8)
    side = np.concatenate([left[:frames], gap, middle[:frames], gap, right[:frames]], axis=2)
    arguments.output.mkdir(parents=True, exist_ok=True)
    name = f"ep{arguments.episode}_{camera}_s{start}{'_fake' if arguments.fake_vae else ''}"
    write_mp4(side, arguments.output / f"{name}.mp4")
    report = {
        "episode": arguments.episode, "hidden_camera": camera, "reference_camera": "ext2" if camera == "ext1" else "ext1",
        "start": start, "augmentation": augmentation, "random_ring": bool(fields["dsf_random_ring"]),
        "hidden_tokens_per_latent_frame": video_hidden.reshape(8, -1).sum(1).tolist(),
        "high_action_steps": np.nonzero(action_high)[0].tolist(), "task": item["task"],
        "vae": "fake (shape only)" if arguments.fake_vae else arguments.vae, "video": f"{name}.mp4",
        "layout": "left: canvas + hidden cells (red); middle: VAE decode; right: decode with high-noise tokens magenta",
    }
    (arguments.output / f"{name}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
