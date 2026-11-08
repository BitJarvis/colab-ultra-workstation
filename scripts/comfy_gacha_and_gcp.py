#!/usr/bin/env python3
"""Colab ↔ GCP (GCS Bucket) + ComfyUI Batch Video Gacha ("抽卡") Engine for G4 (96GB Blackwell).

Core Capabilities:
1. **Batch Multi-Seed Video Gacha ("抽卡模式")**:
   - Submits N seed variations (e.g., `--gacha-runs 4` or `--seeds 1001,1002,1003,1004`) to a hot-VRAM
     ComfyUI instance (`127.0.0.1:8188`) with **0.0s model reload** between takes.
   - Automatically locates all seed nodes (`RandomNoise.noise_seed`, `KSampler.seed`, `KSamplerAdvanced.noise_seed`)
     and output nodes (`SaveVideo`, `VHS_VideoCombine`, `SaveAnimatedWEBP`, `SaveImage`, `H3SaveAVLatent`) in ANY
     ComfyUI API workflow JSON (MiniMax-H3, Wan2.1, HunyuanVideo, LTX-Video, AnimateDiff, Flux, etc.).
   - Automatically renders a **2×2 (or N-way) Side-by-Side Comparison Grid Video (`*_Gacha_Grid.mp4`)** with
     per-tile `Take #k | seed=...` watermarks so you can visually pick the winning card ("抽卡选片") in one glance!

2. **Colab ↔ GCP (Google Cloud Storage) High-Speed Bridge**:
   - Connects Colab GCE VM (which runs inside Google's internal network at `1–2 GB/s` to GCS) with the user's
     GCP project & GCS bucket (`gs://<bucket>/...`), leveraging Google AI Ultra GCP credits.
   - Supports:
     - `gcs-auth`: Interactive or service-account GCP authentication inside Colab (`google.colab.auth` or `gcloud`).
     - `gcs-pull`: Parallel multi-threaded pull (`gcloud storage cp -r` / `gsutil -m cp -r`) of large checkpoints
       from `gs://bucket/...` directly onto Colab local NVMe (`/content/fast_models` or `/content/ComfyUI/models`).
     - `gcs-push`: Automatic backup of generated videos, gacha grids, and 5MB `AV_LATENT` checkpoints to `gs://bucket/...`.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_API_URL = os.environ.get("COMFY_API_URL", "http://127.0.0.1:8188")
SEED_INPUT_KEYS = ("noise_seed", "seed", "rand_seed", "s_seed")
OUTPUT_PREFIX_KEYS = ("filename_prefix",)
LATENT_CKPT_KEYS = ("checkpoint_name",)
SEG_SUFFIX_RE = re.compile(r"^(.*?)(_Seg\d+.*)$")

GCS_URI_RE = re.compile(r"^gs://[a-z0-9][a-z0-9._-]{1,61}[a-z0-9](/[^\s]*)?$")


def _format_gacha_prefix(orig_prefix: str, tag: str, base_prefix: Optional[str] = None) -> str:
    """Format an output `filename_prefix` or `checkpoint_name` stem with `tag` while preserving subfolder and `_SegXX` suffix ordering.

    Example:
      - `DouyinDance_15s_V1_Seg01` + `GachaT01_S1001` -> `DouyinDance_15s_V1_GachaT01_S1001_Seg01`
        (so `stitch_available_segments(f"{prefix}_{tag}", ...)` finds `{prefix}_{tag}_Seg01*.mp4`!)
      - `ComfyUI` + `GachaT01_S1001` (with `base_prefix="WanGacha"`) -> `WanGacha_ComfyUI_GachaT01_S1001`
      - `video/MiniMax_H3` + `GachaT01_S1001` (with `base_prefix="WebUI_abc"`) -> `video/WebUI_abc_MiniMax_H3_GachaT01_S1001`
    """
    norm = str(orig_prefix or "ComfyUI").replace("\\", "/")
    if "/" in norm:
        parent_dir, leaf = norm.rsplit("/", 1)
    else:
        parent_dir, leaf = "", norm
    if base_prefix and not leaf.startswith(base_prefix):
        leaf = f"{base_prefix}_{leaf}"
    m = SEG_SUFFIX_RE.match(leaf)
    if m:
        formatted_leaf = f"{m.group(1)}_{tag}{m.group(2)}"
    else:
        formatted_leaf = f"{leaf}_{tag}"
    return f"{parent_dir}/{formatted_leaf}" if parent_dir else formatted_leaf


def validate_gcs_uri(uri: str) -> str:
    """Validate and normalize a `gs://bucket[/path]` URI to prevent command injection or malformed paths."""
    cleaned = (uri or "").strip()
    if not GCS_URI_RE.match(cleaned):
        raise ValueError(
            f"Invalid GCS URI '{uri}'. Expected format: 'gs://your-bucket-name' or 'gs://your-bucket-name/prefix'."
        )
    return cleaned.rstrip("/")


def parse_seed_list(
    seeds_str: Optional[str] = None,
    seed_base: int = 2026110801,
    gacha_runs: int = 4,
    seed_step: int = 1,
) -> List[int]:
    """Expand either an explicit comma-separated `--seeds` string or `(seed_base, gacha_runs, seed_step)` into a list of int seeds."""
    if seeds_str and seeds_str.strip():
        parsed: List[int] = []
        for part in seeds_str.split(","):
            part = part.strip()
            if not part:
                continue
            val = int(part)
            if val < 0:
                raise ValueError(f"Seed must be non-negative, got {val}")
            parsed.append(val)
        if not parsed:
            raise ValueError("No valid seeds found in --seeds")
        return parsed

    if gacha_runs < 1 or gacha_runs > 64:
        raise ValueError(f"gacha_runs must be between 1 and 64, got {gacha_runs}")
    if seed_base < 0:
        raise ValueError(f"seed_base must be non-negative, got {seed_base}")
    return [int(seed_base + i * seed_step) for i in range(gacha_runs)]


def ensure_api_workflow_format(workflow: Dict[str, Any], source_hint: str = "workflow.json") -> None:
    """Ensure `workflow` is a ComfyUI API prompt dict (`{node_id: {class_type, inputs}}`) rather than UI-export format."""
    if not isinstance(workflow, dict) or not workflow:
        raise ValueError(f"Invalid empty or non-dict ComfyUI workflow in {source_hint}.")
    has_api_nodes = any(isinstance(v, dict) and "class_type" in v for v in workflow.values())
    if not has_api_nodes:
        if "nodes" in workflow and "links" in workflow:
            raise ValueError(
                f"Workflow '{source_hint}' is in ComfyUI Web UI format (contains 'nodes'/'links' arrays) rather than API format. "
                "Please export using 'Save (API Format)' in ComfyUI Dev Mode or use one of the API workflow JSONs."
            )
        raise ValueError(f"Workflow '{source_hint}' does not contain any valid ComfyUI API nodes ('class_type').")


def mutate_workflow_for_gacha_take(
    workflow: Dict[str, Any],
    take_idx: int,
    take_seed: int,
    base_prefix: Optional[str] = None,
    target_seg_id: Optional[int] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Return a deep-copied workflow mutated with `take_seed` and take-specific output prefixes.

    If `target_seg_id` is specified (e.g., Segment 1 uses nodes 20..39, Segment 2 uses 40..59),
    only seed nodes belonging to that segment are shifted by `take_seed`, or if `target_seg_id` is None,
    all seed nodes in the workflow are deterministically offset by `(take_seed + node_offset)`.
    Returns `(mutated_wf, expected_filename_prefixes)`.
    """
    ensure_api_workflow_format(workflow)
    wf = copy.deepcopy(workflow)
    tag = f"GachaT{take_idx:02d}_S{take_seed}"
    expected_prefixes: List[str] = []

    target_noise_nid = str(target_seg_id * 20 + 5) if target_seg_id is not None else None

    # 1. Update seed inputs across all sampler / noise nodes
    seed_nodes_updated = 0
    for nid in sorted(wf.keys(), key=lambda x: int(x) if str(x).isdigit() else 999999):
        node = wf[nid]
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            continue

        if target_noise_nid is not None and str(nid) != target_noise_nid:
            continue

        for sk in SEED_INPUT_KEYS:
            if sk in inputs and isinstance(inputs[sk], (int, float)) and not isinstance(inputs[sk], bool):
                # Keep multi-segment relative offsets distinct while anchoring on take_seed
                inputs[sk] = int(take_seed) + seed_nodes_updated
                seed_nodes_updated += 1

    # Fallback: if target_noise_nid wasn't found in a generic workflow, update all seed nodes
    if seed_nodes_updated == 0 and target_noise_nid is not None:
        for nid in sorted(wf.keys(), key=lambda x: int(x) if str(x).isdigit() else 999999):
            node = wf[nid]
            inputs = node.get("inputs", {}) if isinstance(node, dict) else {}
            for sk in SEED_INPUT_KEYS:
                if sk in inputs and isinstance(inputs[sk], (int, float)) and not isinstance(inputs[sk], bool):
                    inputs[sk] = int(take_seed) + seed_nodes_updated
                    seed_nodes_updated += 1

    # 2. Rewrite filename_prefix on SaveVideo / VHS_VideoCombine / SaveImage nodes and H3SaveAVLatent checkpoints
    for nid in sorted(wf.keys(), key=lambda x: int(x) if str(x).isdigit() else 999999):
        node = wf[nid]
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            continue

        for pk in OUTPUT_PREFIX_KEYS:
            if pk in inputs and isinstance(inputs[pk], str):
                new_prefix = _format_gacha_prefix(inputs[pk], tag, base_prefix=base_prefix)
                inputs[pk] = new_prefix
                expected_prefixes.append(new_prefix)

        if node.get("class_type") == "H3SaveAVLatent":
            for ck in LATENT_CKPT_KEYS:
                if ck in inputs and isinstance(inputs[ck], str):
                    stem = Path(inputs[ck]).stem
                    suffix = Path(inputs[ck]).suffix or ".pt"
                    if stem.endswith("_av_latent"):
                        base_stem = stem[: -len("_av_latent")]
                        new_stem = f"{_format_gacha_prefix(base_stem, tag, base_prefix=base_prefix)}_av_latent"
                    else:
                        new_stem = _format_gacha_prefix(stem, tag, base_prefix=base_prefix)
                    inputs[ck] = f"{new_stem}{suffix}"

    return wf, expected_prefixes


def _ffmpeg_has_drawtext() -> bool:
    """Return True if the installed `ffmpeg` binary supports the `drawtext` filter."""
    if not shutil.which("ffmpeg"):
        return False
    try:
        out = subprocess.check_output(["ffmpeg", "-hide_banner", "-filters"], stderr=subprocess.DEVNULL, text=True)
        return " drawtext " in out
    except Exception:
        return False


def build_ffmpeg_grid_command(
    video_paths: List[Path],
    output_grid_path: Path,
    seeds: Optional[List[int]] = None,
    tile_w: int = 540,
    tile_h: int = 960,
    enable_watermark: Optional[bool] = None,
) -> List[str]:
    """Build an FFmpeg command that stitches 2..9 gacha take videos into a labeled comparison grid (`xstack`)."""
    n = len(video_paths)
    if n < 2:
        raise ValueError("At least 2 videos are required to build a comparison grid.")

    cols = 2 if n <= 4 else 3
    rows = int(math.ceil(n / float(cols)))
    use_drawtext = _ffmpeg_has_drawtext() if enable_watermark is None else bool(enable_watermark)

    cmd: List[str] = ["ffmpeg", "-y"]
    for vp in video_paths:
        cmd.extend(["-i", str(vp)])

    filter_parts: List[str] = []
    layout_coords: List[str] = []

    for idx in range(n):
        r = idx // cols
        c = idx % cols
        seed_lbl = f"seed={seeds[idx]}" if (seeds and idx < len(seeds)) else f"take={idx + 1}"
        drawtext_chain = ""
        if use_drawtext:
            safe_txt = f"Take #{idx + 1}  {seed_lbl}".replace(":", "\\:").replace("'", "")
            drawtext_chain = (
                f",drawtext=text='{safe_txt}':x=16:y=16:fontsize=24:fontcolor=white:"
                f"box=1:boxcolor=black@0.55:boxborderw=6"
            )
        # Scale & pad each input to identical tile dimensions (+ optional watermark)
        filter_parts.append(
            f"[{idx}:v]scale={tile_w}:{tile_h}:force_original_aspect_ratio=decrease,"
            f"pad={tile_w}:{tile_h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1{drawtext_chain}[v{idx}]"
        )
        x_expr = "0" if c == 0 else "+".join([f"w{k}" for k in range(c)])
        y_expr = "0" if r == 0 else "+".join([f"h{k * cols}" for k in range(r)])
        layout_coords.append(f"{x_expr}_{y_expr}")

    # Pad missing slots if n is not a multiple of cols (e.g. 3 videos in 2x2 grid)
    total_slots = rows * cols
    for idx in range(n, total_slots):
        r = idx // cols
        c = idx % cols
        filter_parts.append(f"color=c=black:s={tile_w}x{tile_h}:r=24:d=5[v{idx}]")
        x_expr = "0" if c == 0 else "+".join([f"w{k}" for k in range(c)])
        y_expr = "0" if r == 0 else "+".join([f"h{k * cols}" for k in range(r)])
        layout_coords.append(f"{x_expr}_{y_expr}")

    inputs_concat = "".join(f"[v{i}]" for i in range(total_slots))
    layout_str = "|".join(layout_coords)
    filter_parts.append(f"{inputs_concat}xstack=inputs={total_slots}:layout={layout_str}:fill=black[vout]")

    cmd.extend(
        [
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "[vout]",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-crf",
            "17",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(output_grid_path),
        ]
    )
    return cmd


def render_gacha_comparison_grid(
    video_paths: List[Path],
    output_grid_path: Path,
    seeds: Optional[List[int]] = None,
    tile_w: int = 540,
    tile_h: int = 960,
) -> Optional[Path]:
    """Execute FFmpeg to create the multi-seed comparison grid MP4 and write a companion JSON manifest."""
    existing = [p for p in video_paths if p and p.exists()]
    if len(existing) < 2:
        print(f"ℹ️ 有效抽卡视频数量 ({len(existing)}) < 2，跳过四宫格/多宫格对比视频合成。", flush=True)
        return None

    output_grid_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_ffmpeg_grid_command(existing, output_grid_path, seeds=seeds, tile_w=tile_w, tile_h=tile_h)
    subprocess.run(cmd, check=True)

    manifest_path = output_grid_path.with_suffix(".json")
    manifest = {
        "grid_video": str(output_grid_path),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "takes": [
            {
                "take_index": idx + 1,
                "seed": seeds[idx] if seeds and idx < len(seeds) else None,
                "video_path": str(vp),
            }
            for idx, vp in enumerate(existing)
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"🎰 抽卡同屏对比视频已生成: {output_grid_path} (清单: {manifest_path.name})", flush=True)
    return output_grid_path


def ensure_gcp_authenticated(service_account_json: Optional[str] = None, project_id: Optional[str] = None) -> bool:
    """Authenticate GCP / GCS inside Colab or local environment.

    Order of precedence:
    1. Service Account JSON key (`--gcp-sa-json` or `GOOGLE_APPLICATION_CREDENTIALS`).
    2. Colab native `google.colab.auth.authenticate_user()` (when running inside a Colab kernel).
    3. Existing `gcloud auth list` active account.
    """
    sa_path = service_account_json or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if sa_path and Path(sa_path).exists():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(Path(sa_path).resolve())
        if shutil.which("gcloud"):
            subprocess.run(
                ["gcloud", "auth", "activate-service-account", f"--key-file={sa_path}", "--quiet"],
                check=False,
            )
            if project_id:
                subprocess.run(["gcloud", "config", "set", "project", project_id, "--quiet"], check=False)
        print(f"🔐 已通过 Service Account 凭证激活 GCP: {sa_path}", flush=True)
        return True

    # Try Colab native auth if running in Colab IPython kernel
    try:
        from google.colab import auth as colab_auth  # type: ignore

        colab_auth.authenticate_user()
        if project_id and shutil.which("gcloud"):
            subprocess.run(["gcloud", "config", "set", "project", project_id, "--quiet"], check=False)
        print("🔐 已通过 google.colab.auth 完成 GCP 账号授权！", flush=True)
        return True
    except Exception:
        pass

    if shutil.which("gcloud"):
        try:
            out = subprocess.check_output(
                ["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            if out:
                if project_id:
                    subprocess.run(["gcloud", "config", "set", "project", project_id, "--quiet"], check=False)
                print(f"🔐 检测到已登录的 gcloud 账号: {out.splitlines()[0]}", flush=True)
                return True
        except Exception:
            pass

    print(
        "⚠️ 未检测到已激活的 GCP 凭证。若在 Colab 中运行，请先执行 `from google.colab import auth; auth.authenticate_user()` "
        "或传入 `--gcp-sa-json /path/to/service_account.json`。",
        flush=True,
    )
    return False


def gcs_sync_directory_or_files(
    sources: List[Path],
    gcs_uri: str,
    direction: str = "push",
    service_account_json: Optional[str] = None,
    project_id: Optional[str] = None,
) -> None:
    """High-speed parallel transfer between Colab/local filesystem and a GCS bucket (`gs://...`)."""
    norm_uri = validate_gcs_uri(gcs_uri)
    ensure_gcp_authenticated(service_account_json=service_account_json, project_id=project_id)

    has_gcloud = shutil.which("gcloud") is not None
    has_gsutil = shutil.which("gsutil") is not None

    if direction == "push":
        valid_srcs = [str(p) for p in sources if p and Path(p).exists()]
        if not valid_srcs:
            print("ℹ️ 没有待上传至 GCS 的本地文件。", flush=True)
            return
        print(f"☁️ 正在高速同步 {len(valid_srcs)} 个产出文件至 GCP Cloud Storage: {norm_uri}/ ...", flush=True)
        if has_gcloud:
            cmd = ["gcloud", "storage", "cp", "-r", *valid_srcs, f"{norm_uri}/"]
            subprocess.run(cmd, check=True)
        elif has_gsutil:
            cmd = ["gsutil", "-m", "cp", "-r", *valid_srcs, f"{norm_uri}/"]
            subprocess.run(cmd, check=True)
        else:
            _python_gcs_upload_fallback(valid_srcs, norm_uri, project_id=project_id)
        print(f"✅ GCS 云端归档完成 -> {norm_uri}/", flush=True)

    elif direction == "pull":
        if not sources:
            raise ValueError("Pull direction requires a target local directory in `sources[0]`.")
        dst_dir = Path(sources[0])
        dst_dir.mkdir(parents=True, exist_ok=True)
        print(f"⬇️ 正在从 GCP Cloud Storage ({norm_uri}) 高速拉取至本地 NVMe ({dst_dir})...", flush=True)
        if has_gcloud:
            cmd = ["gcloud", "storage", "cp", "-r", f"{norm_uri}/*", str(dst_dir)]
            subprocess.run(cmd, check=True)
        elif has_gsutil:
            cmd = ["gsutil", "-m", "cp", "-r", f"{norm_uri}/*", str(dst_dir)]
            subprocess.run(cmd, check=True)
        else:
            raise RuntimeError("Neither `gcloud` nor `gsutil` found on PATH for GCS pull.")
        print(f"✅ GCS 模型/素材拉取完成 -> {dst_dir}", flush=True)
    else:
        raise ValueError(f"Unknown direction: {direction}")


def _python_gcs_upload_fallback(local_paths: List[str], gcs_uri: str, project_id: Optional[str] = None) -> None:
    """Fallback uploader using `google-cloud-storage` Python SDK if `gcloud`/`gsutil` CLI is unavailable."""
    try:
        from google.cloud import storage  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "Neither `gcloud`/`gsutil` CLI nor `google-cloud-storage` Python package is installed. "
            "Install Google Cloud SDK (`brew install --cask google-cloud-sdk`) or `pip install google-cloud-storage`."
        ) from e

    without_scheme = gcs_uri[len("gs://") :]
    parts = without_scheme.split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1].strip("/") if len(parts) > 1 else ""

    client = storage.Client(project=project_id) if project_id else storage.Client()
    bucket = client.bucket(bucket_name)
    for lp in local_paths:
        p = Path(lp)
        if p.is_file():
            blob_name = f"{prefix}/{p.name}".lstrip("/")
            bucket.blob(blob_name).upload_from_filename(str(p))


def find_latest_output_video(prefix: str, search_dirs: List[Path]) -> Optional[Path]:
    """Find the most recently modified `.mp4` or `.webp` or `.png` matching `prefix` across ComfyUI output dirs."""
    candidates: List[Path] = []
    leaf_prefix = Path(str(prefix).replace("\\", "/")).name
    for d in search_dirs:
        if d.exists():
            for ext in ("*.mp4", "*.webp", "*.gif", "*.png"):
                candidates.extend([p for p in d.rglob(f"{leaf_prefix}*{ext}") if "_hist_" not in p.name])
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def submit_comfy_prompt_and_wait(workflow: Dict[str, Any], label: str, api_url: str = DEFAULT_API_URL) -> Dict[str, Any]:
    """Submit a ComfyUI API workflow JSON and poll `/history/<prompt_id>` until completion."""
    payload = json.dumps({"prompt": workflow}).encode("utf-8")
    req = urllib.request.Request(
        f"{api_url}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=15).read().decode())
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"❌ 提交 ComfyUI 工作流失败 ({label}): HTTP {e.code} -> {err_body}") from e

    if resp.get("node_errors"):
        raise RuntimeError(f"❌ 节点参数校验失败 ({label}): {json.dumps(resp['node_errors'], ensure_ascii=False)}")

    prompt_id = resp["prompt_id"]
    print(f"🎰 [{label}] 已入队 ComfyUI | prompt_id={prompt_id}", flush=True)
    t0 = time.time()
    while True:
        time.sleep(3)
        hist_raw = urllib.request.urlopen(f"{api_url}/history/{prompt_id}", timeout=10).read().decode()
        hist = json.loads(hist_raw)
        if prompt_id in hist:
            entry = hist[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"❌ ComfyUI 渲染报错 ({label}): {json.dumps(status, ensure_ascii=False)}")
            print(f"✅ [{label}] 抽卡渲染完成！耗时: {time.time() - t0:.1f}s", flush=True)
            return entry


def run_batch_gacha_workflow(
    workflow_path: Path,
    seeds: List[int],
    base_prefix: str = "Gacha",
    target_seg_id: Optional[int] = None,
    api_url: str = DEFAULT_API_URL,
    output_dir: Optional[Path] = None,
    gcs_bucket: Optional[str] = None,
    gcp_project: Optional[str] = None,
    gcp_sa_json: Optional[str] = None,
) -> Dict[str, Any]:
    """Run N multi-seed gacha takes on a generic ComfyUI API workflow JSON, build a 2x2 grid, and optionally sync to GCS."""
    raw_wf = json.loads(workflow_path.read_text(encoding="utf-8"))
    search_dirs = [
        output_dir or Path("/content/ComfyUI/output"),
        Path("/content/drive/MyDrive/AI_Workstation/ComfyUI/output"),
        Path("/tmp"),
    ]
    out_root = output_dir or (
        Path("/content/drive/MyDrive/AI_Workstation/ComfyUI/output")
        if Path("/content/drive/MyDrive").exists()
        else Path("/tmp")
    )
    out_root.mkdir(parents=True, exist_ok=True)

    take_videos: List[Path] = []
    last_outputs: Dict[str, Any] = {}
    for idx, seed in enumerate(seeds, start=1):
        mut_wf, prefixes = mutate_workflow_for_gacha_take(
            raw_wf,
            take_idx=idx,
            take_seed=seed,
            base_prefix=base_prefix,
            target_seg_id=target_seg_id,
        )
        label = f"Gacha Take #{idx}/{len(seeds)} (seed={seed})"
        hist_entry = submit_comfy_prompt_and_wait(mut_wf, label=label, api_url=api_url)
        if isinstance(hist_entry, dict) and isinstance(hist_entry.get("outputs"), dict):
            last_outputs = hist_entry["outputs"]
        if prefixes:
            vid = find_latest_output_video(prefixes[-1], search_dirs)
            if vid:
                dst_vid = out_root / vid.name
                try:
                    if not vid.resolve().is_relative_to(out_root.resolve()):
                        shutil.copy2(vid, dst_vid)
                    else:
                        dst_vid = vid
                except Exception:
                    if vid.resolve() != dst_vid.resolve():
                        shutil.copy2(vid, dst_vid)
                take_videos.append(dst_vid)
                print(f"  🎞️ Take #{idx} (seed={seed}) -> {dst_vid}", flush=True)

    grid_mp4: Optional[Path] = None
    if len(take_videos) >= 2 and all(p.suffix.lower() == ".mp4" for p in take_videos):
        grid_mp4 = out_root / f"{base_prefix}_Gacha_{len(take_videos)}way_Grid.mp4"
        render_gacha_comparison_grid(take_videos, grid_mp4, seeds=seeds)

    if gcs_bucket:
        to_sync = list(take_videos)
        if grid_mp4 and grid_mp4.exists():
            to_sync.append(grid_mp4)
            manifest_p = grid_mp4.with_suffix(".json")
            if manifest_p.exists():
                to_sync.append(manifest_p)
        gcs_sync_directory_or_files(
            to_sync,
            gcs_uri=gcs_bucket,
            direction="push",
            service_account_json=gcp_sa_json,
            project_id=gcp_project,
        )

    return {
        "seeds": seeds,
        "take_videos": [str(p) for p in take_videos],
        "grid_video": str(grid_mp4) if grid_mp4 else None,
        "outputs": last_outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Colab ↔ GCP (GCS) + ComfyUI Multi-Seed Video Gacha ('抽卡') Runner"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # 1. `gacha` subcommand: run multi-seed batch generation on any ComfyUI API workflow JSON
    p_gacha = sub.add_parser("gacha", help="Run N-seed batch gacha on any ComfyUI API workflow JSON + render 2x2 grid")
    p_gacha.add_argument("--workflow", required=True, help="Path to ComfyUI API format workflow.json")
    p_gacha.add_argument("--gacha-runs", type=int, default=4, help="Number of gacha takes to generate (default: 4 for 2x2 grid)")
    p_gacha.add_argument("--seed-base", type=int, default=2026110801, help="Starting random seed")
    p_gacha.add_argument("--seeds", default=None, help="Explicit comma-separated seed list, e.g. '101,202,303,404'")
    p_gacha.add_argument("--prefix", default="ComfyGacha", help="Output filename prefix")
    p_gacha.add_argument("--target-seg", type=int, default=None, help="Optional MiniMax-H3 segment ID to target")
    p_gacha.add_argument("--api-url", default=DEFAULT_API_URL, help="ComfyUI HTTP API URL (default: http://127.0.0.1:8188)")
    p_gacha.add_argument("--output-dir", default=None, help="Directory to store gacha takes and 2x2 comparison grid")
    p_gacha.add_argument("--gcs-bucket", default=None, help="Optional GCS URI (e.g. gs://my-bucket/comfy-outputs) to auto-sync results")
    p_gacha.add_argument("--gcp-project", default=None, help="Optional GCP project ID")
    p_gacha.add_argument("--gcp-sa-json", default=None, help="Optional path to GCP Service Account JSON key")

    # 2. `grid` subcommand: stitch existing MP4 takes into a 2x2 / N-way comparison grid
    p_grid = sub.add_parser("grid", help="Combine existing MP4 takes into a 2x2 / N-way comparison grid")
    p_grid.add_argument("--videos", nargs="+", required=True, help="Paths to MP4 videos")
    p_grid.add_argument("--out", required=True, help="Output grid MP4 path")
    p_grid.add_argument("--seeds", default=None, help="Optional comma-separated seeds matching the videos")
    p_grid.add_argument("--tile-w", type=int, default=540)
    p_grid.add_argument("--tile-h", type=int, default=960)

    # 3. `gcs-push` subcommand
    p_push = sub.add_parser("gcs-push", help="Sync local/Colab outputs or checkpoints to gs://bucket/path")
    p_push.add_argument("--src", nargs="+", required=True, help="Local files or directories to upload")
    p_push.add_argument("--gcs-uri", required=True, help="Destination GCS URI (gs://bucket/path)")
    p_push.add_argument("--gcp-project", default=None)
    p_push.add_argument("--gcp-sa-json", default=None)

    # 4. `gcs-pull` subcommand
    p_pull = sub.add_parser("gcs-pull", help="Pull models/assets from gs://bucket/path to Colab local NVMe")
    p_pull.add_argument("--gcs-uri", required=True, help="Source GCS URI (gs://bucket/path)")
    p_pull.add_argument("--dst", required=True, help="Destination local directory (e.g. /content/fast_models)")
    p_pull.add_argument("--gcp-project", default=None)
    p_pull.add_argument("--gcp-sa-json", default=None)

    args = parser.parse_args()

    if args.cmd == "gacha":
        seeds = parse_seed_list(args.seeds, seed_base=args.seed_base, gacha_runs=args.gacha_runs)
        out_dir = Path(args.output_dir) if args.output_dir else None
        res = run_batch_gacha_workflow(
            workflow_path=Path(args.workflow),
            seeds=seeds,
            base_prefix=args.prefix,
            target_seg_id=args.target_seg,
            api_url=args.api_url,
            output_dir=out_dir,
            gcs_bucket=args.gcs_bucket,
            gcp_project=args.gcp_project,
            gcp_sa_json=args.gcp_sa_json,
        )
        print(json.dumps(res, ensure_ascii=False), flush=True)

    elif args.cmd == "grid":
        vids = [Path(v) for v in args.videos]
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()] if args.seeds else None
        render_gacha_comparison_grid(vids, Path(args.out), seeds=seeds, tile_w=args.tile_w, tile_h=args.tile_h)

    elif args.cmd == "gcs-push":
        srcs = [Path(s) for s in args.src]
        gcs_sync_directory_or_files(
            srcs,
            gcs_uri=args.gcs_uri,
            direction="push",
            service_account_json=args.gcp_sa_json,
            project_id=args.gcp_project,
        )

    elif args.cmd == "gcs-pull":
        gcs_sync_directory_or_files(
            [Path(args.dst)],
            gcs_uri=args.gcs_uri,
            direction="pull",
            service_account_json=args.gcp_sa_json,
            project_id=args.gcp_project,
        )


if __name__ == "__main__":
    main()
