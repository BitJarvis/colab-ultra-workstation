#!/usr/bin/env python3
"""Zero-Cost CPU Canvas ComfyUI Launcher (`--cpu`) with On-Demand Serverless G4 Queue Bridge & Caddy Auth Gateway.

Why Decoupled Control Plane + On-Demand G4 Compute?
---------------------------------------------------
1. **0.00 CCU/h Interactive Canvas**:
   - Runs ComfyUI (`--cpu`) on a free Colab CPU session (`0.00 CCU/h`) with the user's 20TB/30TB Google Drive mounted.
   - Mirrors `custom_nodes` onto local SSD via single-file `custom_nodes_bundle.tar.gz` (1.5s extraction) and symlinks
     `/content/ComfyUI/models/*` and `/content/ComfyUI/output` directly to `/content/drive/MyDrive/AI_Workstation/ComfyUI/`.
   - Because ComfyUI only scans filenames (`os.listdir`) when rendering the Web UI node graph (`GET /object_info`),
     all MiniMax-H3 checkpoints, Qwen3-VL-32B encoders, LoRAs, VAEs, and custom nodes appear in dropdowns with
     **zero model loading and zero GPU CCU burn**!
2. **Smart Queue-to-Drive Bridge (`ComfyUI-Serverless-G4-Bridge` on port 8189 -> ComfyUI 8188)**:
   - Transparently proxies all Web UI HTTP & WebSocket traffic to ComfyUI (`127.0.0.1:8188`), protected by the
     Caddy 192-bit Dual-Mode Security Layer (`127.0.0.1:18188`) + Cloudflare Tunnel (`*.trycloudflare.com`).
   - In `--exec-mode g4-queue` (default), clicking **"Queue Prompt"** in the CPU Web UI intercepts `POST /prompt`
     and `POST /api/prompt`, writes the API workflow payload to
     `/content/drive/MyDrive/AI_Workstation/ComfyUI/serverless_queue/pending/job_<timestamp>_<id>.json`,
     and returns an immediate acknowledgement so the CPU instance NEVER attempts to run a 33B video diffusion model on CPU!
   - The local or remote G4 Serverless Dispatcher (`scripts/comfy_serverless_g4_dispatcher.py`) watches this queue
     (or accepts direct CLI/API workflows), automatically boots/reuses a G4 (96GB Blackwell) session, executes the
     workflow or multi-seed Gacha batch at full speed, writes the MP4/latent outputs back to the shared Google Drive
     `output/` folder (instantly browsable from the CPU Web UI), and auto-stops the G4 GPU after `--idle-timeout`!
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import secrets
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

if os.path.isdir("/content/drive/My Drive") and not os.path.exists("/content/drive/MyDrive"):
    try:
        os.symlink("/content/drive/My Drive", "/content/drive/MyDrive")
    except OSError:
        pass

DRIVE_WORKSTATION = Path("/content/drive/MyDrive/AI_Workstation")
DRIVE_COMFY = DRIVE_WORKSTATION / "ComfyUI"
LOCAL_COMFY = Path("/content/ComfyUI")
NODES_BUNDLE_ARCHIVE = DRIVE_COMFY / "custom_nodes_bundle.tar.gz"
SERVERLESS_QUEUE_DIR = DRIVE_COMFY / "serverless_queue"


def sh(cmd: str) -> str:
    safe_cmd = re.sub(r"Bearer\s+[A-Za-z0-9._-]+", "Bearer [REDACTED]", cmd)
    print(f"🔧 $ {safe_cmd}", flush=True)
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if p.returncode != 0 and p.stderr:
        print(f"⚠️ {p.stderr[-600:]}", flush=True)
    return p.stdout


def get_queue_dirs(queue_root: Path = SERVERLESS_QUEUE_DIR) -> Dict[str, Path]:
    """Ensure and return the shared Drive serverless queue directory hierarchy."""
    dirs = {
        "root": queue_root,
        "pending": queue_root / "pending",
        "running": queue_root / "running",
        "completed": queue_root / "completed",
        "failed": queue_root / "failed",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


_LAST_TS_MS = 0


def enqueue_serverless_job(
    prompt_payload: Dict[str, Any],
    queue_root: Path = SERVERLESS_QUEUE_DIR,
    source: str = "cpu_webui",
    gacha_runs: int = 1,
    seed_base: Optional[int] = None,
    prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """Serialize a ComfyUI `/prompt` request into the shared Drive `serverless_queue/pending/` folder."""
    global _LAST_TS_MS
    if not isinstance(prompt_payload, dict):
        raise ValueError("Invalid prompt payload: expected JSON object.")
    workflow = prompt_payload.get("prompt") if "prompt" in prompt_payload else prompt_payload
    if not isinstance(workflow, dict) or not workflow:
        raise ValueError("Invalid ComfyUI prompt payload: missing non-empty 'prompt' workflow dictionary.")
    has_api_nodes = any(isinstance(v, dict) and "class_type" in v for v in workflow.values())
    if not has_api_nodes:
        raise ValueError("Prompt workflow does not contain valid ComfyUI API nodes ('class_type').")

    qdirs = get_queue_dirs(queue_root)
    job_id = str(prompt_payload.get("prompt_id") or uuid.uuid4())
    # Sanitize job_id to prevent any filename traversal
    safe_job_id = re.sub(r"[^a-zA-Z0-9_-]", "", job_id) or uuid.uuid4().hex
    ts_ms = max(int(time.time() * 1000), _LAST_TS_MS + 1)
    _LAST_TS_MS = ts_ms

    extra_data = prompt_payload.get("extra_data", {})
    if not isinstance(extra_data, dict):
        extra_data = {}

    job_record = {
        "job_id": safe_job_id,
        "prompt_id": safe_job_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "timestamp_ms": ts_ms,
        "source": source,
        "status": "pending",
        "gacha_runs": max(1, int(gacha_runs)),
        "seed_base": seed_base,
        "prefix": prefix,
        "client_id": prompt_payload.get("client_id"),
        "prompt": workflow,
        "extra_data": extra_data,
    }

    job_file = qdirs["pending"] / f"job_{ts_ms}_{safe_job_id}.json"
    tmp_file = qdirs["pending"] / f".tmp_{ts_ms}_{safe_job_id}.json"
    tmp_file.write_text(json.dumps(job_record, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_file.replace(job_file)
    return {
        "prompt_id": safe_job_id,
        "number": ts_ms % 100000,
        "node_errors": {},
        "serverless_g4_queued": True,
        "queue_file": str(job_file),
    }


MODEL_COMPAT_ALIASES: list[tuple[str, str, str, str]] = [
    # Diffusion Models & UNet aliases -> Singularity v1.3 ref2va INT8 or Official fl2va FP8
    ("diffusion_models", "minimax_h3_ref2va_pruned_int8_convrot.safetensors", "diffusion_models", "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors"),
    ("diffusion_models", "minimax_h3_ref2va_int8_convrot.safetensors", "diffusion_models", "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors"),
    ("diffusion_models", "minimax_h3_ref2va_pruned_w4a8_mixed.safetensors", "diffusion_models", "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors"),
    ("diffusion_models", "minimax_h3_r2va_veda_preview_fp8.safetensors", "diffusion_models", "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors"),
    ("diffusion_models", "fasth3_omniref_8step_pruned_int8_convrot.safetensors", "diffusion_models", "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors"),
    ("diffusion_models", "minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors", "diffusion_models", "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors"),
    ("diffusion_models", "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "diffusion_models", "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"),
    ("diffusion_models", "minimax_h3_fl2va_int8_convrot.safetensors", "diffusion_models", "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"),
    ("diffusion_models", "minimax_h3_fl2va_pruned_bf16.safetensors", "diffusion_models", "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"),
    ("diffusion_models", "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors", "diffusion_models", "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"),
    ("diffusion_models", "fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors", "diffusion_models", "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"),
    ("unet", "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors", "diffusion_models", "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors"),
    ("unet", "minimax_h3_fl2va_pruned_fp8_scaled.safetensors", "diffusion_models", "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"),
    ("unet", "minimax_h3_ref2va_pruned_int8_convrot.safetensors", "diffusion_models", "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors"),
    ("unet", "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "diffusion_models", "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"),
    # Text Encoders & CLIP aliases -> Qwen3-VL-32B NVFP4 / INT8
    ("text_encoders", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "text_encoders", "qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors"),
    ("text_encoders", "qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "text_encoders", "qwen3vl_32b_minimax_h3_ultra_uncensored_heretic_int8_convrot.safetensors"),
    ("text_encoders", "qwen3vl_4b_fp8_scaled.safetensors", "text_encoders", "qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors"),
    ("text_encoders", "p-h3-image_text_encoder_int8_convrot.safetensors", "text_encoders", "qwen3vl_32b_minimax_h3_ultra_uncensored_heretic_int8_convrot.safetensors"),
    ("text_encoders", "p-h3-image_prompt_upsampler_w4a8.safetensors", "text_encoders", "qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors"),
    ("clip", "qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors", "text_encoders", "qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors"),
    ("clip", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "text_encoders", "qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors"),
    ("clip", "qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "text_encoders", "qwen3vl_32b_minimax_h3_ultra_uncensored_heretic_int8_convrot.safetensors"),
    ("clip", "qwen3vl_32b_minimax_h3_ultra_uncensored_heretic_int8_convrot.safetensors", "text_encoders", "qwen3vl_32b_minimax_h3_ultra_uncensored_heretic_int8_convrot.safetensors"),
    ("clip", "qwen3vl_4b_fp8_scaled.safetensors", "text_encoders", "qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors"),
    # VAE aliases -> Video VAE FP16
    ("vae", "minimax_h3_video_vae_int8_convrot.safetensors", "vae", "minimax_h3_video_vae_fp16.safetensors"),
    ("vae", "p-h3-image_vae_int8_convrot.safetensors", "vae", "minimax_h3_video_vae_fp16.safetensors"),
    ("vae_approx", "taeh3.safetensors", "vae", "minimax_h3_video_vae_fp16.safetensors"),
    # LoRA aliases -> Turbo 4-step / 8-step & Realism LoRAs
    ("loras", "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", "loras", "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"),
    ("loras", "minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors", "loras", "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"),
    ("loras", "minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors", "loras", "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"),
    ("loras", "Minimax_H3/minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors", "loras", "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"),
    ("loras", "TaoMate-H3-3step-ComfyUI.safetensors", "loras", "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"),
    ("loras", "minimax_h3_turbo_4step_ema_ckpt850.safetensors", "loras", "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"),
    ("loras", "MiniMaxH3_fl2va_pruned_turbo_lora_larryvrh_4step_ema_ckpt850.safetensors", "loras", "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"),
    ("loras", "minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors", "loras", "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"),
    ("loras", "MiniMax H3/fasth3_v1_dense_datafree_comfyui.safetensors", "loras", "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"),
    ("loras", "fasth3_v1_dense_datafree_comfyui.safetensors", "loras", "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"),
    ("loras", "pika_h3_flex_fl2va_v1_bf16.safetensors", "loras", "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"),
    ("loras", "pika_h3_flex_ref2va_v1_bf16.safetensors", "loras", "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"),
    ("loras", "p-h3-image_lora_bf16.safetensors", "loras", "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"),
    ("loras", "minimax_h3_lms_v1.0_r64.safetensors", "loras", "h3-realism-people-t2v-i2v-r2v.safetensors"),
    # Latent Upscalers (`latent_upscale_models` and `h3_latent_upscalers`)
    ("latent_upscale_models", "minimax_h3_latent_upscaler_3d_bf16.safetensors", "latent_upscale_models", "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"),
    ("latent_upscale_models", "minimax_h3_latent_upscaler_3d_fp16.safetensors", "latent_upscale_models", "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"),
    ("latent_upscale_models", "minimax_h3_latent_upscaler_3d_fp32.pth", "latent_upscale_models", "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"),
    ("latent_upscale_models", "h3_clean_latent_upscaler_film_epoch200.safetensors", "latent_upscale_models", "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"),
    ("h3_latent_upscalers", "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors", "latent_upscale_models", "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"),
    ("h3_latent_upscalers", "minimax_h3_latent_upscaler_3d_bf16.safetensors", "latent_upscale_models", "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"),
    ("h3_latent_upscalers", "minimax_h3_latent_upscaler_3d_fp16.safetensors", "latent_upscale_models", "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"),
    ("h3_latent_upscalers", "minimax_h3_latent_upscaler_3d_fp32.pth", "latent_upscale_models", "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"),
    ("h3_latent_upscalers", "h3_clean_latent_upscaler_film_epoch200.safetensors", "latent_upscale_models", "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"),
    # Embeddings & Semantic Bridge / Model Patches
    ("embeddings", "minimaxh3_art_is_explosion.safetensors", "embeddings", "minimaxh3_bullet_time.safetensors"),
    ("semantic_bridge", "BUNNY_H3_ActionLogic_Bridge_V1.safetensors", "model_patches", "BUNNY_H3_ActionLogic_Bridge_V1.safetensors"),
    ("semantic_bridge", "BUNNY_H3_ActionLogic_Bridge_V2.safetensors", "model_patches", "BUNNY_H3_ActionLogic_Bridge_V2.safetensors"),
    ("model_patches", "minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors", "model_patches", "BUNNY_H3_ActionLogic_Bridge_V2.safetensors"),
    # Subgraph & auxiliary template aliases (ControlNet Union pose/detector & Music-3 template placeholders)
    ("checkpoints", "sdpose_wholebody_fp16.safetensors", "diffusion_models", "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors"),
    ("diffusion_models", "rt_detr_v4-x-hgnet_fp16.safetensors", "diffusion_models", "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors"),
    ("unet", "rt_detr_v4-x-hgnet_fp16.safetensors", "diffusion_models", "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors"),
    ("diffusion_models", "minimax_music3_dit_fp16.safetensors", "diffusion_models", "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors"),
    ("unet", "minimax_music3_dit_fp16.safetensors", "diffusion_models", "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors"),
    ("text_encoders", "minimax_music3_text_encoder_pruned_int8_convrot.safetensors", "text_encoders", "qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors"),
    ("clip", "minimax_music3_text_encoder_pruned_int8_convrot.safetensors", "text_encoders", "qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors"),
    ("vae", "minimax_music3_dav.safetensors", "vae", "minimax_h3_audio_vae_fp32.safetensors"),
]


def wire_local_models_with_aliases(
    local_models_root: Path = LOCAL_COMFY / "models",
    drive_models_root: Path = DRIVE_COMFY / "models",
) -> int:
    """Create `/content/ComfyUI/models` on local SSD with symlinks to Drive weights + all compatibility aliases.

    Because Google Drive FUSE (`drivefs`) does not support creating POSIX symlinks inside the mount,
    maintaining `/content/ComfyUI/models/<category>/` on local SSD allows instant `os.listdir`/`os.walk`
    indexing and 100% compatibility with every workflow and ComfyUI template filename.
    """
    if local_models_root.is_symlink() or local_models_root.exists():
        sh(f"rm -rf '{local_models_root}'")
    local_models_root.mkdir(parents=True, exist_ok=True)

    linked = 0
    if drive_models_root.exists():
        for drive_sub in drive_models_root.iterdir():
            if not drive_sub.is_dir():
                continue
            local_sub = local_models_root / drive_sub.name
            local_sub.mkdir(parents=True, exist_ok=True)
            for item in drive_sub.iterdir():
                link_path = local_sub / item.name
                if link_path.exists() or link_path.is_symlink():
                    link_path.unlink()
                os.symlink(item, link_path)
                linked += 1

    for dst_sub, alias_rel, src_sub, canonical_name in MODEL_COMPAT_ALIASES:
        src_target = drive_models_root / src_sub / canonical_name
        if not src_target.exists():
            src_target = local_models_root / src_sub / canonical_name
        if not src_target.exists():
            continue
        alias_path = local_models_root / dst_sub / alias_rel
        alias_path.parent.mkdir(parents=True, exist_ok=True)
        if not alias_path.exists() and not alias_path.is_symlink():
            os.symlink(src_target, alias_path)
            linked += 1
    return linked


def setup_cpu_comfy_and_nodes(refresh_nodes: bool = False) -> None:
    """Clone ComfyUI to `/content/ComfyUI`, extract `custom_nodes` onto local SSD, and wire Drive model/input/output symlinks."""
    t0 = time.time()
    if not (LOCAL_COMFY / "main.py").exists():
        sh(f"git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git {LOCAL_COMFY}")

    local_cn = LOCAL_COMFY / "custom_nodes"
    if local_cn.is_symlink():
        local_cn.unlink()
    local_cn.mkdir(parents=True, exist_ok=True)

    core_repos = {
        "ComfyUI-Manager": "https://github.com/ltdrdata/ComfyUI-Manager.git",
        "ComfyUI-KJNodes": "https://github.com/kijai/ComfyUI-KJNodes.git",
        "ComfyUI-VideoHelperSuite": "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
        "ComfyUI-MiniMaxH3-Easy": "https://github.com/nkxx188/ComfyUI-MiniMaxH3-Easy.git",
        "ComfyUI-H3-Motion-Context": "https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context.git",
        "BUNNY_H3_Conditioning_Bridge": "https://github.com/aa335615543-ux/BUNNY_H3_Conditioning_Bridge.git",
        "Comfyui_Minimax_h3_latent_Upscaler": "https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler.git",
        "ComfyUI-MiniMaxH3_LatentUpscaler": "https://github.com/TripleHeadedMonkey/ComfyUI-MiniMaxH3_LatentUpscaler.git",
        "ComfyUI_MiniMaxH3_Director": "https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director.git",
        "ComfyUI-Impact-Pack": "https://github.com/ltdrdata/ComfyUI-Impact-Pack.git",
        "ComfyUI-Qwen-TTS": "https://github.com/flybirdxx/ComfyUI-Qwen-TTS.git",
        "ComfyUI-Index-TTS": "https://github.com/chenpipi0807/ComfyUI-Index-TTS.git",
    }

    if NODES_BUNDLE_ARCHIVE.exists() and not refresh_nodes:
        print(f"⚡ 检测到云盘插件归档包 ({NODES_BUNDLE_ARCHIVE.name})，秒级解压至本地 SSD...", flush=True)
        sh(f"cp '{NODES_BUNDLE_ARCHIVE}' /tmp/custom_nodes_bundle.tar.gz && tar -xzf /tmp/custom_nodes_bundle.tar.gz -C '{local_cn}/'")
    else:
        print("⚡ 正在通过 GitHub 12 路并发克隆核心节点至本地 SSD (彻底规避 FUSE 小文件遍历风暴，耗时 ~3s)...", flush=True)
        clone_procs = []
        for name, url in core_repos.items():
            target = local_cn / name
            if not (target / "__init__.py").exists() and not (target / "nodes.py").exists():
                shutil.rmtree(target, ignore_errors=True)
                clone_procs.append(subprocess.Popen(f"git clone --depth 1 -q {url} '{target}'", shell=True))
        for cp in clone_procs:
            cp.wait()
        bunny_dst = local_cn / "BUNNY_H3_Conditioning_Bridge/models"
        bunny_dst.mkdir(parents=True, exist_ok=True)
        for bname in ("BUNNY_H3_ActionLogic_Bridge_V1.safetensors", "BUNNY_H3_ActionLogic_Bridge_V2.safetensors"):
            bsrc = DRIVE_COMFY / "models/model_patches" / bname
            if bsrc.exists():
                shutil.copy2(bsrc, bunny_dst / bname)
        sh(f"find '{local_cn}' \\( -name '.git' -o -name '__pycache__' \\) -prune -exec rm -rf {{}} +")
        sh(f"tar -czf /tmp/custom_nodes_bundle.tar.gz -C '{local_cn}' .")
        subprocess.Popen(
            f"nohup cp /tmp/custom_nodes_bundle.tar.gz '{NODES_BUNDLE_ARCHIVE}' >/dev/null 2>&1 &",
            shell=True,
        )

    # Patch TimbreAudioLoader in ComfyUI-Index-TTS so it caches its glob scan instead of re-scanning on every GET /object_info
    timbre_loader = local_cn / "ComfyUI-Index-TTS" / "timbre_audio_loader.py"
    if timbre_loader.exists():
        t_txt = timbre_loader.read_text(encoding="utf-8", errors="ignore")
        if "if not cls.audio_files_cache:" not in t_txt and "cls.scan_timbre_audio_files()" in t_txt:
            t_txt = t_txt.replace(
                "        cls.scan_timbre_audio_files()",
                "        if not cls.audio_files_cache:\n            cls.scan_timbre_audio_files()",
            )
            timbre_loader.write_text(t_txt, encoding="utf-8")

    # Install built-in H3 AV_LATENT Checkpoint Node (H3SaveAVLatent / H3LoadAVLatent) so CPU Canvas has identical node definitions
    ckpt_node_dir = local_cn / "ComfyUI-H3-Latent-Checkpoint"
    ckpt_node_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_node_dir / "__init__.py").write_text(
        '''import copy
from pathlib import Path
import torch

LATENT_DIR = Path("/content/drive/MyDrive/AI_Workstation/ComfyUI/output/latents")


def _safe_ckpt_path(checkpoint_name: str) -> Path:
    base = LATENT_DIR.resolve()
    base.mkdir(parents=True, exist_ok=True)
    normalized = str(checkpoint_name or "").strip().replace(chr(92), "/")
    if not normalized or normalized.endswith("/"):
        raise ValueError(f"[H3LatentCheckpoint] Invalid checkpoint_name: {checkpoint_name!r}")
    raw_name = Path(normalized).name
    if not raw_name or raw_name in (".", "..") or "/" in raw_name or chr(92) in raw_name:
        raise ValueError(f"[H3LatentCheckpoint] Invalid checkpoint_name: {checkpoint_name!r}")
    fname = raw_name if raw_name.endswith(".pt") else f"{raw_name}.pt"
    candidate = (base / fname).resolve()
    if candidate.parent != base:
        raise ValueError(f"[H3LatentCheckpoint] Path traversal blocked: {checkpoint_name!r}")
    return candidate


def _to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu(v) for v in obj)
    return copy.deepcopy(obj)


class H3SaveAVLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "checkpoint_name": ("STRING", {"default": "Seg01_av_latent.pt"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "save_latent"
    OUTPUT_NODE = True
    CATEGORY = "MiniMax-H3/Checkpoint"

    def save_latent(self, latent, checkpoint_name):
        out_path = _safe_ckpt_path(checkpoint_name)
        torch.save(_to_cpu(latent), out_path)
        return (latent,)


class H3LoadAVLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint_name": ("STRING", {"default": "Seg02_av_latent.pt"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "load_latent"
    CATEGORY = "MiniMax-H3/Checkpoint"

    def load_latent(self, checkpoint_name):
        in_path = _safe_ckpt_path(checkpoint_name)
        if not in_path.exists():
            raise FileNotFoundError(f"[H3LatentCheckpoint] Missing latent checkpoint on Drive: {in_path}")
        try:
            latent = torch.load(in_path, map_location="cpu", weights_only=True)
        except Exception:
            latent = torch.load(in_path, map_location="cpu", weights_only=False)
        return (latent,)


NODE_CLASS_MAPPINGS = {
    "H3SaveAVLatent": H3SaveAVLatent,
    "H3LoadAVLatent": H3LoadAVLatent,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3SaveAVLatent": "H3 Save AV Latent Checkpoint (Drive)",
    "H3LoadAVLatent": "H3 Load AV Latent Checkpoint (Drive)",
}
''',
        encoding="utf-8",
    )

    # Wire local `/content/ComfyUI/models` with symlinks to Drive models + all workflow/template compatibility aliases
    wire_local_models_with_aliases()

    # Wire shared `output` and `input` to Google Drive so CPU Canvas & G4 share reference images and rendered MP4s
    for shared_sub in ("output", "input"):
        drive_sub = DRIVE_COMFY / shared_sub
        drive_sub.mkdir(parents=True, exist_ok=True)
        local_sub = LOCAL_COMFY / shared_sub
        if local_sub.is_symlink() or local_sub.exists():
            sh(f"rm -rf '{local_sub}'")
        os.symlink(drive_sub, local_sub)

    # Populate Drive `input/` with all workflow sample_inputs and template reference images so templates never 404
    drive_input = DRIVE_COMFY / "input"
    if (DRIVE_WORKSTATION / "workflows").exists():
        for sample_file in (DRIVE_WORKSTATION / "workflows").rglob("sample_inputs/*"):
            if sample_file.is_file():
                dst_in = drive_input / sample_file.name
                if not dst_in.exists():
                    shutil.copy2(sample_file, dst_in)
    if (drive_input / "4abdabfc6dfd498e891df7984a977b25.jpeg").exists() and not (drive_input / "red_superboy_on_city_roof.png").exists():
        shutil.copy2(drive_input / "4abdabfc6dfd498e891df7984a977b25.jpeg", drive_input / "red_superboy_on_city_roof.png")

    # Keep `/content/ComfyUI/user` on fast local SSD (so SQLite `comfyui.db`, logs, and Manager cache never stall on FUSE),
    # while symlinking only `user/default/workflows` to Google Drive for persistent workflow storage!
    local_user = LOCAL_COMFY / "user"
    if local_user.is_symlink():
        local_user.unlink()
    user_default = local_user / "default"
    user_default.mkdir(parents=True, exist_ok=True)
    (user_default / "subgraphs").mkdir(parents=True, exist_ok=True)

    drive_workflows = DRIVE_COMFY / "user" / "default" / "workflows"
    drive_workflows.mkdir(parents=True, exist_ok=True)
    local_workflows = user_default / "workflows"
    if local_workflows.is_symlink() or local_workflows.exists():
        sh(f"rm -rf '{local_workflows}'")
    os.symlink(drive_workflows, local_workflows)

    get_queue_dirs(SERVERLESS_QUEUE_DIR)

    # Install lightweight dependencies for ComfyUI + custom nodes on CPU
    pkgs = "aiohttp gguf imageio-ffmpeg librosa soundfile tiktoken sentencepiece sox accelerate einops onnxruntime ultralytics opencv-python-headless scipy"
    sh(
        f"if command -v uv >/dev/null 2>&1; then "
        f"uv pip install --system -q -r {LOCAL_COMFY}/requirements.txt {pkgs}; "
        f"else pip install -q -r {LOCAL_COMFY}/requirements.txt {pkgs}; fi"
    )

    # Bootstrap default user files & static user.css so ComfyUI Frontend v1.58+ never 404s on initial page load
    for wf_src_root in (DRIVE_WORKSTATION / "workflows",):
        if wf_src_root.exists():
            for wf_path in wf_src_root.rglob("*.json"):
                if "workflow" in wf_path.name.lower() and "api" not in wf_path.name.lower():
                    dst_wf = drive_workflows / wf_path.name
                    if not dst_wf.exists():
                        shutil.copy2(wf_path, dst_wf)
    default_files = {
        user_default / "user.css": "/* ComfyUI user custom stylesheet */\n",
        user_default / "comfy.settings.json": "{}\n",
        user_default / "comfy.templates.json": "[]\n",
        drive_workflows / ".index.json": "{}\n",
    }
    for fpath, default_content in default_files.items():
        if not fpath.exists():
            fpath.write_text(default_content, encoding="utf-8")
    try:
        import comfyui_frontend_package  # type: ignore

        static_root = Path(comfyui_frontend_package.__file__).resolve().parent / "static"
        if static_root.is_dir() and not (static_root / "user.css").exists():
            (static_root / "user.css").write_text("/* ComfyUI user custom stylesheet */\n", encoding="utf-8")
    except Exception:
        pass

    print(f"✅ 0 CCU CPU 独立画布环境与云盘模型目录软链初始化完成 (耗时 {time.time() - t0:.1f}s)", flush=True)


def write_and_start_queue_bridge(
    comfy_port: int = 8188,
    bridge_port: int = 8189,
    exec_mode: str = "g4-queue",
    queue_dir: Path = SERVERLESS_QUEUE_DIR,
) -> None:
    """Start an async aiohttp reverse-proxy & `/prompt` queue interceptor on `bridge_port` in front of `comfy_port`."""
    bridge_script = Path("/tmp/comfy_cpu_queue_bridge.py")
    bridge_code = f'''#!/usr/bin/env python3
"""Async HTTP + WebSocket Reverse Proxy with On-Demand G4 Queue Interceptor & Real-Time UI Feedback for CPU ComfyUI Canvas."""
import asyncio
import json
import re
import sys
import time
import uuid
from pathlib import Path
import aiohttp
from aiohttp import web

COMFY_UPSTREAM = "http://127.0.0.1:{comfy_port}"
COMFY_WS_UPSTREAM = "ws://127.0.0.1:{comfy_port}"
BRIDGE_PORT = {bridge_port}
EXEC_MODE = {exec_mode!r}
QUEUE_ROOT = Path({str(queue_dir)!r})
OUTPUT_ROOT = Path("/content/drive/MyDrive/AI_Workstation/ComfyUI/output")

for _sub in ("pending", "running", "completed", "failed"):
    (QUEUE_ROOT / _sub).mkdir(parents=True, exist_ok=True)


_LAST_TS_MS = 0
_WS_CLIENTS: set[web.WebSocketResponse] = set()
_NOTIFIED_STATES: dict[str, str] = {{}}


def _infer_output_node_id(prompt: dict) -> str:
    if not isinstance(prompt, dict):
        return "1"
    for nid, node in prompt.items():
        if isinstance(node, dict) and node.get("class_type") in ("SaveVideo", "VHS_VideoCombine", "SaveWEBM", "SaveAnimatedWEBP", "SaveImage"):
            return str(nid)
    return str(next(iter(prompt.keys()), "1"))


def _ensure_output_file_synced_from_drive_api(filename: str, subfolder: str = "") -> Path | None:
    """Ensure a rendered output file exists on local `/content/ComfyUI/output[/subfolder]/filename`, pulling via Drive v3 API if FUSE hasn't refreshed yet."""
    safe_name = Path(str(filename)).name
    if not safe_name:
        return None
    safe_sub = str(subfolder or "").strip("/").replace("..", "")
    dest_dir = (OUTPUT_ROOT / safe_sub) if safe_sub else OUTPUT_ROOT
    dest_path = dest_dir / safe_name
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return dest_path
    try:
        import urllib.parse
        import urllib.request

        req_tok = urllib.request.Request(
            "http://127.0.0.1:8012/computeMetadata/v1/instance/service-accounts/default/token",
            headers={{"Metadata-Flavor": "Google"}},
        )
        tok = json.loads(urllib.request.urlopen(req_tok, timeout=4).read().decode())["access_token"]
        escaped = safe_name.replace("'", "\\'")
        params = urllib.parse.urlencode({{
            "q": f"trashed = false and name = '{{escaped}}'",
            "fields": "files(id,name,size)",
            "pageSize": 5,
        }})
        req_f = urllib.request.Request(
            f"https://www.googleapis.com/drive/v3/files?{{params}}",
            headers={{"Authorization": f"Bearer {{tok}}"}},
        )
        files = json.loads(urllib.request.urlopen(req_f, timeout=6).read().decode()).get("files", [])
        if not files:
            return None
        fid = files[0]["id"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        req_dl = urllib.request.Request(
            f"https://www.googleapis.com/drive/v3/files/{{fid}}?alt=media",
            headers={{"Authorization": f"Bearer {{tok}}"}},
        )
        tmp_dl = dest_dir / f".tmp_{{safe_name}}"
        with urllib.request.urlopen(req_dl, timeout=30) as r_in, open(tmp_dl, "wb") as f_out:
            while True:
                chunk = r_in.read(1024 * 1024)
                if not chunk:
                    break
                f_out.write(chunk)
        tmp_dl.replace(dest_path)
        print(f"⚡ [Drive-v3-Sync] 已直接从云端同步渲染产出至画布目录: {{dest_path}}", flush=True)
        return dest_path
    except Exception as exc:
        print(f"⚠️ [Drive-v3-Sync] 同步 {{safe_name}} 跳过: {{exc}}", flush=True)
        return None


def _build_outputs_from_expected_files(jdata: dict) -> dict:
    existing = jdata.get("outputs")
    if isinstance(existing, dict) and existing:
        for _nid, nout in existing.items():
            if isinstance(nout, dict):
                for item in nout.get("images") or []:
                    if isinstance(item, dict) and item.get("filename"):
                        _ensure_output_file_synced_from_drive_api(item["filename"], item.get("subfolder", ""))
        return existing
    expected = jdata.get("expected_files") or []
    if not expected:
        return {{}}
    prompt = jdata.get("prompt") or {{}}
    out_nid = _infer_output_node_id(prompt)
    items = []
    for fname in expected:
        name = Path(str(fname)).name
        subfolder = "video" if name.lower().endswith(".mp4") else ""
        synced = _ensure_output_file_synced_from_drive_api(name, subfolder)
        if synced is None and (OUTPUT_ROOT / name).exists():
            subfolder = ""
        items.append({{
            "filename": name,
            "subfolder": subfolder,
            "type": "output",
        }})
    if not items:
        return {{}}
    return {{
        out_nid: {{
            "images": items,
            "animated": [True] * len(items),
        }}
    }}


async def _broadcast_ws(payload: dict) -> None:
    if not _WS_CLIENTS:
        return
    msg = json.dumps(payload, ensure_ascii=False)
    stale = []
    for ws in list(_WS_CLIENTS):
        if ws.closed:
            stale.append(ws)
            continue
        try:
            await ws.send_str(msg)
        except Exception:
            stale.append(ws)
    for ws in stale:
        _WS_CLIENTS.discard(ws)


_LAST_DRIVE_QUEUE_SYNC_TS = 0.0
_QUEUE_FOLDER_IDS: dict[str, str] = {{}}


def _sync_queue_transitions_from_drive_api() -> None:
    """Reconcile `pending` / `running` jobs on local FUSE against Google Drive v3 API so cross-VM state changes appear immediately."""
    global _LAST_DRIVE_QUEUE_SYNC_TS
    now = time.time()
    if now - _LAST_DRIVE_QUEUE_SYNC_TS < 2.5:
        return
    local_active = list((QUEUE_ROOT / "pending").glob("job_*.json")) + list((QUEUE_ROOT / "running").glob("job_*.json"))
    if not local_active and (now - _LAST_DRIVE_QUEUE_SYNC_TS < 10.0):
        return
    _LAST_DRIVE_QUEUE_SYNC_TS = now
    try:
        import urllib.parse
        import urllib.request

        req_tok = urllib.request.Request(
            "http://127.0.0.1:8012/computeMetadata/v1/instance/service-accounts/default/token",
            headers={{"Metadata-Flavor": "Google"}},
        )
        tok = json.loads(urllib.request.urlopen(req_tok, timeout=3).read().decode())["access_token"]
        if len(_QUEUE_FOLDER_IDS) < 4:
            q_folders = (
                "trashed = false and mimeType = 'application/vnd.google-apps.folder' and "
                "(name = 'serverless_queue' or name = 'pending' or name = 'running' or name = 'completed' or name = 'failed')"
            )
            p_f = urllib.parse.urlencode({{"q": q_folders, "fields": "files(id,name,parents)", "pageSize": 50}})
            r_f = urllib.request.Request(
                f"https://www.googleapis.com/drive/v3/files?{{p_f}}",
                headers={{"Authorization": f"Bearer {{tok}}"}},
            )
            flist = json.loads(urllib.request.urlopen(r_f, timeout=5).read().decode()).get("files", [])
            sq_ids = {{f["id"] for f in flist if f.get("name") == "serverless_queue"}}
            for f in flist:
                if f.get("name") in ("pending", "running", "completed", "failed"):
                    parents = set(f.get("parents") or [])
                    if not sq_ids or (parents & sq_ids):
                        _QUEUE_FOLDER_IDS[f["id"]] = f["name"]

        p_j = urllib.parse.urlencode({{
            "q": "trashed = false and name contains 'job_' and name contains '.json'",
            "fields": "files(id,name,parents,modifiedTime)",
            "orderBy": "modifiedTime desc",
            "pageSize": 30,
        }})
        r_j = urllib.request.Request(
            f"https://www.googleapis.com/drive/v3/files?{{p_j}}",
            headers={{"Authorization": f"Bearer {{tok}}"}},
        )
        jfiles = json.loads(urllib.request.urlopen(r_j, timeout=5).read().decode()).get("files", [])
        seen_names: set[str] = set()
        for jf in jfiles:
            jname = jf.get("name", "")
            if not (jname.startswith("job_") and jname.endswith(".json")) or jname in seen_names:
                continue
            parents = jf.get("parents") or []
            cloud_st = None
            for pid in parents:
                if pid in _QUEUE_FOLDER_IDS:
                    cloud_st = _QUEUE_FOLDER_IDS[pid]
                    break
            seen_names.add(jname)
            if (QUEUE_ROOT / "completed" / jname).exists() or (QUEUE_ROOT / "failed" / jname).exists():
                for stale_st in ("pending", "running"):
                    (QUEUE_ROOT / stale_st / jname).unlink(missing_ok=True)
                continue
            target_local = (QUEUE_ROOT / cloud_st / jname) if cloud_st else None
            need_pull = False
            if cloud_st in ("running", "completed", "failed") and (
                not target_local.exists()
                or (QUEUE_ROOT / "pending" / jname).exists()
                or (cloud_st in ("completed", "failed") and (QUEUE_ROOT / "running" / jname).exists())
            ):
                need_pull = True
            if need_pull:
                r_dl = urllib.request.Request(
                    f"https://www.googleapis.com/drive/v3/files/{{jf['id']}}?alt=media",
                    headers={{"Authorization": f"Bearer {{tok}}"}},
                )
                raw_bytes = urllib.request.urlopen(r_dl, timeout=8).read()
                jdata = json.loads(raw_bytes.decode("utf-8"))
                real_st = jdata.get("status") if jdata.get("status") in ("pending", "running", "completed", "failed") else (cloud_st or "completed")
                dst_p = QUEUE_ROOT / real_st / jname
                dst_p.parent.mkdir(parents=True, exist_ok=True)
                dst_p.write_bytes(raw_bytes)
                for other_st in ("pending", "running", "completed", "failed"):
                    if other_st != real_st:
                        (QUEUE_ROOT / other_st / jname).unlink(missing_ok=True)
    except Exception:
        pass


def _scan_all_serverless_jobs() -> dict[str, list[tuple[Path, dict]]]:
    _sync_queue_transitions_from_drive_api()
    res: dict[str, list[tuple[Path, dict]]] = {{"pending": [], "running": [], "completed": [], "failed": []}}
    seen_by_name: dict[str, tuple[str, Path, dict]] = {{}}
    state_priority = {{"pending": 1, "running": 2, "completed": 3, "failed": 3}}
    for st in ("pending", "running", "completed", "failed"):
        files = sorted((QUEUE_ROOT / st).glob("job_*.json"), key=lambda p: p.name)
        for jf in files:
            try:
                jdata = json.loads(jf.read_text(encoding="utf-8"))
                real_st = jdata.get("status") if jdata.get("status") in state_priority else st
                prev = seen_by_name.get(jf.name)
                if prev is None or state_priority[real_st] >= state_priority[prev[0]]:
                    seen_by_name[jf.name] = (real_st, jf, jdata)
            except Exception:
                pass
    for real_st, jf, jdata in seen_by_name.values():
        if real_st == "completed" and not jdata.get("outputs") and jdata.get("expected_files"):
            jdata["outputs"] = _build_outputs_from_expected_files(jdata)
        res[real_st].append((jf, jdata))
    return res


def _to_queue_tuple(jdata: dict) -> list:
    ts_ms = int(jdata.get("timestamp_ms") or int(time.time() * 1000))
    pid = str(jdata.get("prompt_id") or jdata.get("job_id") or "")
    prompt = jdata.get("prompt") or {{}}
    extra_data = dict(jdata.get("extra_data") or {{}})
    extra_data.setdefault("create_time", ts_ms)
    out_nid = _infer_output_node_id(prompt)
    return [ts_ms % 100000, pid, prompt, extra_data, [out_nid]]


def _to_job_api_entry(jdata: dict, st: str, include_outputs: bool = False) -> dict:
    ts_ms = int(jdata.get("timestamp_ms") or int(time.time() * 1000))
    pid = str(jdata.get("prompt_id") or jdata.get("job_id") or "")
    extra_data = dict(jdata.get("extra_data") or {{}})
    create_time = int(extra_data.get("create_time") or ts_ms)
    workflow_id = (extra_data.get("extra_pnginfo") or {{}}).get("workflow", {{}}).get("id")
    status_map = {{
        "pending": "pending",
        "running": "in_progress",
        "completed": "completed",
        "failed": "failed",
    }}
    api_status = status_map.get(st, "completed")
    outputs = _build_outputs_from_expected_files(jdata) if st == "completed" else {{}}
    preview_output = None
    outputs_count = 0
    for nid, nout in outputs.items():
        if isinstance(nout, dict):
            imgs = nout.get("images") or []
            outputs_count += len(imgs)
            if imgs and preview_output is None:
                preview_output = {{
                    **imgs[0],
                    "nodeId": str(nid),
                    "mediaType": "images",
                }}
    entry = {{
        "id": pid,
        "status": api_status,
        "priority": ts_ms % 100000,
        "create_time": create_time,
        "outputs_count": outputs_count,
        "previewable_outputs_count": outputs_count,
    }}
    if workflow_id:
        entry["workflow_id"] = workflow_id
    if preview_output:
        entry["preview_output"] = preview_output
    if st in ("running", "completed", "failed"):
        entry["execution_start_time"] = int(jdata.get("started_at_ms") or create_time)
    if st in ("completed", "failed"):
        entry["execution_end_time"] = int(jdata.get("completed_at_ms") or int(time.time() * 1000))
    if st == "failed" and jdata.get("error"):
        entry["execution_error"] = {{"exception_message": str(jdata.get("error")), "node_id": _infer_output_node_id(jdata.get("prompt") or {{}})}}
    if include_outputs:
        entry["outputs"] = outputs
        entry["execution_status"] = {{
            "status_str": "success" if st == "completed" else ("error" if st == "failed" else api_status),
            "completed": st == "completed",
            "serverless_g4": True,
            "expected_files": jdata.get("expected_files", []),
            "messages": [],
        }}
        entry["execution_meta"] = {{}}
        entry["workflow"] = {{
            "prompt": jdata.get("prompt") or {{}},
            "extra_data": extra_data,
        }}
    return entry


def _enqueue_job(payload: dict) -> tuple[dict, dict]:
    global _LAST_TS_MS
    if not isinstance(payload, dict):
        raise ValueError("Invalid prompt payload: expected JSON object.")
    workflow = payload.get("prompt") if "prompt" in payload else payload
    if not isinstance(workflow, dict) or not workflow:
        raise ValueError("Missing non-empty 'prompt' workflow in request.")
    has_api_nodes = any(isinstance(v, dict) and "class_type" in v for v in workflow.values())
    if not has_api_nodes:
        raise ValueError("Prompt workflow does not contain valid ComfyUI API nodes ('class_type').")
    raw_id = str(payload.get("prompt_id") or uuid.uuid4())
    job_id = re.sub(r"[^a-zA-Z0-9_-]", "", raw_id) or uuid.uuid4().hex
    ts_ms = max(int(time.time() * 1000), _LAST_TS_MS + 1)
    _LAST_TS_MS = ts_ms
    extra_data = payload.get("extra_data") if isinstance(payload.get("extra_data"), dict) else {{}}
    extra_data.setdefault("create_time", ts_ms)
    record = {{
        "job_id": job_id,
        "prompt_id": job_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "timestamp_ms": ts_ms,
        "source": "cpu_webui",
        "status": "pending",
        "gacha_runs": max(1, int(payload.get("gacha_runs", 1))),
        "seed_base": payload.get("seed_base"),
        "prefix": payload.get("prefix"),
        "client_id": payload.get("client_id"),
        "prompt": workflow,
        "extra_data": extra_data,
    }}
    job_file = QUEUE_ROOT / "pending" / f"job_{{ts_ms}}_{{job_id}}.json"
    tmp_file = QUEUE_ROOT / "pending" / f".tmp_{{ts_ms}}_{{job_id}}.json"
    tmp_file.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_file.replace(job_file)
    print(f"📥 [G4-Serverless-Queue] 已拦截 Web UI 渲染请求并写入云盘队列: {{job_file.name}}", flush=True)
    return (
        {{
            "prompt_id": job_id,
            "number": ts_ms % 100000,
            "node_errors": {{}},
            "serverless_g4_queued": True,
            "queue_file": str(job_file),
        }},
        record,
    )


async def _watch_queue_state_loop(app: web.Application) -> None:
    last_q_rem = -1
    step_tick = 0
    while True:
        try:
            await asyncio.sleep(1.5)
            step_tick += 1
            by_state = _scan_all_serverless_jobs()
            q_rem = len(by_state["pending"]) + len(by_state["running"])
            if q_rem != last_q_rem:
                last_q_rem = q_rem
                await _broadcast_ws({{"type": "status", "data": {{"status": {{"exec_info": {{"queue_remaining": q_rem}}}}}}}})

            for _jf, jdata in by_state["pending"]:
                pid = str(jdata.get("prompt_id") or jdata.get("job_id") or "")
                if not pid:
                    continue
                if _NOTIFIED_STATES.get(pid) != "pending":
                    _NOTIFIED_STATES[pid] = "pending"
                    out_nid = _infer_output_node_id(jdata.get("prompt") or {{}})
                    await _broadcast_ws({{"type": "execution_start", "data": {{"prompt_id": pid, "timestamp": int(time.time() * 1000)}}}})
                    await _broadcast_ws({{"type": "executing", "data": {{"node": out_nid, "display_node": out_nid, "prompt_id": pid}}}})

            for _jf, jdata in by_state["running"]:
                pid = str(jdata.get("prompt_id") or jdata.get("job_id") or "")
                if not pid:
                    continue
                out_nid = _infer_output_node_id(jdata.get("prompt") or {{}})
                if _NOTIFIED_STATES.get(pid) != "running":
                    _NOTIFIED_STATES[pid] = "running"
                    await _broadcast_ws({{"type": "execution_start", "data": {{"prompt_id": pid, "timestamp": int(time.time() * 1000)}}}})
                    await _broadcast_ws({{"type": "executing", "data": {{"node": out_nid, "display_node": out_nid, "prompt_id": pid}}}})
                await _broadcast_ws({{
                    "type": "progress",
                    "data": {{"value": min(19, (step_tick % 20) + 1), "max": 20, "prompt_id": pid, "node": out_nid}},
                }})

            for _jf, jdata in by_state["completed"]:
                pid = str(jdata.get("prompt_id") or jdata.get("job_id") or "")
                if not pid or _NOTIFIED_STATES.get(pid) == "completed":
                    continue
                prev = _NOTIFIED_STATES.get(pid)
                _NOTIFIED_STATES[pid] = "completed"
                if prev in ("pending", "running"):
                    outputs = _build_outputs_from_expected_files(jdata)
                    for nid, nout in outputs.items():
                        await _broadcast_ws({{
                            "type": "executed",
                            "data": {{"node": nid, "display_node": nid, "output": nout, "prompt_id": pid}},
                        }})
                    await _broadcast_ws({{"type": "execution_success", "data": {{"prompt_id": pid, "timestamp": int(time.time() * 1000)}}}})
                    await _broadcast_ws({{"type": "executing", "data": {{"node": None, "prompt_id": pid}}}})

            for _jf, jdata in by_state["failed"]:
                pid = str(jdata.get("prompt_id") or jdata.get("job_id") or "")
                if not pid or _NOTIFIED_STATES.get(pid) == "failed":
                    continue
                prev = _NOTIFIED_STATES.get(pid)
                _NOTIFIED_STATES[pid] = "failed"
                if prev in ("pending", "running"):
                    out_nid = _infer_output_node_id(jdata.get("prompt") or {{}})
                    await _broadcast_ws({{
                        "type": "execution_error",
                        "data": {{
                            "prompt_id": pid,
                            "node_id": out_nid,
                            "node_type": "ServerlessG4",
                            "exception_message": str(jdata.get("error") or "Serverless G4 execution failed"),
                            "exception_type": "RuntimeError",
                            "traceback": [],
                            "current_inputs": {{}},
                            "current_outputs": [],
                        }},
                    }})
                    await _broadcast_ws({{"type": "executing", "data": {{"node": None, "prompt_id": pid}}}})
        except asyncio.CancelledError:
            break
        except Exception:
            pass


async def handle_serverless_status(request: web.Request) -> web.Response:
    by_state = _scan_all_serverless_jobs()
    counts = {{st: len(by_state[st]) for st in ("pending", "running", "completed", "failed")}}
    recent = []
    for st in ("pending", "running", "completed", "failed"):
        items = sorted(by_state[st], key=lambda item: int(item[1].get("timestamp_ms") or 0), reverse=True)
        for f, data in items[:5]:
            recent.append({{
                "job_id": data.get("job_id"),
                "status": st,
                "created_at": data.get("created_at"),
                "completed_at": data.get("completed_at"),
                "expected_files": data.get("expected_files", []),
                "file": f.name,
            }})

    rt_state: dict = {{}}
    for rt_cand in (Path("/tmp/ct_comfyui/g4_runtime_state.json"), QUEUE_ROOT / "g4_runtime_state.json"):
        if rt_cand.exists():
            try:
                rt_state = json.loads(rt_cand.read_text(encoding="utf-8"))
                break
            except Exception:
                pass

    g4_state = str(rt_state.get("g4_state") or "stopped")
    if counts.get("running", 0) > 0 and g4_state in ("stopped", "ready_warm"):
        g4_state = "executing"
    elif counts.get("pending", 0) > 0 and g4_state == "stopped":
        g4_state = "booting"

    last_act = float(rt_state.get("last_activity_ts") or 0.0)
    if g4_state == "ready_warm" and last_act > 0:
        idle_seconds = round(max(0.0, time.time() - last_act), 1)
    else:
        idle_seconds = float(rt_state.get("idle_seconds") or 0.0) if g4_state == "ready_warm" else 0.0
    idle_timeout_seconds = int(rt_state.get("idle_timeout_seconds") or 600)

    return web.json_response({{
        "exec_mode": EXEC_MODE,
        "queue_root": str(QUEUE_ROOT),
        "g4_session": rt_state.get("g4_session", "comfy-g4"),
        "g4_state": g4_state,
        "idle_seconds": idle_seconds,
        "idle_timeout_seconds": idle_timeout_seconds,
        "auto_start_enabled": bool(rt_state.get("auto_start_enabled", True)),
        "auto_stop_enabled": bool(rt_state.get("auto_stop_enabled", True)),
        "g4_runtime": rt_state,
        "counts": counts,
        "recent_jobs": recent[:10],
    }})


async def handle_proxy(request: web.Request) -> web.StreamResponse:
    path_qs = request.rel_url.path_qs
    path = request.rel_url.path

    # 1. Status endpoint for the Serverless G4 Dispatcher
    if path in ("/serverless/status", "/api/serverless/status"):
        return await handle_serverless_status(request)

    # 2. WebSocket proxy (`/ws`) with real-time Serverless G4 status & execution event injection
    if request.headers.get("Upgrade", "").lower() == "websocket":
        ws_server = web.WebSocketResponse()
        await ws_server.prepare(request)
        _WS_CLIENTS.add(ws_server)
        session: aiohttp.ClientSession = request.app["client_session"]
        try:
            async with session.ws_connect(
                f"{{COMFY_WS_UPSTREAM}}{{path_qs}}",
                origin=COMFY_UPSTREAM,
            ) as ws_client:
                async def _c2s():
                    async for msg in ws_server:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await ws_client.send_str(msg.data)
                        elif msg.type == aiohttp.WSMsgType.BINARY:
                            await ws_client.send_bytes(msg.data)
                        elif msg.type == aiohttp.WSMsgType.CLOSE:
                            await ws_client.close()
                async def _s2c():
                    async for msg in ws_client:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                parsed = json.loads(msg.data)
                                if isinstance(parsed, dict) and parsed.get("type") == "status":
                                    by_st = _scan_all_serverless_jobs()
                                    q_rem = len(by_st["pending"]) + len(by_st["running"])
                                    exec_info = parsed.setdefault("data", {{}}).setdefault("status", {{}}).setdefault("exec_info", {{}})
                                    exec_info["queue_remaining"] = int(exec_info.get("queue_remaining") or 0) + q_rem
                                    await ws_server.send_str(json.dumps(parsed, ensure_ascii=False))
                                    continue
                            except Exception:
                                pass
                            await ws_server.send_str(msg.data)
                        elif msg.type == aiohttp.WSMsgType.BINARY:
                            await ws_server.send_bytes(msg.data)
                        elif msg.type == aiohttp.WSMsgType.CLOSE:
                            await ws_server.close()
                await asyncio.gather(_c2s(), _s2c(), return_exceptions=True)
        finally:
            _WS_CLIENTS.discard(ws_server)
        return ws_server

    # 3. Intercept `POST /prompt` or `POST /api/prompt` when in `g4-queue` mode
    if request.method == "POST" and path in ("/prompt", "/api/prompt"):
        force_cpu = request.headers.get("X-Comfy-Exec-Target", "").lower() == "cpu"
        if EXEC_MODE == "g4-queue" and not force_cpu:
            try:
                body = await request.json()
                res, record = _enqueue_job(body)
                pid = res["prompt_id"]
                _NOTIFIED_STATES[pid] = "pending"
                by_st = _scan_all_serverless_jobs()
                q_rem = len(by_st["pending"]) + len(by_st["running"])
                out_nid = _infer_output_node_id(record.get("prompt") or {{}})
                await _broadcast_ws({{"type": "status", "data": {{"status": {{"exec_info": {{"queue_remaining": q_rem}}}}}}}})
                await _broadcast_ws({{"type": "execution_start", "data": {{"prompt_id": pid, "timestamp": record["timestamp_ms"]}}}})
                await _broadcast_ws({{"type": "executing", "data": {{"node": out_nid, "display_node": out_nid, "prompt_id": pid}}}})
                return web.json_response(res)
            except Exception as exc:
                return web.json_response({{"error": str(exc), "node_errors": {{}}}}, status=400)

    # 3b. Ensure `/view` or `/api/view` output files are synced via Drive v3 API if FUSE cache hasn't refreshed yet
    if request.method == "GET" and path in ("/view", "/api/view"):
        q_fname = request.rel_url.query.get("filename", "")
        q_sub = request.rel_url.query.get("subfolder", "")
        q_type = request.rel_url.query.get("type", "output")
        if q_fname and q_type == "output":
            _ensure_output_file_synced_from_drive_api(q_fname, q_sub)

    # 4. Proxy and augment `/queue`, `/prompt` (GET), `/api/jobs`, and `/history` with Serverless G4 queue items
    session: aiohttp.ClientSession = request.app["client_session"]
    target_url = f"{{COMFY_UPSTREAM}}{{path_qs}}"
    body_bytes = await request.read() if request.can_read_body else None
    headers = {{
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length", "origin", "referer", "sec-fetch-site", "sec-fetch-mode", "sec-fetch-dest")
    }}

    async with session.request(request.method, target_url, headers=headers, data=body_bytes, allow_redirects=False) as resp:
        resp_body = await resp.read()
        out_headers = {{
            k: v for k, v in resp.headers.items()
            if k.lower() not in ("content-encoding", "transfer-encoding", "content-length")
        }}

        if request.method == "GET" and path in ("/prompt", "/api/prompt") and resp.status == 200:
            try:
                p_obj = json.loads(resp_body.decode("utf-8"))
                by_st = _scan_all_serverless_jobs()
                q_rem = len(by_st["pending"]) + len(by_st["running"])
                exec_info = p_obj.setdefault("exec_info", {{}})
                exec_info["queue_remaining"] = int(exec_info.get("queue_remaining") or 0) + q_rem
                return web.json_response(p_obj, status=200)
            except Exception:
                pass

        if request.method == "GET" and path in ("/queue", "/api/queue") and resp.status == 200:
            try:
                q_obj = json.loads(resp_body.decode("utf-8"))
                by_st = _scan_all_serverless_jobs()
                q_run = q_obj.setdefault("queue_running", [])
                q_pend = q_obj.setdefault("queue_pending", [])
                for _jf, jdata in by_st["running"]:
                    q_run.append(_to_queue_tuple(jdata))
                for _jf, jdata in by_st["pending"]:
                    q_pend.append(_to_queue_tuple(jdata))
                return web.json_response(q_obj, status=200)
            except Exception:
                pass

        if request.method == "GET" and path == "/api/jobs" and resp.status == 200:
            try:
                j_obj = json.loads(resp_body.decode("utf-8"))
                jobs_list = j_obj.get("jobs") if isinstance(j_obj.get("jobs"), list) else []
                existing_ids = {{j.get("id") for j in jobs_list if isinstance(j, dict)}}
                status_param = request.rel_url.query.get("status", "")
                status_filter = {{s.strip().lower() for s in status_param.split(",") if s.strip()}} if status_param else None
                by_st = _scan_all_serverless_jobs()
                for st in ("running", "pending", "completed", "failed"):
                    for _jf, jdata in by_st[st]:
                        entry = _to_job_api_entry(jdata, st, include_outputs=False)
                        if entry["id"] in existing_ids:
                            continue
                        if status_filter and entry["status"] not in status_filter:
                            continue
                        jobs_list.append(entry)
                        existing_ids.add(entry["id"])
                jobs_list.sort(key=lambda x: x.get("create_time", 0), reverse=(request.rel_url.query.get("sort_order", "desc").lower() != "asc"))
                j_obj["jobs"] = jobs_list
                if isinstance(j_obj.get("pagination"), dict):
                    j_obj["pagination"]["total"] = len(jobs_list)
                return web.json_response(j_obj, status=200)
            except Exception:
                pass

        if request.method == "GET" and path.startswith("/api/jobs/") and not path.endswith("/cancel"):
            target_jid = path.rstrip("/").split("/")[-1]
            by_st = _scan_all_serverless_jobs()
            for st in ("running", "pending", "completed", "failed"):
                for _jf, jdata in by_st[st]:
                    pid = str(jdata.get("prompt_id") or jdata.get("job_id") or "")
                    if pid == target_jid:
                        return web.json_response(_to_job_api_entry(jdata, st, include_outputs=True), status=200)

        if request.method == "GET" and (path.startswith("/history") or path.startswith("/api/history")) and resp.status == 200:
            try:
                hist_obj = json.loads(resp_body.decode("utf-8"))
                if isinstance(hist_obj, dict):
                    filter_id = path.rstrip("/").split("/")[-1]
                    if filter_id in ("history", ""):
                        filter_id = None
                    by_st = _scan_all_serverless_jobs()
                    for st in ("completed", "failed"):
                        for _jf, jdata in by_st[st]:
                            pid = str(jdata.get("prompt_id") or jdata.get("job_id") or "")
                            if not pid or (filter_id and pid != filter_id) or pid in hist_obj:
                                continue
                            outputs = _build_outputs_from_expected_files(jdata) if st == "completed" else {{}}
                            ts_ms = int(jdata.get("timestamp_ms") or int(time.time() * 1000))
                            out_nid = _infer_output_node_id(jdata.get("prompt") or {{}})
                            extra_data = dict(jdata.get("extra_data") or {{}})
                            extra_data.setdefault("create_time", ts_ms)
                            hist_obj[pid] = {{
                                "prompt": [ts_ms % 100000, pid, jdata.get("prompt", {{}}), extra_data, [out_nid]],
                                "outputs": outputs,
                                "status": {{
                                    "status_str": "success" if st == "completed" else "error",
                                    "completed": st == "completed",
                                    "serverless_g4": True,
                                    "expected_files": jdata.get("expected_files", []),
                                    "error": jdata.get("error"),
                                    "messages": [
                                        ["execution_start", {{"prompt_id": pid, "timestamp": ts_ms}}],
                                        [
                                            "execution_success" if st == "completed" else "execution_error",
                                            {{"prompt_id": pid, "timestamp": int(jdata.get("completed_at_ms") or (ts_ms + 45000))}},
                                        ],
                                    ],
                                }},
                            }}
                    return web.json_response(hist_obj, status=200)
            except Exception:
                pass
        return web.Response(body=resp_body, status=resp.status, headers=out_headers)


async def on_startup(app: web.Application):
    app["client_session"] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300))
    for st in ("completed", "failed"):
        for jf in (QUEUE_ROOT / st).glob("job_*.json"):
            try:
                d = json.loads(jf.read_text(encoding="utf-8"))
                pid = str(d.get("prompt_id") or d.get("job_id") or "")
                if pid:
                    _NOTIFIED_STATES[pid] = st
            except Exception:
                pass
    app["watcher_task"] = asyncio.create_task(_watch_queue_state_loop(app))


async def on_cleanup(app: web.Application):
    watcher = app.get("watcher_task")
    if watcher:
        watcher.cancel()
    await app["client_session"].close()


def main():
    app = web.Application(client_max_size=512 * 1024 * 1024)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_route("*", "/{{tail:.*}}", handle_proxy)
    web.run_app(app, host="127.0.0.1", port=BRIDGE_PORT, print=None)


if __name__ == "__main__":
    main()
'''
    bridge_script.write_text(bridge_code, encoding="utf-8")
    sh("pkill -f '/tmp/comfy_cpu_queue_bridge.py' || true")
    bridge_log = open("/tmp/comfy_cpu_queue_bridge.log", "w")
    subprocess.Popen(
        [sys.executable, "-u", str(bridge_script)],
        stdout=bridge_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def start_caddy_auth_gateway(service_port: int = 8189, public_port: int = 18188, comfy_port: int = 8188) -> dict:
    """Launch Caddy Dual-Mode Security Layer (Basic Auth + Cookie + Bearer Token) in front of the CPU Canvas Bridge & ComfyUI."""
    work_dir = Path("/tmp/ct_comfyui")
    work_dir.mkdir(parents=True, exist_ok=True)
    creds_path = work_dir / "credentials.json"

    existing = {}
    if creds_path.exists():
        try:
            existing = json.loads(creds_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    auth_user = os.environ.get("CT_AUTH_USER") or existing.get("auth_user") or "admin"
    auth_pass = os.environ.get("CT_AUTH_PASS") or existing.get("auth_pass") or secrets.token_urlsafe(12)
    api_token = os.environ.get("CT_API_TOKEN") or existing.get("api_token") or ("sk-comfy-" + secrets.token_urlsafe(20))

    arch = {"x86_64": "amd64", "aarch64": "arm64"}.get(platform.machine(), "amd64")
    if not Path("/usr/local/bin/caddy").exists():
        print("🔒 正在安装 Caddy 安全鉴权网关 (/usr/local/bin/caddy)...", flush=True)
        tag = json.loads(
            urllib.request.urlopen("https://api.github.com/repos/caddyserver/caddy/releases/latest", timeout=30).read()
        )["tag_name"]
        tgz = "/tmp/caddy.tgz"
        url = f"https://github.com/caddyserver/caddy/releases/download/{tag}/caddy_{tag.lstrip('v')}_linux_{arch}.tar.gz"
        urllib.request.urlretrieve(url, tgz)
        subprocess.run(["tar", "-xzf", tgz, "-C", "/usr/local/bin", "caddy"], check=True)

    cookie_tok = hashlib.sha256(f"ct-comfy-v2:{auth_user}:{auth_pass}:{api_token}".encode()).hexdigest()
    hashed = subprocess.check_output(["/usr/local/bin/caddy", "hash-password", "--plaintext", auth_pass], text=True).strip()
    upstream = f"""@static_cache {{
                path /assets/* /templates/* /extensions/*
            }}
            header @static_cache Cache-Control "public, max-age=86400"
            @bridge_routes {{
                path /ws* /prompt /api/prompt /queue* /api/queue* /history* /api/history* /api/jobs* /view* /api/view* /serverless/* /api/serverless/*
            }}
            handle @bridge_routes {{
                reverse_proxy 127.0.0.1:{service_port} {{
                    header_up -Authorization
                    header_up -Origin
                    header_up -Sec-Fetch-Site
                    flush_interval -1
                }}
            }}
            handle {{
                reverse_proxy 127.0.0.1:{comfy_port} {{
                    header_up -Authorization
                    header_up -Origin
                    header_up -Sec-Fetch-Site
                    header_down -Cache-Control
                    flush_interval -1
                }}
            }}"""
    caddyfile = f"""{{
    admin off
    auto_https off
}}
:{public_port} {{
    encode zstd gzip
    @bearer header Authorization "Bearer {api_token}"
    handle @bearer {{
        route {{
            {upstream}
        }}
    }}
    @cookie header Cookie *ct_comfy_session={cookie_tok}*
    handle @cookie {{
        route {{
            {upstream}
        }}
    }}
    handle {{
        route {{
            basic_auth {{
                "{auth_user}" {hashed}
            }}
            header +Set-Cookie "ct_comfy_session={cookie_tok}; Path=/; HttpOnly; SameSite=Lax"
            {upstream}
        }}
    }}
}}
"""
    caddy_cfg = work_dir / "Caddyfile"
    caddy_cfg.write_text(caddyfile, encoding="utf-8")
    sh("pkill -f 'caddy run.*ct_comfyui' || true")
    caddy_log = open(work_dir / "caddy.log", "a")
    subprocess.Popen(
        ["/usr/local/bin/caddy", "run", "--config", str(caddy_cfg), "--adapter", "caddyfile"],
        stdout=caddy_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    creds = {
        "service_port": service_port,
        "public_port": public_port,
        "auth_user": auth_user,
        "auth_pass": auth_pass,
        "api_token": api_token,
    }
    creds_path.write_text(json.dumps(creds, indent=2), encoding="utf-8")
    os.chmod(creds_path, 0o600)
    return creds


def ensure_cloud_autonomous_watcher_on_cpu_vm(
    cpu_session: str = "c6e1c7",
    g4_session: str = "comfy-g4",
    idle_timeout: int = 600,
) -> Optional[int]:
    """Ensure `google-colab-cli` + `colab-dfs-ext` and `comfy_serverless_g4_dispatcher.py watch` are running inside the CPU Canvas VM."""
    # Ensure helper scripts exist in `/content/` (copying from `/content/drive/MyDrive/AI_Workstation/scripts/` if needed)
    for sname in ("comfy_serverless_g4_dispatcher.py", "colab_meter.py", "launch_g4_comfyui_fast.py", "comfy_gacha_and_gcp.py"):
        local_s = Path("/content") / sname
        drive_s = DRIVE_WORKSTATION / "scripts" / sname
        if not local_s.exists() and drive_s.exists():
            try:
                shutil.copy2(drive_s, local_s)
            except Exception:
                pass
        elif local_s.exists() and drive_s.parent.exists():
            try:
                if local_s.resolve() != drive_s.resolve():
                    shutil.copy2(local_s, drive_s)
            except Exception:
                pass

    if not shutil.which("colab"):
        sh("uv pip install --system -q google-colab-cli socksio || pip install -q google-colab-cli socksio")
    if not Path("/root/.colab-dfs-ext").exists():
        sh("curl -fsSL https://raw.githubusercontent.com/deluxebear/colab-dfs-ext/main/install.sh | sh")

    disp_script = Path("/content/comfy_serverless_g4_dispatcher.py")
    if not disp_script.exists():
        alt = Path(__file__).resolve().parent / "comfy_serverless_g4_dispatcher.py"
        if alt.exists():
            disp_script = alt
        else:
            return None

    target_arg_str = f"--cpu-session {cpu_session} --g4-session {g4_session} --idle-timeout {int(idle_timeout)}"
    try:
        out_procs = subprocess.getoutput("pgrep -af 'comfy_serverless_g4_dispatcher.py watch'")
        existing_match_pid = None
        for line in out_procs.splitlines():
            parts = line.strip().split(None, 1)
            if not parts or not parts[0].isdigit():
                continue
            pid_int = int(parts[0])
            if pid_int == os.getpid():
                continue
            cmdline = parts[1] if len(parts) > 1 else ""
            if target_arg_str in cmdline and existing_match_pid is None:
                existing_match_pid = pid_int
            else:
                try:
                    os.kill(pid_int, 9)
                except OSError:
                    pass
        if existing_match_pid is not None:
            print(
                f"♻️ [Cloud-Autonomous-Watcher] 复用已在运行的 G4 自动启停守护进程 (PID={existing_match_pid}, idle_timeout={idle_timeout}s)",
                flush=True,
            )
            return existing_match_pid
    except Exception:
        pass
    w_log = open(f"/tmp/serverless_g4_watcher_{cpu_session}.log", "a")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            str(disp_script),
            "watch",
            "--cpu-session",
            cpu_session,
            "--g4-session",
            g4_session,
            "--idle-timeout",
            str(int(idle_timeout)),
        ],
        stdout=w_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(
        f"🚀 [Cloud-Autonomous-Watcher] 已在 CPU 画布实例内部启动 G4 自动启停守护进程 (PID={proc.pid}, idle_timeout={idle_timeout}s)",
        flush=True,
    )
    return proc.pid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-Cost CPU ComfyUI Canvas Launcher (`--cpu`) + On-Demand G4 Serverless Queue Bridge"
    )
    parser.add_argument(
        "--exec-mode",
        choices=["g4-queue", "local-cpu"],
        default="g4-queue",
        help="Execution mode when clicking 'Queue Prompt' in Web UI: 'g4-queue' (default, saves job to Drive queue for G4 dispatcher) or 'local-cpu'",
    )
    parser.add_argument(
        "--cpu-session",
        default="c6e1c7",
        help="Colab CPU session name (default: c6e1c7)",
    )
    parser.add_argument(
        "--g4-session",
        default="comfy-g4",
        help="Colab G4 GPU session name for on-demand auto-start and idle auto-stop (default: comfy-g4)",
    )
    parser.add_argument(
        "--idle-timeout",
        type=int,
        default=600,
        help="Idle timeout in seconds before automatically stopping/releasing G4 (default: 600s = 10m)",
    )
    parser.add_argument(
        "--refresh-nodes",
        action="store_true",
        help="Force re-sync custom_nodes from Drive and rebuild custom_nodes_bundle.tar.gz",
    )
    parser.add_argument(
        "--no-tunnel",
        action="store_true",
        help="Do not start public Cloudflare tunnel (local 127.0.0.1:8188/8189 only)",
    )
    parser.add_argument(
        "--restart-tunnel",
        action="store_true",
        help="Force restart cloudflared even if an active trycloudflare.com tunnel is already running",
    )
    args = parser.parse_args()

    t_boot = time.time()
    print("=== [1/3] 检查 0 CCU CPU 环境与 20TB/30TB Google Drive 模型目录 ===", flush=True)
    if not (DRIVE_COMFY / "models").exists():
        raise RuntimeError(f"❌ 未检测到云盘模型库: {DRIVE_COMFY}/models，请先执行 `colab dfs mount` 挂载 Google Drive！")

    drive_script_dst = DRIVE_WORKSTATION / "scripts/launch_cpu_canvas_comfyui.py"
    try:
        drive_script_dst.parent.mkdir(parents=True, exist_ok=True)
        if Path(__file__).resolve() != drive_script_dst.resolve():
            shutil.copy2(__file__, drive_script_dst)
            print(f"✅ 已自动同步 0 CCU 独立画布启动器至云盘: {drive_script_dst}", flush=True)
    except Exception as exc:
        print(f"⚠️ 同步云盘脚本跳过: {exc}", flush=True)

    print("=== [2/3] 初始化本地 SSD ComfyUI 核心、插件镜像与云盘软链 (0 模型加载) ===", flush=True)
    setup_cpu_comfy_and_nodes(refresh_nodes=args.refresh_nodes)

    print(f"=== [3/3] 启动 ComfyUI (--cpu) + G4 弹性队列桥 ({args.exec_mode}) + Caddy 鉴权网关 + Cloudflare 隧道 ===", flush=True)
    sh("pkill -f 'python3 main.py' || true")

    comfy_log = open("/tmp/comfyui_cpu.log", "w")
    subprocess.Popen(
        [
            sys.executable,
            "main.py",
            "--listen",
            "127.0.0.1",
            "--port",
            "8188",
            "--cpu",
            "--enable-compress-response-body",
        ],
        cwd=str(LOCAL_COMFY),
        stdout=comfy_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    write_and_start_queue_bridge(
        comfy_port=8188,
        bridge_port=8189,
        exec_mode=args.exec_mode,
        queue_dir=SERVERLESS_QUEUE_DIR,
    )

    cloud_watcher_pid = None
    if args.exec_mode == "g4-queue":
        cloud_watcher_pid = ensure_cloud_autonomous_watcher_on_cpu_vm(
            cpu_session=args.cpu_session,
            g4_session=args.g4_session,
            idle_timeout=args.idle_timeout,
        )

    auth_creds = None
    cf_url = None
    if not args.no_tunnel:
        auth_creds = start_caddy_auth_gateway(service_port=8189, public_port=18188)
        if not Path("/usr/local/bin/cloudflared").exists():
            sh(
                "wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 "
                "-O /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared"
            )
        # Preserve an already-running cloudflared tunnel URL unless --restart-tunnel is passed
        cf_running = subprocess.run("pgrep -f 'cloudflared tunnel'", shell=True, capture_output=True).returncode == 0
        if cf_running and not args.restart_tunnel and Path("/tmp/cloudflared_cpu.log").exists():
            txt = Path("/tmp/cloudflared_cpu.log").read_text(errors="ignore")
            m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", txt)
            if m:
                cf_url = m.group(0)
                print(f"♻️ 复用当前活跃的 Cloudflare 隧道地址: {cf_url}", flush=True)
        if not cf_url:
            sh("pkill -f 'cloudflared tunnel' || true")
            cf_log = open("/tmp/cloudflared_cpu.log", "w")
            subprocess.Popen(
                ["/usr/local/bin/cloudflared", "tunnel", "--protocol", "http2", "--url", "http://127.0.0.1:18188"],
                stdout=cf_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

    comfy_ready = False
    for _ in range(160):
        time.sleep(1.5)
        if not args.no_tunnel and not cf_url and Path("/tmp/cloudflared_cpu.log").exists():
            txt = Path("/tmp/cloudflared_cpu.log").read_text(errors="ignore")
            m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", txt)
            if m:
                cf_url = m.group(0)
        if not comfy_ready and Path("/tmp/comfyui_cpu.log").exists():
            if "To see the GUI go to:" in Path("/tmp/comfyui_cpu.log").read_text(errors="ignore"):
                comfy_ready = True
        if not comfy_ready:
            try:
                if urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=1).status == 200:
                    comfy_ready = True
            except Exception:
                pass
        if (cf_url or args.no_tunnel) and comfy_ready:
            break

    state_summary = {
        "mode": "cpu_canvas",
        "exec_mode": args.exec_mode,
        "comfy_ready": comfy_ready,
        "boot_seconds": round(time.time() - t_boot, 1),
        "public_url": cf_url,
        "queue_dir": str(SERVERLESS_QUEUE_DIR),
        "idle_timeout_seconds": args.idle_timeout,
        "cloud_watcher_pid": cloud_watcher_pid,
        "auth_user": auth_creds["auth_user"] if auth_creds else None,
        "auth_pass": auth_creds["auth_pass"] if auth_creds else None,
        "api_token": auth_creds["api_token"] if auth_creds else None,
    }
    Path("/tmp/ct_comfyui/canvas_state.json").write_text(json.dumps(state_summary, indent=2), encoding="utf-8")
    os.chmod("/tmp/ct_comfyui/canvas_state.json", 0o600)

    print("\n" + "=" * 72, flush=True)
    print(
        f"🎨 ComfyUI 零成本独立画布 (CPU 0.00 CCU/h | 算力解耦模式={args.exec_mode}) 已就绪: {comfy_ready} | 耗时: {time.time() - t_boot:.1f}s",
        flush=True,
    )
    if auth_creds:
        print(f"🌐 您的 0 CCU ComfyUI 独立画布地址 (Caddy 双模鉴权保护): {cf_url}", flush=True)
        print(f"🔑 网页登录账号: {auth_creds['auth_user']} | 密码: {auth_creds['auth_pass']}", flush=True)
        print(f"🛡️ API Bearer Token: {auth_creds['api_token']}", flush=True)
        print(f"📥 G4 弹性任务队列目录: {SERVERLESS_QUEUE_DIR}/pending", flush=True)
    else:
        print("🔒 本地 API 监听模式: ComfyUI=127.0.0.1:8188 | QueueBridge=127.0.0.1:8189", flush=True)
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()

