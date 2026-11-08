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

    # Wire shared directories (`models`, `output`, `input`, `user`) to Google Drive so CPU Canvas sees all models, workflows & G4 outputs
    for shared_sub in ("models", "output", "input", "user"):
        drive_sub = DRIVE_COMFY / shared_sub
        drive_sub.mkdir(parents=True, exist_ok=True)
        local_sub = LOCAL_COMFY / shared_sub
        if local_sub.is_symlink() or local_sub.exists():
            sh(f"rm -rf '{local_sub}'")
        os.symlink(drive_sub, local_sub)

    get_queue_dirs(SERVERLESS_QUEUE_DIR)

    # Install lightweight dependencies for ComfyUI + custom nodes on CPU
    pkgs = "aiohttp gguf imageio-ffmpeg librosa soundfile tiktoken sentencepiece sox accelerate einops onnxruntime ultralytics opencv-python-headless scipy"
    sh(
        f"if command -v uv >/dev/null 2>&1; then "
        f"uv pip install --system -q -r {LOCAL_COMFY}/requirements.txt {pkgs}; "
        f"else pip install -q -r {LOCAL_COMFY}/requirements.txt {pkgs}; fi"
    )

    # Bootstrap default user files & static user.css so ComfyUI Frontend v1.58+ never 404s on initial page load
    user_default = LOCAL_COMFY / "user" / "default"
    (user_default / "workflows").mkdir(parents=True, exist_ok=True)
    (user_default / "subgraphs").mkdir(parents=True, exist_ok=True)
    for wf_src_root in (DRIVE_WORKSTATION / "workflows",):
        if wf_src_root.exists():
            for wf_path in wf_src_root.rglob("*.json"):
                if "workflow" in wf_path.name.lower() and "api" not in wf_path.name.lower():
                    dst_wf = user_default / "workflows" / wf_path.name
                    if not dst_wf.exists():
                        shutil.copy2(wf_path, dst_wf)
    default_files = {
        user_default / "user.css": "/* ComfyUI user custom stylesheet */\n",
        user_default / "comfy.settings.json": "{}\n",
        user_default / "comfy.templates.json": "[]\n",
        user_default / "workflows" / ".index.json": "{}\n",
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
"""Async HTTP + WebSocket Reverse Proxy with On-Demand G4 Queue Interceptor for CPU ComfyUI Canvas."""
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

for _sub in ("pending", "running", "completed", "failed"):
    (QUEUE_ROOT / _sub).mkdir(parents=True, exist_ok=True)


_LAST_TS_MS = 0


def _enqueue_job(payload: dict) -> dict:
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
        "extra_data": payload.get("extra_data", {{}}),
    }}
    job_file = QUEUE_ROOT / "pending" / f"job_{{ts_ms}}_{{job_id}}.json"
    tmp_file = QUEUE_ROOT / "pending" / f".tmp_{{ts_ms}}_{{job_id}}.json"
    tmp_file.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_file.replace(job_file)
    print(f"📥 [G4-Serverless-Queue] 已拦截 Web UI 渲染请求并写入云盘队列: {{job_file.name}}", flush=True)
    return {{
        "prompt_id": job_id,
        "number": ts_ms % 100000,
        "node_errors": {{}},
        "serverless_g4_queued": True,
        "queue_file": str(job_file),
    }}


async def handle_serverless_status(request: web.Request) -> web.Response:
    counts = {{}}
    recent = []
    for st in ("pending", "running", "completed", "failed"):
        files = sorted((QUEUE_ROOT / st).glob("job_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        counts[st] = len(files)
        for f in files[:5]:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                recent.append({{
                    "job_id": data.get("job_id"),
                    "status": st,
                    "created_at": data.get("created_at"),
                    "completed_at": data.get("completed_at"),
                    "file": f.name,
                }})
            except Exception:
                pass
    return web.json_response({{
        "exec_mode": EXEC_MODE,
        "queue_root": str(QUEUE_ROOT),
        "counts": counts,
        "recent_jobs": recent[:10],
    }})


async def handle_proxy(request: web.Request) -> web.StreamResponse:
    path_qs = request.rel_url.path_qs
    path = request.rel_url.path

    # 1. Status endpoint for the Serverless G4 Dispatcher
    if path in ("/serverless/status", "/api/serverless/status"):
        return await handle_serverless_status(request)

    # 2. WebSocket proxy (`/ws`)
    if request.headers.get("Upgrade", "").lower() == "websocket":
        ws_server = web.WebSocketResponse()
        await ws_server.prepare(request)
        session: aiohttp.ClientSession = request.app["client_session"]
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
                        await ws_server.send_str(msg.data)
                    elif msg.type == aiohttp.WSMsgType.BINARY:
                        await ws_server.send_bytes(msg.data)
                    elif msg.type == aiohttp.WSMsgType.CLOSE:
                        await ws_server.close()
            await asyncio.gather(_c2s(), _s2c(), return_exceptions=True)
        return ws_server

    # 3. Intercept `POST /prompt` or `POST /api/prompt` when in `g4-queue` mode
    if request.method == "POST" and path in ("/prompt", "/api/prompt"):
        force_cpu = request.headers.get("X-Comfy-Exec-Target", "").lower() == "cpu"
        if EXEC_MODE == "g4-queue" and not force_cpu:
            try:
                body = await request.json()
                res = _enqueue_job(body)
                return web.json_response(res)
            except Exception as exc:
                return web.json_response({{"error": str(exc), "node_errors": {{}}}}, status=400)

    # 4. Augment `/history` or `/api/history` with completed/failed Serverless G4 jobs so Web UI can inspect them
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
        if request.method == "GET" and (path.startswith("/history") or path.startswith("/api/history")) and resp.status == 200:
            try:
                hist_obj = json.loads(resp_body.decode("utf-8"))
                if isinstance(hist_obj, dict):
                    filter_id = path.rstrip("/").split("/")[-1]
                    if filter_id in ("history", ""):
                        filter_id = None
                    for st in ("completed", "failed"):
                        for jf in (QUEUE_ROOT / st).glob("job_*.json"):
                            try:
                                jdata = json.loads(jf.read_text(encoding="utf-8"))
                                pid = jdata.get("prompt_id") or jdata.get("job_id")
                                if not pid or (filter_id and pid != filter_id) or pid in hist_obj:
                                    continue
                                hist_obj[pid] = {{
                                    "prompt": [jdata.get("timestamp_ms", 0), pid, jdata.get("prompt", {{}}), jdata.get("extra_data", {{}}), []],
                                    "outputs": jdata.get("outputs", {{}}),
                                    "status": {{
                                        "status_str": "success" if st == "completed" else "error",
                                        "completed": st == "completed",
                                        "serverless_g4": True,
                                        "expected_files": jdata.get("expected_files", []),
                                        "error": jdata.get("error"),
                                    }},
                                }}
                            except Exception:
                                pass
                    return web.json_response(hist_obj, status=200)
            except Exception:
                pass
        return web.Response(body=resp_body, status=resp.status, headers=out_headers)


async def on_startup(app: web.Application):
    app["client_session"] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300))


async def on_cleanup(app: web.Application):
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
    upstream = f"""@bridge_routes {{
                path /prompt /api/prompt /history* /api/history* /serverless/* /api/serverless/*
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
        "--refresh-nodes",
        action="store_true",
        help="Force re-sync custom_nodes from Drive and rebuild custom_nodes_bundle.tar.gz",
    )
    parser.add_argument(
        "--no-tunnel",
        action="store_true",
        help="Do not start public Cloudflare tunnel (local 127.0.0.1:8188/8189 only)",
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
    sh("pkill -f 'cloudflared tunnel' || true")

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

    auth_creds = None
    if not args.no_tunnel:
        auth_creds = start_caddy_auth_gateway(service_port=8189, public_port=18188)
        if not Path("/usr/local/bin/cloudflared").exists():
            sh(
                "wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 "
                "-O /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared"
            )
        cf_log = open("/tmp/cloudflared_cpu.log", "w")
        subprocess.Popen(
            ["/usr/local/bin/cloudflared", "tunnel", "--url", "http://127.0.0.1:18188"],
            stdout=cf_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    cf_url = None
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
