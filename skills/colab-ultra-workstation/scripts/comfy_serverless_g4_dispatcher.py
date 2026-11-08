#!/usr/bin/env python3
"""Decoupled Control Plane & On-Demand Serverless G4 Compute Dispatcher for ComfyUI.

Architecture Overview (画布与算力解耦的智能弹性网关):
=====================================================
1. **Zero-Cost Control Plane (`canvas` subcommand, 0.00 CCU/h)**:
   - Runs ComfyUI Web UI (`--cpu`) on a free Colab CPU session (e.g., `-s c6e1c7`).
   - Mounts the 20TB/30TB Google Drive so all checkpoints, custom nodes, LoRAs, VAEs, and past outputs
     are visible in the Web UI with **0.00 CCU/h GPU burn** while you design workflows, edit prompts, or inspect videos.
   - Protected by the 192-bit Caddy Dual-Mode Security Layer + Cloudflare Tunnel.
   - Intercepts `POST /prompt` ("Queue Prompt" in the Web UI) and writes the job payload into
     `/content/drive/MyDrive/AI_Workstation/ComfyUI/serverless_queue/pending/`.

2. **On-Demand Serverless G4 Compute Engine (`dispatch`, `watch`, `stop-g4` subcommands)**:
   - **Lazy G4 Provisioning**: Never boots a G4 (`10.05 CCU/h`) until a workflow/gacha job is actually submitted
     (either via the CPU Web UI queue or directly via `dispatch --workflow ...`).
   - **Automated Fast-Boot & Warm Reuse**:
     - If G4 session (`--g4-session comfy-g4`) is not running, automatically executes:
       1) `colab_meter.py start -s <g4_session> -g G4`
       2) `colab new -s <g4_session> --gpu G4`
       3) `colab dfs mount -s <g4_session>` (mounts shared 30TB Google Drive)
       4) Launches `launch_g4_comfyui_fast.py --profile scheme_a` (Tri-Parallel NVMe Fast Boot, ~75s cold start)
     - If G4 session is already hot, reuses it immediately (**0.0s model reload**)!
   - **Batch Gacha + 2x2 Grid + GCS Sync**:
     - Executes single workflows or N-seed Gacha batches (`--gacha-runs 4`), builds the 2x2 comparison grid MP4,
       writes all outputs directly to `/content/drive/MyDrive/AI_Workstation/ComfyUI/output/` (immediately visible
       in the CPU Canvas Web UI!), and optionally syncs to `gs://<bucket>/...`.
   - **Drive Flush Verification & Auto-Stop Idle Timeout (`--idle-timeout 600` or `--auto-stop-g4`)**:
     - Before stopping the G4 instance, verifies via the Google Drive v3 API (`files.list`) that all newly generated
       `.mp4` / `.pt` files have flushed to cloud storage (`size > 0`).
     - Automatically stops/unassigns the G4 GPU session (`colab_meter.py stop -s <g4_session>`) either immediately
       after batch completion (`--auto-stop-g4`) or after `--idle-timeout` seconds of inactivity (default: 600s = 10 mins),
       printing the standardized Dual-Track CCU Settlement Report!
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import comfy_gacha_and_gcp as gacha  # noqa: E402
import launch_cpu_canvas_comfyui as cpu_canvas  # noqa: E402

COLAB_BIN = os.environ.get("COLAB_BIN") or shutil.which("colab") or str(Path.home() / ".local/bin/colab")
METER_SCRIPT = SCRIPT_DIR / "colab_meter.py"
CPU_LAUNCHER_SCRIPT = SCRIPT_DIR / "launch_cpu_canvas_comfyui.py"
G4_LAUNCHER_SCRIPT = SCRIPT_DIR / "launch_g4_comfyui_fast.py"
GACHA_SCRIPT = SCRIPT_DIR / "comfy_gacha_and_gcp.py"

SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,62}$")
GPU_VARIANT_SET = {"G4", "L4", "T4", "A100", "H100"}


def validate_session_name(session: str) -> str:
    """Validate a Colab session name to prevent shell/command injection."""
    cleaned = (session or "").strip()
    if not SESSION_ID_RE.match(cleaned):
        raise ValueError(
            f"Invalid session name '{session}'. Allowed characters: alphanumeric, '_', '-' (1..63 chars)."
        )
    return cleaned


def validate_gpu_variant(gpu: str) -> str:
    """Validate GPU accelerator name."""
    cleaned = (gpu or "G4").strip().upper()
    if cleaned not in GPU_VARIANT_SET:
        raise ValueError(f"Unsupported GPU variant '{gpu}'. Supported: {sorted(GPU_VARIANT_SET)}")
    return cleaned


class ServerlessG4LifecycleController:
    """Manages on-demand G4 GPU provisioning, job dispatch, Drive flush verification, and idle auto-stop."""

    def __init__(
        self,
        g4_session: str = "comfy-g4",
        cpu_session: str = "c6e1c7",
        gpu_variant: str = "G4",
        profile: str = "scheme_a",
        idle_timeout_sec: int = 600,
        auto_stop_immediately: bool = False,
        runner: Optional[Callable[[List[str], Optional[str], float], Tuple[int, str]]] = None,
    ) -> None:
        self.g4_session = validate_session_name(g4_session)
        self.cpu_session = validate_session_name(cpu_session)
        self.gpu_variant = validate_gpu_variant(gpu_variant)
        if profile not in ("scheme_a", "scheme_b", "all"):
            raise ValueError(f"Invalid profile: {profile}")
        self.profile = profile
        if idle_timeout_sec < 0:
            raise ValueError(f"idle_timeout_sec must be >= 0, got {idle_timeout_sec}")
        self.idle_timeout_sec = int(idle_timeout_sec)
        self.auto_stop_immediately = bool(auto_stop_immediately)
        self._runner = runner or self._default_runner
        self.g4_active: bool = False
        self.last_activity_ts: float = 0.0

    @staticmethod
    def _default_runner(cmd: List[str], stdin_text: Optional[str] = None, timeout: float = 120.0) -> Tuple[int, str]:
        proc = subprocess.run(
            cmd,
            input=stdin_text,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, out.strip()

    def colab_exec(self, session: str, python_code: str, timeout: float = 45.0) -> Tuple[int, str]:
        """Execute a short non-blocking Python snippet on a Colab session."""
        sess = validate_session_name(session)
        return self._runner([COLAB_BIN, "exec", "-s", sess, "--timeout", str(timeout)], python_code, timeout + 15.0)

    def colab_upload(self, session: str, local_path: Path, remote_path: str, timeout: float = 60.0) -> None:
        sess = validate_session_name(session)
        rc, out = self._runner([COLAB_BIN, "upload", "-s", sess, str(local_path), remote_path], None, timeout)
        if rc != 0:
            raise RuntimeError(f"Failed to upload {local_path} to [{sess}]:{remote_path}: {out}")

    def is_session_alive(self, session: str) -> bool:
        sess = validate_session_name(session)
        rc, out = self._runner([COLAB_BIN, "sessions"], None, 25.0)
        if rc != 0:
            return False
        return f"[{sess}]" in out

    def is_g4_comfy_ready(self) -> bool:
        """Check if G4 session exists and ComfyUI (`127.0.0.1:8188`) is responding."""
        if not self.is_session_alive(self.g4_session):
            self.g4_active = False
            return False
        probe_code = (
            "import urllib.request\n"
            "try:\n"
            "    r = urllib.request.urlopen('http://127.0.0.1:8188/system_stats', timeout=3)\n"
            "    print('COMFY_READY_OK' if r.status == 200 else 'COMFY_NOT_READY')\n"
            "except Exception:\n"
            "    print('COMFY_NOT_READY')\n"
        )
        rc, out = self.colab_exec(self.g4_session, probe_code, timeout=15.0)
        ready = rc == 0 and "COMFY_READY_OK" in out
        self.g4_active = ready
        return ready

    def ensure_g4_ready(self, boot_timeout_sec: int = 420) -> Dict[str, Any]:
        """Spin up or reuse the G4 GPU session, mount 30TB Drive, and fast-boot ComfyUI (`--highvram --disable-mmap`)."""
        if self.is_g4_comfy_ready():
            self.last_activity_ts = time.time()
            print(f"⚡ [G4-Reuse] G4 算力实例 [{self.g4_session}] 已处于热显存就绪状态 (0.0s 重载)！", flush=True)
            return {"session": self.g4_session, "reused": True}

        t0 = time.time()
        if not self.is_session_alive(self.g4_session):
            print(f"🚀 [G4-Provision] 正在按需拉起 {self.gpu_variant} (96GB Blackwell) 算力实例: [{self.g4_session}] ...", flush=True)
            self._runner(
                [sys.executable, str(METER_SCRIPT), "start", "-s", self.g4_session, "-g", self.gpu_variant],
                None,
                30.0,
            )
            rc, out = self._runner(
                [COLAB_BIN, "new", "-s", self.g4_session, "--gpu", self.gpu_variant],
                None,
                180.0,
            )
            if rc != 0:
                raise RuntimeError(f"❌ 无法分配 {self.gpu_variant} 实例 [{self.g4_session}]: {out}")

        print(f"☁️ [G4-Mount] 正在为 [{self.g4_session}] 零弹窗挂载 20TB/30TB Google Drive...", flush=True)
        rc, out = self._runner([COLAB_BIN, "dfs", "mount", "-s", self.g4_session], None, 150.0)
        if rc != 0:
            raise RuntimeError(f"❌ [{self.g4_session}] 挂载 Google Drive 失败: {out}")

        # Upload latest fast launcher and gacha engine
        self.colab_upload(self.g4_session, G4_LAUNCHER_SCRIPT, "/content/launch_g4_comfyui_fast.py")
        self.colab_upload(self.g4_session, GACHA_SCRIPT, "/content/comfy_gacha_and_gcp.py")

        print(f"🔥 [G4-FastBoot] 启动三路全并发 NVMe 预热 (profile={self.profile})...", flush=True)
        start_code = (
            "import subprocess\n"
            f"subprocess.Popen('nohup python3 -u /content/launch_g4_comfyui_fast.py --profile {self.profile} "
            "> /tmp/g4_fastboot.log 2>&1 &', shell=True)\n"
            "print('G4_FASTBOOT_STARTED')\n"
        )
        self.colab_exec(self.g4_session, start_code, timeout=20.0)

        deadline = time.time() + boot_timeout_sec
        while time.time() < deadline:
            time.sleep(8)
            if self.is_g4_comfy_ready():
                self.g4_active = True
                self.last_activity_ts = time.time()
                boot_dt = time.time() - t0
                print(f"✅ [G4-Ready] G4 算力引擎 [{self.g4_session}] 启动就绪！总耗时: {boot_dt:.1f}s", flush=True)
                return {"session": self.g4_session, "reused": False, "boot_seconds": round(boot_dt, 1)}

            # Print last 3 lines of progress log
            _, log_tail = self.colab_exec(
                self.g4_session,
                "import subprocess; print(subprocess.getoutput('tail -n 3 /tmp/g4_fastboot.log'))",
                timeout=15.0,
            )
            if log_tail:
                last_line = [ln for ln in log_tail.splitlines() if ln.strip()][-1:]
                if last_line:
                    print(f"  ⏳ [G4 预热进度] {last_line[0]}", flush=True)

        raise TimeoutError(f"❌ G4 实例 [{self.g4_session}] 在 {boot_timeout_sec}s 内未能完成预热启动。")

    def verify_drive_flush_before_stop(self, expected_filenames: Optional[List[str]] = None) -> bool:
        """Verify via Google Drive v3 API that generated outputs have flushed from FUSE cache to cloud storage."""
        if not self.is_session_alive(self.g4_session):
            return True
        names_json = json.dumps(expected_filenames or [])
        verify_code = f'''import json, time, urllib.parse, urllib.request
expected = set({names_json})
try:
    req_tok = urllib.request.Request(
        "http://127.0.0.1:8012/computeMetadata/v1/instance/service-accounts/default/token",
        headers={{"Metadata-Flavor": "Google"}},
    )
    tok = json.loads(urllib.request.urlopen(req_tok, timeout=5).read().decode())["access_token"]
    params = urllib.parse.urlencode({{
        "q": "trashed = false and (name contains '.mp4' or name contains '.pt' or name contains '.png')",
        "fields": "files(name,size)",
        "pageSize": 100,
    }})
    req_files = urllib.request.Request(
        f"https://www.googleapis.com/drive/v3/files?{{params}}",
        headers={{"Authorization": f"Bearer {{tok}}"}},
    )
    files = json.loads(urllib.request.urlopen(req_files, timeout=10).read().decode()).get("files", [])
    cloud_names = {{f["name"] for f in files if int(f.get("size", 0)) > 0}}
    missing = [n for n in expected if n not in cloud_names]
    print(json.dumps({{"ok": len(missing) == 0, "missing": missing, "cloud_count": len(cloud_names)}}))
except Exception as e:
    print(json.dumps({{"ok": False, "error": str(e)}}))
'''
        for _ in range(6):
            rc, out = self.colab_exec(self.g4_session, verify_code, timeout=20.0)
            if rc == 0 and out:
                for line in reversed(out.splitlines()):
                    line = line.strip()
                    if line.startswith("{") and line.endswith("}"):
                        try:
                            data = json.loads(line)
                            if data.get("ok"):
                                return True
                        except Exception:
                            pass
            time.sleep(3)
        return False

    def stop_g4_session(self, reason: str = "idle_timeout", expected_files: Optional[List[str]] = None) -> Dict[str, Any]:
        """Verify Drive sync, unassign ONLY the G4 session, and output Dual-Track CCU Settlement Report."""
        print(f"🛑 [G4-AutoStop] 触发 G4 算力释放 (原因: {reason})，正在核验云盘落盘状态...", flush=True)
        self.verify_drive_flush_before_stop(expected_filenames=expected_files)
        rc, out = self._runner(
            [sys.executable, str(METER_SCRIPT), "stop", "-s", self.g4_session],
            None,
            45.0,
        )
        self.g4_active = False
        print(out, flush=True)
        return {"stopped_session": self.g4_session, "reason": reason, "meter_output": out, "rc": rc}

    def check_idle_and_stop_if_needed(self, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Check if G4 has exceeded `idle_timeout_sec` and stop it if so."""
        if not self.g4_active:
            return None
        cur = now if now is not None else time.time()
        idle_for = cur - self.last_activity_ts
        if self.auto_stop_immediately or (self.idle_timeout_sec >= 0 and idle_for >= self.idle_timeout_sec):
            reason = "auto_stop_after_batch" if self.auto_stop_immediately else f"idle_{int(idle_for)}s_exceeded_{self.idle_timeout_sec}s"
            return self.stop_g4_session(reason=reason)
        return None

    def dispatch_workflow_on_g4(
        self,
        workflow_path: Path,
        gacha_runs: int = 1,
        seed_base: int = 2026110801,
        seeds_str: Optional[str] = None,
        prefix: str = "ServerlessG4",
        target_seg_id: Optional[int] = None,
        gcs_bucket: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ensure G4 is booted, upload the workflow JSON, run single or multi-seed Gacha on G4, and manage auto-stop."""
        wf_dict = json.loads(workflow_path.read_text(encoding="utf-8"))
        gacha.ensure_api_workflow_format(wf_dict, source_hint=str(workflow_path))

        self.ensure_g4_ready()
        remote_wf = f"/tmp/job_{int(time.time())}_{workflow_path.name}"
        self.colab_upload(self.g4_session, workflow_path, remote_wf)
        self.colab_upload(self.g4_session, GACHA_SCRIPT, "/content/comfy_gacha_and_gcp.py")

        cmd_parts = [
            "python3",
            "-u",
            "/content/comfy_gacha_and_gcp.py",
            "gacha",
            "--workflow",
            remote_wf,
            "--gacha-runs",
            str(max(1, int(gacha_runs))),
            "--seed-base",
            str(int(seed_base)),
            "--prefix",
            prefix,
        ]
        if seeds_str:
            cmd_parts.extend(["--seeds", seeds_str])
        if target_seg_id is not None:
            cmd_parts.extend(["--target-seg", str(int(target_seg_id))])
        if gcs_bucket:
            norm_gcs = gacha.validate_gcs_uri(gcs_bucket)
            cmd_parts.extend(["--gcs-bucket", norm_gcs])

        remote_cmd_str = " ".join(json.dumps(p) for p in cmd_parts)
        launch_code = (
            "import subprocess\n"
            "open('/tmp/g4_job.done', 'w').close()\n"
            "import os; os.remove('/tmp/g4_job.done')\n"
            f"subprocess.Popen('nohup bash -c \"{remote_cmd_str} > /tmp/g4_job.log 2>&1; echo $? > /tmp/g4_job.done\" &', shell=True)\n"
            "print('JOB_DISPATCHED')\n"
        )
        self.colab_exec(self.g4_session, launch_code, timeout=20.0)
        print(f"🎬 [G4-Execute] 已派发渲染任务至 [{self.g4_session}] (gacha_runs={gacha_runs}, prefix={prefix})...", flush=True)

        t0 = time.time()
        while True:
            time.sleep(6)
            self.last_activity_ts = time.time()
            poll_code = (
                "import os, subprocess\n"
                "done = os.path.exists('/tmp/g4_job.done')\n"
                "rc = open('/tmp/g4_job.done').read().strip() if done else ''\n"
                "tail = subprocess.getoutput('tail -n 15 /tmp/g4_job.log')\n"
                "import json; print(json.dumps({'done': done, 'rc': rc, 'tail': tail}))\n"
            )
            rc, out = self.colab_exec(self.g4_session, poll_code, timeout=20.0)
            if rc == 0 and out:
                for ln in reversed(out.splitlines()):
                    ln = ln.strip()
                    if ln.startswith("{") and ln.endswith("}"):
                        state_obj = json.loads(ln)
                        if state_obj.get("done"):
                            job_rc = int(state_obj.get("rc") or 0)
                            tail_txt = state_obj.get("tail", "")
                            if job_rc != 0:
                                raise RuntimeError(f"❌ G4 渲染任务失败 (rc={job_rc}):\n{tail_txt}")
                            print(f"🎉 [G4-Complete] 渲染完成 (耗时 {time.time() - t0:.1f}s)！产出已写入共享云盘 output/。", flush=True)
                            result_summary: Dict[str, Any] = {
                                "status": "completed",
                                "g4_session": self.g4_session,
                                "elapsed_seconds": round(time.time() - t0, 1),
                                "log_tail": tail_txt,
                            }
                            if self.auto_stop_immediately:
                                stop_info = self.stop_g4_session(reason="auto_stop_after_batch")
                                result_summary["auto_stopped"] = stop_info
                            return result_summary


def launch_cpu_canvas_on_session(
    cpu_session: str = "c6e1c7",
    exec_mode: str = "g4-queue",
    refresh_nodes: bool = False,
) -> Dict[str, Any]:
    """Upload and start `launch_cpu_canvas_comfyui.py` on the free CPU Colab session and return connection info."""
    sess = validate_session_name(cpu_session)
    ctrl = ServerlessG4LifecycleController(cpu_session=sess)
    if not ctrl.is_session_alive(sess):
        print(f"🆕 正在创建免费 0.00 CCU/h CPU 画布会话 [{sess}]...", flush=True)
        ctrl._runner([COLAB_BIN, "new", "-s", sess], None, 120.0)
        ctrl._runner([COLAB_BIN, "dfs", "mount", "-s", sess], None, 120.0)

    ctrl.colab_upload(sess, CPU_LAUNCHER_SCRIPT, "/content/launch_cpu_canvas_comfyui.py")
    flags = f"--exec-mode {exec_mode}" + (" --refresh-nodes" if refresh_nodes else "")
    start_code = (
        "import subprocess\n"
        f"subprocess.Popen('nohup python3 -u /content/launch_cpu_canvas_comfyui.py {flags} > /tmp/cpu_canvas_boot.log 2>&1 &', shell=True)\n"
        "print('CPU_CANVAS_BOOT_STARTED')\n"
    )
    ctrl.colab_exec(sess, start_code, timeout=20.0)
    print(f"⏳ 正在启动 [{sess}] 上的 0.00 CCU/h ComfyUI 独立画布与 Caddy 安全网关...", flush=True)

    for _ in range(90):
        time.sleep(3)
        probe = (
            "import json, os, urllib.request\n"
            "p = '/tmp/ct_comfyui/canvas_state.json'\n"
            "if os.path.exists(p):\n"
            "    d = json.load(open(p))\n"
            "    if not d.get('comfy_ready'):\n"
            "        try:\n"
            "            if urllib.request.urlopen('http://127.0.0.1:8188/system_stats', timeout=2).status == 200:\n"
            "                d['comfy_ready'] = True\n"
            "                json.dump(d, open(p, 'w'), indent=2)\n"
            "        except Exception:\n"
            "            pass\n"
            "    if d.get('comfy_ready') and d.get('public_url'):\n"
            "        print(json.dumps(d))\n"
        )
        rc, out = ctrl.colab_exec(sess, probe, timeout=15.0)
        if rc == 0 and out:
            for ln in reversed(out.splitlines()):
                ln = ln.strip()
                if ln.startswith("{") and ln.endswith("}"):
                    info = json.loads(ln)
                    print("\n" + "=" * 72, flush=True)
                    print(f"🎨 [0.00 CCU/h 独立画布就绪] 会话: [{sess}] | 模式: {info.get('exec_mode')}", flush=True)
                    print(f"🌐 专属 ComfyUI 网页画布地址: {info.get('public_url')}", flush=True)
                    print(f"🔑 登录账号: {info.get('auth_user')} | 密码: {info.get('auth_pass')}", flush=True)
                    print(f"🛡️ API Bearer Token: {info.get('api_token')}", flush=True)
                    print("=" * 72, flush=True)
                    return info

    _, tail = ctrl.colab_exec(
        sess,
        "import subprocess; print(subprocess.getoutput('tail -n 30 /tmp/cpu_canvas_boot.log'))",
        timeout=15.0,
    )
    raise RuntimeError(f"❌ CPU 独立画布在预期时间内未就绪，最新日志:\n{tail}")


def watch_serverless_queue_loop(
    ctrl: ServerlessG4LifecycleController,
    poll_interval_sec: int = 8,
    max_polls: Optional[int] = None,
) -> None:
    """Poll the CPU session's `/content/drive/MyDrive/AI_Workstation/ComfyUI/serverless_queue/pending/` and dispatch to G4 on demand."""
    print(
        f"👀 [Serverless-Watcher] 正在监听 [{ctrl.cpu_session}] 云盘任务队列 | "
        f"G4 实例: [{ctrl.g4_session}] | 空闲自动关机阈值: {ctrl.idle_timeout_sec}s",
        flush=True,
    )
    polls = 0
    while True:
        polls += 1
        check_code = '''import json
from pathlib import Path
q_pending = Path("/content/drive/MyDrive/AI_Workstation/ComfyUI/serverless_queue/pending")
q_running = Path("/content/drive/MyDrive/AI_Workstation/ComfyUI/serverless_queue/running")
q_pending.mkdir(parents=True, exist_ok=True)
q_running.mkdir(parents=True, exist_ok=True)
jobs = sorted(q_pending.glob("job_*.json"), key=lambda p: p.stat().st_mtime)
if not jobs:
    print(json.dumps({"claimed": None}))
else:
    target = jobs[0]
    dst = q_running / target.name
    target.replace(dst)
    data = json.loads(dst.read_text(encoding="utf-8"))
    data["running_file"] = str(dst)
    print(json.dumps({"claimed": data}))
'''
        rc, out = ctrl.colab_exec(ctrl.cpu_session, check_code, timeout=20.0)
        claimed_job = None
        if rc == 0 and out:
            for ln in reversed(out.splitlines()):
                ln = ln.strip()
                if ln.startswith("{") and ln.endswith("}"):
                    try:
                        claimed_job = json.loads(ln).get("claimed")
                    except Exception:
                        pass
                    break

        if claimed_job:
            job_id = claimed_job.get("job_id", "unknown")
            gacha_runs = int(claimed_job.get("gacha_runs") or 1)
            prefix = claimed_job.get("prefix") or f"WebUI_{job_id[:8]}"
            running_file = claimed_job.get("running_file", "")
            print(f"⚡ [Queue-Hit] 捕获到来自 0 CCU 画布的新任务: job_id={job_id} (gacha_runs={gacha_runs})！", flush=True)

            tmp_wf = Path(f"/tmp/serverless_job_{job_id}.json")
            tmp_wf.write_text(json.dumps(claimed_job["prompt"], indent=2, ensure_ascii=False), encoding="utf-8")
            try:
                res = ctrl.dispatch_workflow_on_g4(
                    workflow_path=tmp_wf,
                    gacha_runs=gacha_runs,
                    prefix=prefix,
                )
                mark_status = "completed"
                err_msg = None
            except Exception as exc:
                mark_status = "failed"
                err_msg = str(exc)
                print(f"❌ [Queue-Error] 任务 {job_id} 执行失败: {err_msg}", flush=True)
            finally:
                tmp_wf.unlink(missing_ok=True)

            finalize_code = f'''import json, time
from pathlib import Path
src = Path({running_file!r})
dst_dir = Path("/content/drive/MyDrive/AI_Workstation/ComfyUI/serverless_queue") / {mark_status!r}
dst_dir.mkdir(parents=True, exist_ok=True)
if src.exists():
    data = json.loads(src.read_text(encoding="utf-8"))
    data["status"] = {mark_status!r}
    data["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if {err_msg!r}:
        data["error"] = {err_msg!r}
    dst = dst_dir / src.name
    dst.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    src.unlink(missing_ok=True)
'''
            ctrl.colab_exec(ctrl.cpu_session, finalize_code, timeout=20.0)
        else:
            # No pending job: check if G4 should be auto-stopped due to idle timeout
            ctrl.check_idle_and_stop_if_needed()

        if max_polls is not None and polls >= max_polls:
            break
        time.sleep(poll_interval_sec)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decoupled CPU ComfyUI Control Plane (0 CCU/h) & On-Demand Serverless G4 Compute Dispatcher"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # 1. `canvas`: start 0 CCU CPU ComfyUI Web UI on CPU session
    p_canvas = sub.add_parser("canvas", help="Launch 0.00 CCU/h CPU ComfyUI Canvas Web UI with G4 queue bridge")
    p_canvas.add_argument("-s", "--cpu-session", default="c6e1c7", help="Colab CPU session name (default: c6e1c7)")
    p_canvas.add_argument(
        "--exec-mode",
        choices=["g4-queue", "local-cpu"],
        default="g4-queue",
        help="Intercept 'Queue Prompt' into Drive serverless queue ('g4-queue') or execute on CPU ('local-cpu')",
    )
    p_canvas.add_argument("--refresh-nodes", action="store_true")

    # 2. `dispatch`: on-demand spin up / reuse G4, run workflow or multi-seed Gacha, and auto-stop G4
    p_disp = sub.add_parser("dispatch", help="On-demand boot/reuse G4 GPU, run workflow/Gacha, and auto-stop G4 when done/idle")
    p_disp.add_argument("--workflow", required=True, help="Path to ComfyUI API workflow JSON")
    p_disp.add_argument("--g4-session", default="comfy-g4", help="Colab G4 GPU session name (default: comfy-g4)")
    p_disp.add_argument("--gpu", default="G4", choices=sorted(GPU_VARIANT_SET), help="GPU variant (default: G4)")
    p_disp.add_argument("--profile", default="scheme_a", choices=["scheme_a", "scheme_b", "all"])
    p_disp.add_argument("--gacha-runs", type=int, default=1, help="Number of multi-seed gacha takes (e.g. 4 for 2x2 grid)")
    p_disp.add_argument("--seed-base", type=int, default=2026110801)
    p_disp.add_argument("--seeds", default=None, help="Explicit comma-separated seeds")
    p_disp.add_argument("--prefix", default="ServerlessG4", help="Output filename prefix")
    p_disp.add_argument("--target-seg", type=int, default=None)
    p_disp.add_argument("--gcs-bucket", default=None, help="Optional gs://bucket/path to sync outputs")
    p_disp.add_argument(
        "--auto-stop-g4",
        action="store_true",
        help="Immediately unassign & stop G4 GPU session after this job/batch finishes (0 idle CCU waste)",
    )
    p_disp.add_argument("--idle-timeout", type=int, default=600, help="Idle timeout in seconds (default: 600s = 10m)")

    # 3. `watch`: daemon watching CPU Web UI's `serverless_queue/pending/` and auto-provisioning/auto-stopping G4
    p_watch = sub.add_parser("watch", help="Watch CPU Web UI queue, auto-boot G4 on demand, and auto-stop G4 after idle-timeout")
    p_watch.add_argument("--cpu-session", default="c6e1c7", help="CPU Canvas session name (default: c6e1c7)")
    p_watch.add_argument("--g4-session", default="comfy-g4", help="G4 Compute session name (default: comfy-g4)")
    p_watch.add_argument("--gpu", default="G4", choices=sorted(GPU_VARIANT_SET))
    p_watch.add_argument("--profile", default="scheme_a", choices=["scheme_a", "scheme_b", "all"])
    p_watch.add_argument("--idle-timeout", type=int, default=600, help="Seconds of inactivity before auto-stopping G4 (default: 600)")
    p_watch.add_argument("--auto-stop-g4", action="store_true", help="Stop G4 immediately after each queued job finishes")
    p_watch.add_argument("--poll-interval", type=int, default=8, help="Queue poll interval in seconds (default: 8)")

    # 4. `stop-g4`: manually stop the G4 compute session while keeping the 0 CCU CPU Canvas alive
    p_stop = sub.add_parser("stop-g4", help="Verify Drive sync and stop ONLY the G4 GPU session (keeps 0 CCU CPU Canvas alive)")
    p_stop.add_argument("--g4-session", default="comfy-g4", help="G4 session name to stop (default: comfy-g4)")

    args = parser.parse_args()

    if args.cmd == "canvas":
        launch_cpu_canvas_on_session(
            cpu_session=args.cpu_session,
            exec_mode=args.exec_mode,
            refresh_nodes=args.refresh_nodes,
        )

    elif args.cmd == "dispatch":
        ctrl = ServerlessG4LifecycleController(
            g4_session=args.g4_session,
            gpu_variant=args.gpu,
            profile=args.profile,
            idle_timeout_sec=args.idle_timeout,
            auto_stop_immediately=args.auto_stop_g4,
        )
        res = ctrl.dispatch_workflow_on_g4(
            workflow_path=Path(args.workflow),
            gacha_runs=args.gacha_runs,
            seed_base=args.seed_base,
            seeds_str=args.seeds,
            prefix=args.prefix,
            target_seg_id=args.target_seg,
            gcs_bucket=args.gcs_bucket,
        )
        print(json.dumps(res, indent=2, ensure_ascii=False))

    elif args.cmd == "watch":
        ctrl = ServerlessG4LifecycleController(
            g4_session=args.g4_session,
            cpu_session=args.cpu_session,
            gpu_variant=args.gpu,
            profile=args.profile,
            idle_timeout_sec=args.idle_timeout,
            auto_stop_immediately=args.auto_stop_g4,
        )
        watch_serverless_queue_loop(ctrl, poll_interval_sec=args.poll_interval)

    elif args.cmd == "stop-g4":
        ctrl = ServerlessG4LifecycleController(g4_session=args.g4_session)
        ctrl.stop_g4_session(reason="manual_stop_g4")


if __name__ == "__main__":
    main()
