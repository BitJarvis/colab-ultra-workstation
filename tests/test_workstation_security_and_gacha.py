#!/usr/bin/env python3
"""Automated Security & Functional Test Suite for `colab-ultra-workstation`.

Tests:
1. CRITICAL FIX 1 (`mcp_dfs_mount.py`):
   - Verifies `in_mounts()` detection and ensures `rm -rf` is NEVER present on `/content/drive`.
   - Verifies `rmdir` is used instead of `rm -rf`.
2. CRITICAL FIX 2 (`launch_g4_comfyui_fast.py`):
   - Dynamically extracts `_safe_ckpt_path()` from `launch_g4_comfyui_fast.py` (never from a hardcoded copy)
     and verifies it blocks path traversal (`../../etc/passwd`, `..\\..\\etc\\passwd`, `/tmp/evil.pt`, `.`, `..`, empty, `/`)
     and confines checkpoints strictly inside `LATENT_DIR`.
   - Verifies `start_caddy_auth_gateway()` and `--no-tunnel` are wired into `launch_g4_comfyui_fast.py`
     so port 8188 is never exposed unauthenticated to `trycloudflare.com`.
3. CRITICAL FIX 3 (Hardcoded `/Users/delroy` removal & conditional proxy):
   - Scans the entire repository to ensure zero occurrences of `/Users/delroy`.
   - Verifies `fast_start_g4.sh` checks whether port 7890 is listening before exporting `127.0.0.1:7890`.
4. ComfyUI Batch Video Gacha & Multi-Segment Stitching Integration (`comfy_gacha_and_gcp.py`, `run_dance.py`, `run_drama.py`, `run_g4_v25_4seg_drama.py`):
   - Tests `parse_seed_list`, `mutate_workflow_for_gacha_take`, `ensure_api_workflow_format`, `build_ffmpeg_grid_command`, `validate_gcs_uri`.
   - Verifies `_SegXX` suffix preservation so multi-segment `stitch_available_segments()` in `run_dance.py` and `run_drama.py`
     finds and stitches per-take segment MP4s (`{prefix}_GachaT01_S{seed}_Seg01*.mp4` -> `{prefix}_GachaT01_S{seed}_Master.mp4`).
   - Verifies boolean inputs are not corrupted as integer seeds (`isinstance(True, int)` edge case) and Web UI workflows are rejected cleanly.
5. GCS URI Validation:
   - Verifies `validate_gcs_uri` accepts valid buckets and rejects injection/invalid URIs.
6. FFmpeg 2x2 Grid Synthesis & Watermark:
   - Executes an end-to-end FFmpeg 2x2 gacha comparison grid synthesis test using synthetic test MP4s.
7. Repository-wide Syntax & Compilation Check:
   - Compiles every `.py` file with `py_compile` and checks every `.sh` file with `bash -n`.
8. Skill Mirror Parity Check:
   - Verifies `skills/colab-ultra-workstation/` matches the root `SKILL.md`, `scripts/`, and `references/`.
"""

from __future__ import annotations

import importlib.util
import json
import os
import py_compile
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import comfy_gacha_and_gcp as gacha  # noqa: E402


def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestWorkstationSecurityAndGacha(unittest.TestCase):
    def test_01_drive_mount_never_uses_rm_rf(self) -> None:
        """Verify CRITICAL FIX 1: `mcp_dfs_mount.py` checks `in_mounts()` and uses `rmdir`, never `rm -rf` on MP."""
        targets = [
            REPO_ROOT / "scripts" / "mcp_dfs_mount.py",
            REPO_ROOT / "skills" / "colab-ultra-workstation" / "scripts" / "mcp_dfs_mount.py",
        ]
        for p in targets:
            self.assertTrue(p.exists(), f"Missing {p}")
            content = p.read_text(encoding="utf-8")
            self.assertNotIn('rm -rf "{MP}"', content, f"Unsafe 'rm -rf \"{{MP}}\"' found in {p}")
            self.assertNotIn("rm -rf /content/drive", content, f"Unsafe 'rm -rf /content/drive' found in {p}")
            self.assertIn("def in_mounts()", content, f"Missing in_mounts() guard in {p}")
            self.assertIn("CRITICAL SAFETY STOP: mountpoint", content, f"Missing safety stop in {p}")
            self.assertIn('rmdir "{MP}"', content, f"Missing safe rmdir in {p}")

    def test_02_comfyui_caddy_auth_and_safe_ckpt_path(self) -> None:
        """Verify CRITICAL FIX 2: Caddy Auth Gateway on port 18188 + dynamic extraction of `_safe_ckpt_path`."""
        targets = [
            REPO_ROOT / "scripts" / "launch_g4_comfyui_fast.py",
            REPO_ROOT / "skills" / "colab-ultra-workstation" / "scripts" / "launch_g4_comfyui_fast.py",
        ]
        for p in targets:
            self.assertTrue(p.exists(), f"Missing {p}")
            content = p.read_text(encoding="utf-8")
            self.assertIn("def start_caddy_auth_gateway", content)
            self.assertIn("http://127.0.0.1:18188", content)
            self.assertNotIn('"http://127.0.0.1:8188"', content)
            self.assertIn("def _safe_ckpt_path", content)
            self.assertIn("weights_only=True", content)

        # Dynamically extract the actual `_safe_ckpt_path` definition from `scripts/launch_g4_comfyui_fast.py`
        src_text = (REPO_ROOT / "scripts" / "launch_g4_comfyui_fast.py").read_text(encoding="utf-8")
        m = re.search(
            r"(def _safe_ckpt_path\(checkpoint_name: str\) -> Path:.*?return candidate)",
            src_text,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(m, "Could not extract _safe_ckpt_path from launch_g4_comfyui_fast.py")
        extracted_code = m.group(1)

        with tempfile.TemporaryDirectory() as tmpdir:
            latent_dir = Path(tmpdir) / "latents"
            latent_dir.mkdir(parents=True, exist_ok=True)

            ns = {"Path": Path, "LATENT_DIR": latent_dir}
            exec(extracted_code, ns)
            _safe_ckpt_path = ns["_safe_ckpt_path"]

            # Valid filename (with and without .pt suffix)
            valid = _safe_ckpt_path("Seg01_av_latent.pt")
            self.assertEqual(valid, latent_dir.resolve() / "Seg01_av_latent.pt")
            valid_no_ext = _safe_ckpt_path("Seg02_av_latent")
            self.assertEqual(valid_no_ext, latent_dir.resolve() / "Seg02_av_latent.pt")

            # Unix & Windows traversal attempts get stripped to basename inside LATENT_DIR
            for attempt in ("../../etc/passwd", "..\\..\\etc\\passwd", "/tmp/evil.pt"):
                confined = _safe_ckpt_path(attempt)
                self.assertEqual(confined.parent, latent_dir.resolve())
                self.assertIn(confined.name, ("passwd.pt", "evil.pt"))

            for bad in ("", "   ", ".", "..", "/", "\\", "foo/bar/"):
                with self.assertRaises(ValueError, msg=f"Expected ValueError for {bad!r}"):
                    _safe_ckpt_path(bad)

    def test_03_no_hardcoded_delroy_paths_and_conditional_proxy(self) -> None:
        """Verify CRITICAL FIX 3: Zero `/Users/delroy` references and conditional proxy in `fast_start_g4.sh`."""
        for root, dirs, files in os.walk(REPO_ROOT):
            if ".git" in dirs:
                dirs.remove(".git")
            if "__pycache__" in dirs:
                dirs.remove("__pycache__")
            for fname in files:
                fpath = Path(root) / fname
                if fpath.resolve() == Path(__file__).resolve():
                    continue
                if fpath.suffix.lower() in {".py", ".sh", ".md", ".json", ".toml", ".yaml", ".yml"}:
                    txt = fpath.read_text(encoding="utf-8", errors="ignore")
                    self.assertNotIn("/Users/delroy", txt, f"Found hardcoded /Users/delroy in {fpath}")

        sh_targets = [
            REPO_ROOT / "scripts" / "deepseek_g4" / "fast_start_g4.sh",
            REPO_ROOT / "skills" / "colab-ultra-workstation" / "scripts" / "deepseek_g4" / "fast_start_g4.sh",
        ]
        for sp in sh_targets:
            txt = sp.read_text(encoding="utf-8")
            self.assertIn("nc -z 127.0.0.1 7890", txt)

    def test_04_gacha_seed_parsing_workflow_mutation_and_stitching_integration(self) -> None:
        """Verify multi-seed parsing, `_SegXX` prefix preservation, boolean safety, Web UI format check, and stitching."""
        seeds = gacha.parse_seed_list(None, seed_base=1000, gacha_runs=4, seed_step=10)
        self.assertEqual(seeds, [1000, 1010, 1020, 1030])

        explicit = gacha.parse_seed_list("42, 99, 777")
        self.assertEqual(explicit, [42, 99, 777])

        with self.assertRaises(ValueError):
            gacha.parse_seed_list("-5,10")
        with self.assertRaises(ValueError):
            gacha.parse_seed_list(None, seed_base=10, gacha_runs=0)

        # Reject ComfyUI Web UI format JSON cleanly
        webui_wf_path = REPO_ROOT / "workflows" / "04_douyin_beauty_dance" / "h3-motion-transfer-workflow.json"
        if webui_wf_path.exists():
            webui_wf = json.loads(webui_wf_path.read_text(encoding="utf-8"))
            with self.assertRaises(ValueError):
                gacha.mutate_workflow_for_gacha_take(webui_wf, take_idx=1, take_seed=12345)

        sample_wf = {
            "25": {"class_type": "RandomNoise", "inputs": {"noise_seed": 111, "seed": True}},
            "35": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "Dance_Seg01"}},
            "45": {"class_type": "RandomNoise", "inputs": {"noise_seed": 112}},
            "55": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "Dance_Seg02"}},
            "901": {"class_type": "H3SaveAVLatent", "inputs": {"checkpoint_name": "Dance_Seg01_av_latent.pt"}},
        }
        mut_wf, prefixes = gacha.mutate_workflow_for_gacha_take(sample_wf, take_idx=2, take_seed=8888)
        self.assertEqual(mut_wf["25"]["inputs"]["noise_seed"], 8888)
        # Ensure boolean input was NOT corrupted into an integer seed
        self.assertIs(mut_wf["25"]["inputs"]["seed"], True)
        self.assertEqual(mut_wf["45"]["inputs"]["noise_seed"], 8889)
        # Ensure _GachaT02_S8888 is inserted BEFORE _Seg01 / _Seg02 so stitch_available_segments finds them
        self.assertIn("Dance_GachaT02_S8888_Seg01", prefixes)
        self.assertIn("Dance_GachaT02_S8888_Seg02", prefixes)
        self.assertEqual(
            mut_wf["901"]["inputs"]["checkpoint_name"],
            "Dance_GachaT02_S8888_Seg01_av_latent.pt",
        )
        # Ensure original workflow was not mutated in-place
        self.assertEqual(sample_wf["25"]["inputs"]["noise_seed"], 111)

        # End-to-end integration test with `run_dance.py` and `run_drama.py` stitch_available_segments
        if shutil.which("ffmpeg"):
            run_dance_mod = _load_module_from_path(
                "run_dance_mod",
                REPO_ROOT / "workflows" / "04_douyin_beauty_dance" / "run_dance.py",
            )
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_out = Path(tmpdir)
                for seg_prefix in prefixes:
                    seg_mp4 = tmp_out / f"{seg_prefix}_00001_.mp4"
                    subprocess.run(
                        [
                            "ffmpeg", "-y",
                            "-f", "lavfi", "-i", "color=c=blue:s=160x284:r=12:d=1",
                            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                            "-t", "1",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-shortest",
                            str(seg_mp4),
                        ],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                take_prefix = "Dance_GachaT02_S8888"
                master_mp4 = run_dance_mod.stitch_available_segments(take_prefix, tmp_out, segment_count=2)
                self.assertIsNotNone(master_mp4, "stitch_available_segments failed to find gacha segment MP4s")
                self.assertTrue(master_mp4.exists())
                self.assertEqual(master_mp4.name, "Dance_GachaT02_S8888_Master.mp4")

    def test_05_gcs_uri_validation(self) -> None:
        """Verify GCS URI validation accepts valid buckets and rejects injection/invalid URIs."""
        self.assertEqual(gacha.validate_gcs_uri("gs://my-ultra-bucket/comfy/out/"), "gs://my-ultra-bucket/comfy/out")
        self.assertEqual(gacha.validate_gcs_uri("gs://colab_models_2026"), "gs://colab_models_2026")
        for invalid in ("", "http://bucket/a", "gs://", "gs://-bad", "gs://bucket/path; rm -rf /"):
            with self.assertRaises(ValueError):
                gacha.validate_gcs_uri(invalid)

    def test_06_ffmpeg_2x2_grid_command_and_execution(self) -> None:
        """Verify FFmpeg 2x2 and 3-way filtergraph command generation, and execute if ffmpeg is installed."""
        dummy_4 = [Path(f"/tmp/take_{i}.mp4") for i in range(1, 5)]
        cmd_4 = gacha.build_ffmpeg_grid_command(
            dummy_4, Path("/tmp/grid4.mp4"), seeds=[101, 102, 103, 104], tile_w=540, tile_h=960
        )
        self.assertIn("-filter_complex", cmd_4)
        fc_4 = cmd_4[cmd_4.index("-filter_complex") + 1]
        self.assertIn("xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0:fill=black[vout]", fc_4)

        dummy_3 = [Path(f"/tmp/take_{i}.mp4") for i in range(1, 4)]
        cmd_3 = gacha.build_ffmpeg_grid_command(
            dummy_3, Path("/tmp/grid3.mp4"), seeds=[101, 102, 103], tile_w=540, tile_h=960
        )
        fc_3 = cmd_3[cmd_3.index("-filter_complex") + 1]
        self.assertIn("color=c=black:s=540x960:r=24:d=5[v3]", fc_3)
        self.assertIn("xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0:fill=black[vout]", fc_3)

        if shutil.which("ffmpeg"):
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                vids = []
                for i, col in enumerate(["red", "green", "blue", "yellow"], start=1):
                    vp = tmp / f"take_{i}.mp4"
                    subprocess.run(
                        [
                            "ffmpeg", "-y",
                            "-f", "lavfi", "-i", f"color=c={col}:s=160x284:r=12:d=1",
                            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                            "-t", "1",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-shortest",
                            str(vp),
                        ],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    vids.append(vp)

                grid_out = tmp / "gacha_2x2_grid.mp4"
                res = gacha.render_gacha_comparison_grid(
                    vids, grid_out, seeds=[101, 102, 103, 104], tile_w=160, tile_h=284
                )
                self.assertIsNotNone(res)
                self.assertTrue(grid_out.exists())

    def test_07_compile_all_python_and_shell_scripts(self) -> None:
        """Compile all .py files and syntax-check all .sh files in the repository."""
        py_count = 0
        sh_count = 0
        for root, dirs, files in os.walk(REPO_ROOT):
            if ".git" in dirs:
                dirs.remove(".git")
            if "__pycache__" in dirs:
                dirs.remove("__pycache__")
            for f in files:
                p = Path(root) / f
                if p.suffix == ".py":
                    py_compile.compile(str(p), doraise=True)
                    py_count += 1
                elif p.suffix == ".sh":
                    subprocess.run(["bash", "-n", str(p)], check=True)
                    sh_count += 1
        self.assertGreaterEqual(py_count, 20)
        self.assertGreaterEqual(sh_count, 2)

    def test_08_skill_mirror_parity(self) -> None:
        """Verify `skills/colab-ultra-workstation/` matches the root `SKILL.md`, `scripts/`, and `references/`."""
        skill_root = REPO_ROOT / "skills" / "colab-ultra-workstation"
        for root, dirs, files in os.walk(skill_root):
            if "__pycache__" in dirs:
                dirs.remove("__pycache__")
            for f in files:
                sp = Path(root) / f
                rel = sp.relative_to(skill_root)
                rp = REPO_ROOT / rel
                self.assertTrue(rp.exists(), f"Skill file {rel} has no counterpart at {rp}")
                self.assertEqual(
                    sp.read_bytes(),
                    rp.read_bytes(),
                    f"Skill mirror {sp} is out of sync with root {rp}",
                )

    def test_09_decoupled_cpu_canvas_and_serverless_g4_dispatcher(self) -> None:
        """Verify Decoupled CPU Canvas (`launch_cpu_canvas_comfyui.py`) and On-Demand G4 Dispatcher (`comfy_serverless_g4_dispatcher.py`)."""
        import comfy_serverless_g4_dispatcher as disp
        import launch_cpu_canvas_comfyui as cpu_canvas

        # 1. Verify CPU Canvas launcher uses `--cpu`, Caddy auth on port 18188 -> bridge 8189 -> ComfyUI 8188
        cpu_src = (REPO_ROOT / "scripts" / "launch_cpu_canvas_comfyui.py").read_text(encoding="utf-8")
        self.assertIn('"--cpu"', cpu_src)
        self.assertIn("def start_caddy_auth_gateway", cpu_src)
        self.assertIn("http://127.0.0.1:18188", cpu_src)

        # 2. Test `enqueue_serverless_job` validation, sanitization, and atomic file creation
        with tempfile.TemporaryDirectory() as tmpdir:
            q_root = Path(tmpdir) / "serverless_queue"
            sample_wf = {
                "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}},
                "20": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "TestCanvas"}},
            }
            res = cpu_canvas.enqueue_serverless_job(
                {"prompt": sample_wf, "prompt_id": "../../evil_id/../job123", "client_id": "web_client_1"},
                queue_root=q_root,
                gacha_runs=4,
                prefix="CanvasGacha",
            )
            self.assertTrue(res["serverless_g4_queued"])
            # Ensure path traversal characters were stripped from prompt_id
            self.assertNotIn("/", res["prompt_id"])
            self.assertNotIn(".", res["prompt_id"])
            qfile = Path(res["queue_file"])
            self.assertTrue(qfile.exists())
            self.assertEqual(qfile.parent.resolve(), (q_root / "pending").resolve())
            saved = json.loads(qfile.read_text(encoding="utf-8"))
            self.assertEqual(saved["gacha_runs"], 4)
            self.assertEqual(saved["prefix"], "CanvasGacha")
            self.assertEqual(saved["status"], "pending")

            # Reject empty or invalid workflows
            with self.assertRaises(ValueError):
                cpu_canvas.enqueue_serverless_job({"prompt": {}}, queue_root=q_root)
            with self.assertRaises(ValueError):
                cpu_canvas.enqueue_serverless_job({"prompt": {"nodes": [], "links": []}}, queue_root=q_root)

        # 3. Test session name & GPU variant validation
        self.assertEqual(disp.validate_session_name("c6e1c7"), "c6e1c7")
        self.assertEqual(disp.validate_session_name("comfy-g4_01"), "comfy-g4_01")
        for bad_sess in ("", "bad; rm -rf /", "../etc", "a" * 80):
            with self.assertRaises(ValueError):
                disp.validate_session_name(bad_sess)
        self.assertEqual(disp.validate_gpu_variant("g4"), "G4")
        with self.assertRaises(ValueError):
            disp.validate_gpu_variant("RTX9999")

        # 4. Test `ServerlessG4LifecycleController` lazy boot, reuse, dispatch, and idle auto-stop with mock runner
        calls = []
        state = {"alive": False, "comfy_ready": False}

        def mock_runner(cmd, stdin_text=None, timeout=60.0):
            calls.append((cmd, stdin_text))
            cmd_str = " ".join(cmd)
            if "sessions" in cmd:
                return (0, "[comfy-g4] m-s-g4 | Hardware: G4" if state["alive"] else "[c6e1c7] Hardware: CPU")
            if "new" in cmd and "--gpu" in cmd:
                state["alive"] = True
                return (0, "Created G4 session")
            if "dfs" in cmd and "mount" in cmd:
                return (0, "Mounted Drive")
            if "upload" in cmd:
                return (0, "Uploaded")
            if "exec" in cmd:
                if stdin_text and "launch_g4_comfyui_fast.py" in stdin_text:
                    state["comfy_ready"] = True
                    return (0, "G4_FASTBOOT_STARTED")
                if stdin_text and "system_stats" in stdin_text:
                    return (0, "COMFY_READY_OK" if state["comfy_ready"] else "COMFY_NOT_READY")
                if stdin_text and "g4_job.done" in stdin_text and "Popen" not in stdin_text:
                    return (0, json.dumps({"done": True, "rc": "0", "tail": "Gacha done"}))
                if stdin_text and "files?{params}" in stdin_text:
                    return (0, json.dumps({"ok": True, "missing": [], "cloud_count": 5}))
                return (0, "OK")
            if "colab_meter.py" in cmd_str and "stop" in cmd:
                state["alive"] = False
                state["comfy_ready"] = False
                return (0, "✅ 算力释放与计费核验报告：会话 'comfy-g4' 已释放")
            return (0, "OK")

        # Patch time.sleep briefly inside controller test so unit test runs in <0.1s
        orig_sleep = disp.time.sleep
        disp.time.sleep = lambda _s: None
        try:
            ctrl = disp.ServerlessG4LifecycleController(
                g4_session="comfy-g4",
                cpu_session="c6e1c7",
                gpu_variant="G4",
                idle_timeout_sec=300,
                auto_stop_immediately=False,
                runner=mock_runner,
            )
            # First ensure_g4_ready triggers lazy provisioning (`colab new --gpu G4` + `colab dfs mount`)
            boot_res = ctrl.ensure_g4_ready(boot_timeout_sec=30)
            self.assertFalse(boot_res["reused"])
            self.assertTrue(ctrl.g4_active)

            # Second ensure_g4_ready reuses the warm G4 instance without re-provisioning
            reuse_res = ctrl.ensure_g4_ready(boot_timeout_sec=30)
            self.assertTrue(reuse_res["reused"])

            # Idle check before timeout does NOT stop G4
            self.assertIsNone(ctrl.check_idle_and_stop_if_needed(now=ctrl.last_activity_ts + 100))
            self.assertTrue(ctrl.g4_active)

            # Idle check after 305s (>= 300s) triggers Drive flush check + `colab_meter.py stop -s comfy-g4`
            stop_res = ctrl.check_idle_and_stop_if_needed(now=ctrl.last_activity_ts + 305)
            self.assertIsNotNone(stop_res)
            self.assertEqual(stop_res["stopped_session"], "comfy-g4")
            self.assertFalse(ctrl.g4_active)
        finally:
            disp.time.sleep = orig_sleep


if __name__ == "__main__":
    unittest.main(verbosity=2)

