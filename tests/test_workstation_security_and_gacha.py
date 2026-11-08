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
        import launch_g4_comfyui_fast as g4_fast

        # 1. Verify CPU Canvas launcher uses `--cpu`, Caddy auth on port 18188 -> bridge 8189 + direct ComfyUI 8188,
        #    strips Origin/Sec-Fetch-Site to prevent ComfyUI v0.44+ 403 CSRF rejections, sets static Cache-Control,
        #    uses --protocol http2 for cloudflared, and initializes user/default + user.css on local SSD
        cpu_src = (REPO_ROOT / "scripts" / "launch_cpu_canvas_comfyui.py").read_text(encoding="utf-8")
        self.assertIn('"--cpu"', cpu_src)
        self.assertIn('"--enable-compress-response-body"', cpu_src)
        self.assertIn('"--protocol", "http2"', cpu_src)
        self.assertIn("def start_caddy_auth_gateway", cpu_src)
        self.assertIn("http://127.0.0.1:18188", cpu_src)
        self.assertIn("path /ws* /prompt /api/prompt /queue* /api/queue* /history* /api/history* /api/jobs* /view* /api/view* /serverless/* /api/serverless/*", cpu_src)
        self.assertIn('header @static_cache Cache-Control "public, max-age=86400"', cpu_src)
        self.assertIn("header_up -Origin", cpu_src)
        self.assertIn("header_up -Sec-Fetch-Site", cpu_src)
        self.assertIn("user.css", cpu_src)
        self.assertIn(".index.json", cpu_src)

        # 1b. Verify `wire_local_models_with_aliases` and `wire_hybrid_models_dir` resolve all expected MiniMax-H3 workflow/template filenames
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            fake_drive_models = tmp_root / "drive_models"
            fake_fast_models = tmp_root / "fast_models"
            fake_local_models_cpu = tmp_root / "local_models_cpu"
            fake_local_models_g4 = tmp_root / "local_models_g4"

            canonical_files = [
                ("diffusion_models", "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors"),
                ("diffusion_models", "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"),
                ("text_encoders", "qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors"),
                ("text_encoders", "qwen3vl_32b_minimax_h3_ultra_uncensored_heretic_int8_convrot.safetensors"),
                ("vae", "minimax_h3_video_vae_fp16.safetensors"),
                ("vae", "minimax_h3_audio_vae_fp32.safetensors"),
                ("loras", "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"),
                ("loras", "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"),
                ("loras", "h3-realism-people-t2v-i2v-r2v.safetensors"),
                ("latent_upscale_models", "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"),
                ("embeddings", "minimaxh3_bullet_time.safetensors"),
                ("model_patches", "BUNNY_H3_ActionLogic_Bridge_V1.safetensors"),
                ("model_patches", "BUNNY_H3_ActionLogic_Bridge_V2.safetensors"),
            ]
            for sub, fname in canonical_files:
                p = fake_drive_models / sub / fname
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"weights")

            # Also put one weight in fast_models to verify NVMe preference on G4
            nvme_ref = fake_fast_models / "diffusion_models" / "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors"
            nvme_ref.parent.mkdir(parents=True, exist_ok=True)
            nvme_ref.write_bytes(b"nvme_weights")

            cnt_cpu = cpu_canvas.wire_local_models_with_aliases(fake_local_models_cpu, fake_drive_models)
            cnt_g4 = g4_fast.wire_hybrid_models_dir(fake_local_models_g4, fake_drive_models, fake_fast_models)
            self.assertGreaterEqual(cnt_cpu, 40)
            self.assertGreaterEqual(cnt_g4, 40)

            expected_aliases = [
                ("diffusion_models", "minimax_h3_ref2va_pruned_int8_convrot.safetensors"),
                ("diffusion_models", "minimax_h3_fl2va_pruned_int8_convrot.safetensors"),
                ("unet", "minimax_h3_ref2va_pruned_int8_convrot.safetensors"),
                ("text_encoders", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"),
                ("clip", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"),
                ("vae", "minimax_h3_video_vae_int8_convrot.safetensors"),
                ("loras", "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"),
                ("loras", "Minimax_H3/minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors"),
                ("latent_upscale_models", "minimax_h3_latent_upscaler_3d_bf16.safetensors"),
                ("h3_latent_upscalers", "minimax_h3_latent_upscaler_3d_bf16.safetensors"),
                ("semantic_bridge", "BUNNY_H3_ActionLogic_Bridge_V2.safetensors"),
                ("checkpoints", "sdpose_wholebody_fp16.safetensors"),
                ("unet", "rt_detr_v4-x-hgnet_fp16.safetensors"),
                ("unet", "minimax_music3_dit_fp16.safetensors"),
                ("clip", "minimax_music3_text_encoder_pruned_int8_convrot.safetensors"),
                ("vae", "minimax_music3_dav.safetensors"),
            ]
            for sub, alias_rel in expected_aliases:
                self.assertTrue((fake_local_models_cpu / sub / alias_rel).exists(), f"Missing CPU alias {sub}/{alias_rel}")
                self.assertTrue((fake_local_models_g4 / sub / alias_rel).exists(), f"Missing G4 alias {sub}/{alias_rel}")

            # Verify G4 alias points to fast_models when staged on NVMe
            self.assertEqual(
                (fake_local_models_g4 / "diffusion_models" / "minimax_h3_ref2va_pruned_int8_convrot.safetensors").resolve(),
                nvme_ref.resolve(),
            )

        # 1c. Verify `_format_gacha_prefix` preserves subfolder prefixes like `video/MiniMax_H3`
        formatted_sub = gacha._format_gacha_prefix("video/MiniMax_H3", "GachaT01_S1001", base_prefix="WebUI_b8e0b962")
        self.assertEqual(formatted_sub, "video/WebUI_b8e0b962_MiniMax_H3_GachaT01_S1001")

        # 1d. Verify `mutate_workflow_for_gacha_take(..., take_seed=None)` preserves the user's exact workflow seed
        webui_single_wf = {
            "129": {"class_type": "RandomNoise", "inputs": {"noise_seed": 261662374822964}},
            "92": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "video/MiniMax_H3"}},
        }
        mut_single, single_prefixes = gacha.mutate_workflow_for_gacha_take(
            webui_single_wf, take_idx=1, take_seed=None, base_prefix="WebUI_b8e0b962"
        )
        self.assertEqual(mut_single["129"]["inputs"]["noise_seed"], 261662374822964)
        self.assertEqual(single_prefixes, ["video/WebUI_b8e0b962_MiniMax_H3_GachaT01_S261662374822964"])

        # 1e. Verify G4 launcher includes `start_g4_local_queue_worker_daemon` and CPU Canvas includes `_sync_queue_transitions_from_drive_api`
        g4_src = (REPO_ROOT / "scripts" / "launch_g4_comfyui_fast.py").read_text(encoding="utf-8")
        self.assertIn("def start_g4_local_queue_worker_daemon", g4_src)
        self.assertIn("def _sync_queue_transitions_from_drive_api", cpu_src)

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

            # Test dispatch_workflow_on_g4 and verify no nested double-quote truncation + expected_files extraction
            with tempfile.TemporaryDirectory() as tmpdir:
                wf_p = Path(tmpdir) / "test_api_wf.json"
                wf_p.write_text(
                    json.dumps(
                        {
                            "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}},
                            "20": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "TestOut"}},
                        }
                    ),
                    encoding="utf-8",
                )
                # Update mock_runner to return a JSON summary in tail so expected_files is populated
                orig_mock = ctrl._runner

                def mock_runner_with_outputs(cmd, stdin_text=None, timeout=60.0):
                    if "exec" in cmd and stdin_text and "Popen" in stdin_text and "comfy_gacha_and_gcp.py" in stdin_text:
                        # Ensure bash -c is invoked via list args without broken nested double quotes
                        self.assertIn("['nohup', 'bash', '-c',", stdin_text)
                        self.assertIn("--workflow", stdin_text)
                    if "exec" in cmd and stdin_text and "g4_job.done" in stdin_text and "Popen" not in stdin_text:
                        summary_json = json.dumps(
                            {
                                "seeds": [101, 102],
                                "take_videos": ["/content/drive/MyDrive/AI_Workstation/ComfyUI/output/TestOut_T01.mp4"],
                                "grid_video": "/content/drive/MyDrive/AI_Workstation/ComfyUI/output/TestOut_Grid.mp4",
                                "outputs": {"20": {"images": [{"filename": "TestOut_T01.mp4", "subfolder": "video", "type": "output"}]}},
                            }
                        )
                        return (0, json.dumps({"done": True, "rc": "0", "tail": f"Done\n{summary_json}"}))
                    return orig_mock(cmd, stdin_text, timeout)

                ctrl._runner = mock_runner_with_outputs
                disp_res = ctrl.dispatch_workflow_on_g4(wf_p, gacha_runs=2, prefix="TestOut")
                self.assertEqual(disp_res["status"], "completed")
                self.assertEqual(disp_res["expected_files"], ["TestOut_T01.mp4", "TestOut_Grid.mp4"])
                self.assertIn("20", disp_res["outputs"])
                self.assertEqual(ctrl.last_output_files, ["TestOut_T01.mp4", "TestOut_Grid.mp4"])

            # Idle check after 305s (>= 300s) triggers Drive flush check + `colab_meter.py stop -s comfy-g4`
            stop_res = ctrl.check_idle_and_stop_if_needed(now=ctrl.last_activity_ts + 305)
            self.assertIsNotNone(stop_res)
            self.assertEqual(stop_res["stopped_session"], "comfy-g4")
            self.assertFalse(ctrl.g4_active)

            # 5. Test concurrent FIFO multi-job queuing & watch_serverless_queue_loop sequential processing
            with tempfile.TemporaryDirectory() as tmpdir:
                q_root = Path(tmpdir) / "serverless_queue"
                wf_a = {"1": {"class_type": "RandomNoise", "inputs": {"noise_seed": 1}}}
                wf_b = {"1": {"class_type": "RandomNoise", "inputs": {"noise_seed": 2}}}
                wf_c = {"1": {"class_type": "RandomNoise", "inputs": {"noise_seed": 3}}}
                r1 = cpu_canvas.enqueue_serverless_job({"prompt": wf_a, "prompt_id": "jobA"}, queue_root=q_root, prefix="PrefA")
                r2 = cpu_canvas.enqueue_serverless_job({"prompt": wf_b, "prompt_id": "jobB"}, queue_root=q_root, prefix="PrefB")
                r3 = cpu_canvas.enqueue_serverless_job({"prompt": wf_c, "prompt_id": "jobC"}, queue_root=q_root, prefix="PrefC")
                # Even if called in the same millisecond, filenames are strictly ordered
                pending_names = [p.name for p in sorted((q_root / "pending").glob("job_*.json"), key=lambda p: p.name)]
                self.assertEqual(len(pending_names), 3)
                self.assertTrue(pending_names[0].endswith("_jobA.json"))
                self.assertTrue(pending_names[1].endswith("_jobB.json"))
                self.assertTrue(pending_names[2].endswith("_jobC.json"))

            # 6. Test cloud-autonomous `write_runtime_state`, `_local_cpu_vm` execution, and timeout resilience
            with tempfile.TemporaryDirectory() as tmpdir:
                state_dir = Path(tmpdir)
                orig_local_sf = disp.G4_LOCAL_STATE_FILE
                orig_drive_sf = disp.G4_DRIVE_STATE_FILE
                disp.G4_LOCAL_STATE_FILE = state_dir / "local_g4_state.json"
                disp.G4_DRIVE_STATE_FILE = state_dir / "drive_g4_state.json"
                try:
                    local_ctrl = disp.ServerlessG4LifecycleController(
                        g4_session="comfy-g4",
                        cpu_session="c6e1c7",
                        gpu_variant="G4",
                        idle_timeout_sec=600,
                    )
                    local_ctrl._local_cpu_vm = True

                    # Verify `write_runtime_state` persists all required fields for `/serverless/status`
                    local_ctrl.g4_active = True
                    rt_booting = local_ctrl.write_runtime_state("booting", {"note": "cold_start"})
                    self.assertEqual(rt_booting["g4_state"], "booting")
                    self.assertTrue(rt_booting["g4_active"])
                    self.assertEqual(rt_booting["idle_timeout_seconds"], 600)
                    self.assertTrue(rt_booting["auto_start_enabled"])
                    self.assertTrue(rt_booting["auto_stop_enabled"])
                    self.assertTrue(disp.G4_LOCAL_STATE_FILE.exists())
                    self.assertTrue(disp.G4_DRIVE_STATE_FILE.exists())

                    local_ctrl.last_activity_ts = disp.time.time() - 42.0
                    rt_warm = local_ctrl.write_runtime_state("ready_warm")
                    self.assertEqual(rt_warm["g4_state"], "ready_warm")
                    self.assertGreaterEqual(rt_warm["idle_seconds"], 40.0)

                    local_ctrl.g4_active = False
                    rt_stopped = local_ctrl.write_runtime_state("stopped")
                    self.assertEqual(rt_stopped["g4_state"], "stopped")
                    self.assertFalse(rt_stopped["g4_active"])
                    self.assertEqual(rt_stopped["idle_seconds"], 0.0)

                    # Verify `_local_cpu_vm=True` executes `cpu_session` commands via `sys.executable -c` instead of `colab exec`
                    rc_loc, out_loc = local_ctrl.colab_exec("c6e1c7", "print('LOCAL_VM_EXEC_OK')", timeout=5.0)
                    self.assertEqual(rc_loc, 0)
                    self.assertIn("LOCAL_VM_EXEC_OK", out_loc)

                    src_f = state_dir / "src.txt"
                    dst_f = state_dir / "dst.txt"
                    src_f.write_text("hello_local_vm", encoding="utf-8")
                    local_ctrl.colab_upload("c6e1c7", src_f, str(dst_f))
                    self.assertEqual(dst_f.read_text(encoding="utf-8"), "hello_local_vm")

                    # Verify `colab_exec` and `is_session_alive` gracefully survive `subprocess.TimeoutExpired`
                    def timeout_runner(cmd, stdin_text=None, timeout=60.0):
                        import subprocess as _sp
                        raise _sp.TimeoutExpired(cmd=cmd, timeout=timeout)

                    local_ctrl._runner = timeout_runner
                    rc_to, out_to = local_ctrl.colab_exec("comfy-g4", "print(1)", timeout=1.0)
                    self.assertEqual(rc_to, 124)
                    self.assertEqual(out_to, "TIMEOUT")
                    self.assertFalse(local_ctrl.is_session_alive("comfy-g4"))
                    with self.assertRaises(RuntimeError) as ctx_up:
                        local_ctrl.colab_upload("comfy-g4", src_f, "/content/dst.txt", timeout=1.0)
                    self.assertIn("TIMEOUT", str(ctx_up.exception))

                    # Verify `is_g4_comfy_ready` restores `last_activity_ts` from persisted state if watcher restarts while G4 is warm
                    saved_act_ts = disp.time.time() - 180.0
                    disp.G4_LOCAL_STATE_FILE.write_text(
                        json.dumps({"g4_state": "ready_warm", "last_activity_ts": saved_act_ts}),
                        encoding="utf-8",
                    )
                    restart_ctrl = disp.ServerlessG4LifecycleController(
                        g4_session="comfy-g4",
                        cpu_session="c6e1c7",
                        gpu_variant="G4",
                        idle_timeout_sec=600,
                        runner=lambda cmd, stdin_text=None, timeout=60.0: (
                            (0, "[comfy-g4] Hardware: G4") if "sessions" in cmd else (0, "COMFY_READY_OK")
                        ),
                    )
                    self.assertTrue(restart_ctrl.is_g4_comfy_ready())
                    self.assertAlmostEqual(restart_ctrl.last_activity_ts, saved_act_ts, delta=1.0)

                    # Verify `sync_colab_cli_to_cpu_session` syncs credentials with 0600 permissions and never touches Drive
                    fake_cfg = state_dir / "colab-cli"
                    fake_cfg.mkdir(parents=True, exist_ok=True)
                    (fake_cfg / "sessions.json").write_text('{"c6e1c7": {}}', encoding="utf-8")
                    (fake_cfg / "dfs_credentials.json").write_text('{"refresh_token": "sec"}', encoding="utf-8")
                    captured_exec = []
                    local_ctrl._local_cpu_vm = False
                    local_ctrl._runner = lambda cmd, stdin_text=None, timeout=60.0: (
                        captured_exec.append((cmd, stdin_text)),
                        (0, '{"ok": true}'),
                    )[1]
                    sync_res = disp.sync_colab_cli_to_cpu_session(local_ctrl, cpu_session="c6e1c7", local_config_dir=fake_cfg)
                    self.assertIn("sessions.json", sync_res["synced_files"])
                    self.assertIn("dfs_credentials.json", sync_res["synced_files"])
                    self.assertTrue(any("0o600" in (s or "") and "/root/.config/colab-cli" in (s or "") for _, s in captured_exec))
                finally:
                    disp.G4_LOCAL_STATE_FILE = orig_local_sf
                    disp.G4_DRIVE_STATE_FILE = orig_drive_sf

            # 7. Verify G4 in-VM idle self-destruct timer & CPU Canvas autonomous watcher hooks exist in source
            self.assertIn("def _self_unassign_g4", g4_src)
            self.assertIn("def _discover_tbe_runtime_addr", g4_src)
            self.assertIn("def ensure_cloud_autonomous_watcher_on_cpu_vm", cpu_src)
            self.assertIn('"g4_state": g4_state', cpu_src)
            self.assertIn('"auto_start_enabled":', cpu_src)
            self.assertIn('"auto_stop_enabled":', cpu_src)
        finally:
            disp.time.sleep = orig_sleep


if __name__ == "__main__":
    unittest.main(verbosity=2)


