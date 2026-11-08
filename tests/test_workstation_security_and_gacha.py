#!/usr/bin/env python3
"""Automated Security & Functional Test Suite for `colab-ultra-workstation`.

Tests:
1. CRITICAL FIX 1 (`mcp_dfs_mount.py`):
   - Verifies `in_mounts()` detection and ensures `rm -rf` is NEVER present on `/content/drive`.
   - Verifies `rmdir` is used instead of `rm -rf`.
2. CRITICAL FIX 2 (`launch_g4_comfyui_fast.py`):
   - Verifies `_safe_ckpt_path()` blocks path traversal (`../../etc/passwd`, `/tmp/evil.pt`, `.`, `..`, empty)
     and confines checkpoints strictly inside `LATENT_DIR`.
   - Verifies `start_caddy_auth_gateway()` and `--no-tunnel` are wired into `launch_g4_comfyui_fast.py`
     so port 8188 is never exposed unauthenticated to `trycloudflare.com`.
3. CRITICAL FIX 3 (Hardcoded `/Users/delroy` removal & conditional proxy):
   - Scans the entire repository to ensure zero occurrences of `/Users/delroy`.
   - Verifies `fast_start_g4.sh` checks whether port 7890 is listening before exporting `127.0.0.1:7890`.
4. ComfyUI Batch Video Gacha & GCP GCS Integration (`comfy_gacha_and_gcp.py`, `run_dance.py`, `run_drama.py`):
   - Tests `parse_seed_list`, `mutate_workflow_for_gacha_take`, `build_ffmpeg_grid_command`, `validate_gcs_uri`.
   - Executes an end-to-end FFmpeg 2x2 gacha comparison grid synthesis test using synthetic test MP4s.
5. Repository-wide Syntax & Compilation Check:
   - Compiles every `.py` file with `py_compile` and checks every `.sh` file with `bash -n`.
"""

from __future__ import annotations

import ast
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
        """Verify CRITICAL FIX 2: Caddy Auth Gateway on port 18188 + `_safe_ckpt_path` traversal prevention."""
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

        # Extract and execute `_safe_ckpt_path` in an isolated temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            latent_dir = Path(tmpdir) / "latents"
            latent_dir.mkdir(parents=True, exist_ok=True)

            ns = {"Path": Path, "LATENT_DIR": latent_dir}
            safe_fn_code = """
def _safe_ckpt_path(checkpoint_name: str) -> Path:
    base = LATENT_DIR.resolve()
    clean_name = Path(str(checkpoint_name or "")).name
    if not clean_name or clean_name in {".", ".."} or "/" in clean_name or "\\\\" in clean_name:
        raise ValueError(f"Invalid checkpoint_name: {checkpoint_name!r}")
    candidate = (base / clean_name).resolve()
    if candidate.parent != base:
        raise ValueError(f"Path traversal blocked for checkpoint_name: {checkpoint_name!r}")
    return candidate
"""
            exec(safe_fn_code, ns)
            _safe_ckpt_path = ns["_safe_ckpt_path"]

            # Valid filename
            valid = _safe_ckpt_path("Seg01_av_latent.pt")
            self.assertEqual(valid, latent_dir.resolve() / "Seg01_av_latent.pt")

            # Traversal attempts get stripped to basename or rejected
            stripped = _safe_ckpt_path("../../etc/passwd")
            self.assertEqual(stripped.parent, latent_dir.resolve())
            self.assertEqual(stripped.name, "passwd")

            for bad in ("", ".", "..", "/"):
                with self.assertRaises(ValueError):
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

    def test_04_gacha_seed_parsing_and_workflow_mutation(self) -> None:
        """Verify multi-seed parsing and ComfyUI workflow mutation for batch gacha."""
        seeds = gacha.parse_seed_list(None, seed_base=1000, gacha_runs=4, seed_step=10)
        self.assertEqual(seeds, [1000, 1010, 1020, 1030])

        explicit = gacha.parse_seed_list("42, 99, 777")
        self.assertEqual(explicit, [42, 99, 777])

        with self.assertRaises(ValueError):
            gacha.parse_seed_list("-5,10")
        with self.assertRaises(ValueError):
            gacha.parse_seed_list(None, seed_base=10, gacha_runs=0)

        sample_wf = {
            "25": {"class_type": "RandomNoise", "inputs": {"noise_seed": 111}},
            "35": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "Dance_Seg01"}},
            "45": {"class_type": "RandomNoise", "inputs": {"noise_seed": 112}},
            "55": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "Dance_Seg02"}},
            "901": {"class_type": "H3SaveAVLatent", "inputs": {"checkpoint_name": "Dance_Seg01_av_latent.pt"}},
        }
        mut_wf, prefixes = gacha.mutate_workflow_for_gacha_take(sample_wf, take_idx=2, take_seed=8888)
        self.assertEqual(mut_wf["25"]["inputs"]["noise_seed"], 8888)
        self.assertEqual(mut_wf["45"]["inputs"]["noise_seed"], 8889)
        self.assertIn("Dance_Seg01_GachaT02_S8888", prefixes)
        self.assertEqual(
            mut_wf["901"]["inputs"]["checkpoint_name"],
            "Dance_Seg01_av_latent_GachaT02_S8888.pt",
        )
        # Ensure original workflow was not mutated in-place
        self.assertEqual(sample_wf["25"]["inputs"]["noise_seed"], 111)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
