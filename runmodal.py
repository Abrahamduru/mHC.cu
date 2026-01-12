import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import modal

APP_NAME = "mhc-cu"

app = modal.App(APP_NAME)

base_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.10"
    )
    .apt_install(
        "build-essential",
        "cmake",
        "ninja-build",
        "python3-dev",
        "python3-pip",
    )
    .run_commands(
        "ln -sf /usr/bin/python3 /usr/bin/python",
        "python3 -m pip install numpy ninja pytest",
        "python3 -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128",
    )
)
image = base_image.add_local_dir(Path(__file__).parent, remote_path="/workspace")


def _print_env() -> None:
    import torch

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def _run_make(target: str, cuda_arch: Optional[str]) -> None:
    env = os.environ.copy()
    env.setdefault("CC", "gcc")
    env.setdefault("CXX", "g++")
    env.setdefault("CUDACXX", "/usr/local/cuda/bin/nvcc")
    env.setdefault("PIP_NO_BUILD_ISOLATION", "1")
    if cuda_arch:
        env["CUDA_ARCH"] = cuda_arch
    print(f"Running: make {target}")
    subprocess.run(["make", target], check=True, cwd="/workspace", env=env)


def _run_tests(scope: str, cuda_arch: Optional[str]) -> None:
    if scope in ("native", "all"):
        _run_make("test", cuda_arch)
    if scope in ("python", "all"):
        _run_make("test-python", cuda_arch)


def _run_benchmarks(scope: str, cuda_arch: Optional[str]) -> None:
    if scope in ("native", "all"):
        _run_make("bench", cuda_arch)
    if scope in ("python", "all"):
        _run_make("bench-python", cuda_arch)


def _run_job(mode: str, scope: str, cuda_arch: Optional[str]) -> None:
    sys.path.insert(0, "/workspace")
    _print_env()
    if mode == "test":
        _run_tests(scope, cuda_arch)
        return
    if mode == "bench":
        _run_benchmarks(scope, cuda_arch)
        return
    if mode == "all":
        _run_tests(scope, cuda_arch)
        _run_benchmarks(scope, cuda_arch)
        return
    raise ValueError("mode must be 'test', 'bench', or 'all'")


@app.function(gpu="H100", image=image, timeout=3600)
def run_h100(mode: str, scope: str, cuda_arch: Optional[str]) -> None:
    _run_job(mode, scope, cuda_arch)


@app.function(gpu="B200", image=image, timeout=3600)
def run_b200(mode: str, scope: str, cuda_arch: Optional[str]) -> None:
    _run_job(mode, scope, cuda_arch)


def _default_arch(gpu: str) -> Optional[str]:
    gpu_norm = gpu.strip().lower()
    if gpu_norm == "h100":
        return "90"
    if gpu_norm == "b200":
        return "100"
    return None


@app.local_entrypoint()
def main(
    gpu: str = "h100",
    mode: str = "test",
    scope: str = "all",
    cuda_arch: Optional[str] = None,
) -> None:
    gpu_norm = gpu.strip().lower()
    mode_norm = mode.strip().lower()
    scope_norm = scope.strip().lower()
    if scope_norm not in ("native", "python", "all"):
        raise ValueError("scope must be 'native', 'python', or 'all'")
    resolved_arch = cuda_arch or _default_arch(gpu_norm)
    if gpu_norm == "h100":
        run_h100.remote(mode_norm, scope_norm, resolved_arch)
        return
    if gpu_norm == "b200":
        run_b200.remote(mode_norm, scope_norm, resolved_arch)
        return
    raise ValueError("gpu must be 'h100' or 'b200'")
