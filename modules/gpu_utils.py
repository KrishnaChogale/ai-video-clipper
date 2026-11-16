"""
modules/gpu_utils.py — GPU detection, VRAM monitoring, device selection.
Never crashes; always falls back to CPU gracefully.
Optimised for RTX 3050 Laptop GPU (6GB VRAM) running on Windows 11.
"""
import logging
import shutil
import subprocess

logger = logging.getLogger(__name__)


def get_device_info() -> dict:
    """
    Returns a dict with GPU/CUDA status.
    Safe to call even when PyTorch or CUDA is not installed.
    """
    info = {
        "cuda_available": False,
        "gpu_name": "N/A",
        "vram_total_gb": 0.0,
        "vram_free_gb": 0.0,
        "torch_version": "N/A",
        "torch_cuda_version": "N/A",
        "nvenc_available": False,
        "ffmpeg_path": shutil.which("ffmpeg") or "NOT FOUND",
        "device": "cpu",
        "driver_version": "N/A",
    }

    # ── PyTorch ──
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["torch_cuda_version"] = torch.version.cuda or "None"
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["vram_total_gb"] = round(props.total_memory / 1e9, 2)
            free, total = torch.cuda.mem_get_info(0)
            info["vram_free_gb"] = round(free / 1e9, 2)
            info["device"] = "cuda"
    except Exception as e:
        logger.debug("PyTorch CUDA check failed: %s", e)

    # ── nvidia-smi fallback ──
    try:
        smi = shutil.which("nvidia-smi")
        if smi:
            r = subprocess.run(
                [smi, "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if r.returncode == 0:
                parts = r.stdout.strip().split(",")
                if len(parts) >= 2:
                    if info["gpu_name"] == "N/A":
                        info["gpu_name"] = parts[0].strip()
                    mb = float(parts[1].strip().replace(" MiB", ""))
                    if info["vram_total_gb"] == 0.0:
                        info["vram_total_gb"] = round(mb / 1024, 2)
                if len(parts) >= 3:
                    info["driver_version"] = parts[2].strip()
    except Exception as e:
        logger.debug("nvidia-smi check failed: %s", e)

    # ── NVENC via FFmpeg ──
    try:
        from modules.clip_editor import FFMPEG_EXE, _use_nvenc
        info["ffmpeg_path"] = FFMPEG_EXE
        info["nvenc_available"] = _use_nvenc()
    except Exception as e:
        logger.debug("FFmpeg NVENC check failed: %s", e)

    return info


def select_device(preferred: str = "auto", min_free_vram_gb: float = 1.0) -> str:
    """
    Returns 'cuda' or 'cpu'.
    preferred: 'auto' | 'cuda' | 'cpu'
    min_free_vram_gb: minimum free VRAM required to use CUDA (default 1.0 GB).
    """
    if preferred == "cpu":
        return "cpu"
    info = get_device_info()
    if info["cuda_available"] and info["vram_free_gb"] >= min_free_vram_gb:
        return "cuda"
    if preferred == "cuda":
        logger.warning("CUDA requested but insufficient VRAM (%s GB free). Falling back to CPU.", info["vram_free_gb"])
    return "cpu"


def release_gpu_memory():
    """Force-release cached GPU memory between model loads."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def get_vram_free_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info(0)
            return round(free / 1e9, 2)
    except Exception:
        pass
    return 0.0
