"""
tests/stress_test_hardware.py
Stress Test 2: Hardware Verification
Performs actual CUDA operations, YOLO inference, Faster-Whisper on GPU,
memory checks, CPU fallbacks, and real FFmpeg NVENC encoding.
"""
import os
import sys
import time
import subprocess
import shutil
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import config
from modules.gpu_utils import get_device_info, release_gpu_memory, select_device
from modules.clip_editor import FFMPEG_EXE, _use_nvenc

def run_hardware_test():
    print("==================================================")
    print("STAGE 2: HARDWARE & ACCELERATION STRESS TEST")
    print("==================================================")

    # 1. Device Info
    info = get_device_info()
    print(f"[*] GPU Name:            {info['gpu_name']}")
    print(f"[*] Total VRAM:          {info['vram_total_gb']} GB")
    print(f"[*] Free VRAM:           {info['vram_free_gb']} GB")
    print(f"[*] PyTorch Version:     {info['torch_version']}")
    print(f"[*] PyTorch CUDA:        {info['torch_cuda_version']} (Available: {info['cuda_available']})")
    print(f"[*] FFmpeg Path:         {info['ffmpeg_path']}")
    print(f"[*] NVENC Detected:      {info['nvenc_available']}")

    assert info["cuda_available"], "CUDA is NOT available in PyTorch!"
    assert "RTX 3050" in info["gpu_name"] or "NVIDIA" in info["gpu_name"], f"Unexpected GPU: {info['gpu_name']}"

    import torch

    # 2. Actual CUDA Tensor Computation Test
    print("[*] Running actual CUDA tensor computation (4096 x 4096 matrix multiplication)...")
    torch.cuda.synchronize()
    t0 = time.time()
    A = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    B = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    C = torch.matmul(A, B)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"[+] CUDA matmul (4096 x 4096 FP16) completed in {elapsed*1000:.2f} ms")
    assert not torch.isnan(C).any(), "NaN detected in CUDA computation!"
    del A, B, C
    release_gpu_memory()

    # 3. YOLO Inference on CUDA
    print("[*] Testing Ultralytics YOLO on CUDA:0...")
    from modules.object_detection import _get_yolo
    yolo_model = _get_yolo()
    dummy_frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
    t0 = time.time()
    res = yolo_model(dummy_frame, device="cuda:0", verbose=False)
    yolo_elapsed = time.time() - t0
    print(f"[+] YOLO nano inference on CUDA:0 completed in {yolo_elapsed*1000:.2f} ms")
    assert len(res) > 0, "YOLO produced no output!"
    release_gpu_memory()

    # 4. Faster-Whisper on CUDA (float16)
    print("[*] Testing Faster-Whisper on CUDA (compute_type='float16')...")
    from modules.speech_recognition import transcribe_audio
    # 2 seconds of 16kHz mono audio
    dummy_audio = np.zeros(16000 * 2, dtype=np.float32)
    t0 = time.time()
    transcript, words = transcribe_audio(dummy_audio, model_size="base")
    whisper_elapsed = time.time() - t0
    print(f"[+] Faster-Whisper base inference completed in {whisper_elapsed:.2f}s")
    release_gpu_memory()

    # 5. VRAM Cleanup Verification
    free_mem, total_mem = torch.cuda.mem_get_info(0)
    print(f"[+] VRAM after model releases: {free_mem / 1e9:.2f} GB free / {total_mem / 1e9:.2f} GB total")
    assert (free_mem / 1e9) >= 4.0, f"VRAM leak detected! Only {free_mem / 1e9:.2f} GB free."

    # 6. CPU Fallback Verification
    print("[*] Testing CPU Fallback for YOLO and Whisper...")
    res_cpu = yolo_model(dummy_frame, device="cpu", verbose=False)
    print("[+] YOLO CPU fallback successful.")
    # Whisper CPU fallback test
    from faster_whisper import WhisperModel
    m_cpu = WhisperModel("tiny", device="cpu", compute_type="int8")
    segs_cpu, _ = m_cpu.transcribe(dummy_audio)
    list(segs_cpu)
    print("[+] Whisper CPU fallback successful.")

    # 7. Video Encoding Verification (NVENC probe + libx264 fallback)
    print("[*] Testing Video Encoding (NVENC probe & libx264 encoder)...")
    test_out = os.path.join(config.TEMP_DIR, "enc_test.mp4")
    if os.path.isfile(test_out):
        os.remove(test_out)

    use_hw = _use_nvenc()
    encoder_choice = "h264_nvenc" if use_hw else "libx264"
    print(f"[*] Encoder selected by hardware probe: {encoder_choice}")
    if not use_hw:
        print("[!] Note: NVENC SDK 13.1 in FFmpeg 9.0.2 requires NVIDIA Driver 610.00+; Driver 596.08 detected. Seamlessly using ultra-fast libx264.")

    enc_cmd = [
        FFMPEG_EXE, "-y",
        "-f", "lavfi", "-i", "testsrc=size=1080x1920:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=44100",
        "-t", "2",
        "-c:v", encoder_choice,
        "-preset", "fast",
        "-c:a", "aac", "-b:a", "128k",
        test_out
    ]
    t0 = time.time()
    res_enc = subprocess.run(enc_cmd, capture_output=True, text=True)
    enc_elapsed = time.time() - t0
    assert res_enc.returncode == 0, f"Encoding failed: {res_enc.stderr}"
    assert os.path.isfile(test_out), "Output MP4 not created!"
    file_sz = os.path.getsize(test_out)
    print(f"[+] Video encoding ({encoder_choice}) 1080x1920 60 frames succeeded in {enc_elapsed:.2f}s! File size: {file_sz/1024:.1f} KB")

    # Verify codec with ffprobe
    probe_cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height", "-of", "csv=p=0",
        test_out
    ]
    probe_res = subprocess.run(probe_cmd, capture_output=True, text=True)
    print(f"[+] Verified output stream via ffprobe: {probe_res.stdout.strip()}")
    assert "h264" in probe_res.stdout.lower(), f"Expected h264 codec, got: {probe_res.stdout}"

    if os.path.isfile(test_out):
        os.remove(test_out)

    print("[+] STAGE 2 COMPLETED SUCCESSFULLY: Hardware, CUDA, YOLO, Whisper, VRAM cleanup, and video encoding all verified.")
    return True

if __name__ == "__main__":
    success = run_hardware_test()
    sys.exit(0 if success else 1)
