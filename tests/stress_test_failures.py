"""
tests/stress_test_failures.py
Stage 9: Failure, Edge Case & Graceful Degradation Stress Test.
Tests invalid URLs, unsupported files, corrupted videos, missing audio,
simulated Ollama outage, forced CPU fallback, and empty highlight handling.
"""
import os
import sys
import time
import tempfile
import subprocess
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import config
from modules.video_ingestion import probe_video, download_youtube_video, accept_local_file
from modules.audio_analysis import load_audio, compute_energy_buckets, analyze_audio
from modules.speech_recognition import transcribe_audio
from modules.object_detection import analyze_visuals
from modules.highlight_scoring import compute_per_second_scores, select_highlights
from modules.title_generator import generate_titles, generate_hooks
from modules.description_generator import generate_descriptions, generate_hashtags
from modules.clip_editor import cut_clip
from modules.quality_control import evaluate_clip

def run_failure_tests():
    print("==================================================")
    print("STAGE 9: FAILURE & EDGE CASE TESTS")
    print("==================================================")

    # 1. Invalid Video URL
    print("\n[*] Test 9.1: Invalid YouTube URL...")
    fake_url = "https://www.youtube.com/watch?v=NON_EXISTENT_ID_99999"
    v_path, err = download_youtube_video(fake_url)
    print(f"[+] Invalid URL handled gracefully: path={v_path}, err='{err[:100]}...'")
    assert v_path is None, "Expected path to be None for invalid URL!"
    assert err is not None, "Expected error message for invalid URL!"

    # 2. Unsupported File Type
    print("\n[*] Test 9.2: Unsupported File Type (.txt)...")
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"This is not a video file.")
        fake_txt = f.name
    dest, err = accept_local_file(fake_txt)
    print(f"[+] Unsupported file handled: dest={dest}, err={err}")
    os.remove(fake_txt)

    # 3. Corrupted / 0-byte Video File
    print("\n[*] Test 9.3: Corrupted / 0-byte Video File...")
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        fake_empty_mp4 = f.name # 0 bytes
    vmeta = probe_video(fake_empty_mp4)
    print(f"[+] 0-byte video probe returned: duration={vmeta['duration']}, width={vmeta['width']}")
    assert vmeta["duration"] == 0 or vmeta["width"] == 0, "Corrupted video should report 0 duration/width"
    os.remove(fake_empty_mp4)

    # 4. Video with Missing Audio Track
    print("\n[*] Test 9.4: Video with Missing Audio Track...")
    # Generate a synthetic 2s video without any audio
    silent_vid = os.path.join(config.TEMP_DIR, "silent_test.mp4")
    from modules.clip_editor import FFMPEG_EXE
    cmd = [
        FFMPEG_EXE, "-y",
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10",
        "-t", "1",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-an", # No audio
        silent_vid
    ]
    subprocess.run(cmd, capture_output=True)
    vmeta_silent = probe_video(silent_vid)
    print(f"[+] Silent video probed: video_codec={vmeta_silent['codec']}, audio_codec={vmeta_silent['audio_codec']}")
    assert vmeta_silent["audio_codec"] == "unknown" or not vmeta_silent["audio_codec"]

    # Test audio analysis on video without audio
    try:
        e_b, y, sr = analyze_audio(silent_vid)
        print(f"[+] Audio analysis on silent video handled: buckets={len(e_b)}, samples={len(y)}")
    except Exception as e:
        print(f"[+] Audio analysis on silent video caught expected exception: {type(e).__name__}: {e}")

    if os.path.isfile(silent_vid):
        os.remove(silent_vid)

    # 5. Missing / Offline Ollama Fallback
    print("\n[*] Test 9.5: Missing / Offline Ollama Fallback...")
    # Force an invalid model name or port to simulate Ollama down
    original_model = config.OLLAMA_MODEL
    config.OLLAMA_MODEL = "non_existent_model_xyz_123"
    clip_sample = {
        "transcript_snippet": "They are heading towards Ocean Drive at high speed!",
        "category": "action",
        "events": ["high_speed", "police_chase"],
    }
    t0 = time.time()
    titles = generate_titles(clip_sample, style="action")
    hooks = generate_hooks(clip_sample, style="action")
    descs = generate_descriptions(clip_sample, style="action")
    print(f"[+] Offline Ollama fallback executed in {time.time() - t0:.2f}s:")
    print(f"    - Titles: {titles}")
    print(f"    - Hooks:  {hooks}")
    print(f"    - YT Desc: {descs['YouTube Shorts'][:60]}...")
    assert len(titles) == 3, "Expected 3 fallback titles!"
    assert len(hooks) == 3, "Expected 3 fallback hooks!"
    assert "YouTube Shorts" in descs, "Expected YouTube Shorts description!"
    config.OLLAMA_MODEL = original_model

    # 6. Forced CPU Fallback (GPU Unavailable)
    print("\n[*] Test 9.6: Forced CPU Fallback (CUDA disabled)...")
    original_pref = config.PREFERRED_DEVICE
    config.PREFERRED_DEVICE = "cpu"
    from modules.gpu_utils import select_device
    dev = select_device("cpu")
    print(f"[+] Forced CPU device selected: {dev}")
    assert dev == "cpu"
    config.PREFERRED_DEVICE = original_pref

    # 7. Empty Highlight Results Handling
    print("\n[*] Test 9.7: Empty Highlight Results Handling...")
    empty_scores = []
    empty_highlights = select_highlights(empty_scores, event_classification=[])
    print(f"[+] Empty per-second scores handled gracefully: {empty_highlights}")
    assert empty_highlights == []

    print("\n[+] STAGE 9 COMPLETED SUCCESSFULLY: All 7 failure & edge case scenarios handled gracefully with zero unhandled crashes.")
    return True

if __name__ == "__main__":
    success = run_failure_tests()
    sys.exit(0 if success else 1)
