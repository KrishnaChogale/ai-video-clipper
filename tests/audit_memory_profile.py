"""tests/audit_memory_profile.py
Memory profiling during processing of 30-minute real streamer footage:
input/321206426716_20261118_171844_live_segment.mkv (1800.2s = 30.0 min, 1.75 GB).
"""
import os
import sys
import time
import gc
import psutil
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from modules.video_ingestion import probe_video
from modules.audio_analysis import analyze_audio, clear_audio_cache
from modules.speech_recognition import transcribe_audio, clear_model_cache
from modules.object_detection import analyze_visuals

proc = psutil.Process()
test_file = os.path.join(config.INPUT_DIR, "321206426716_20261118_171844_live_segment.mkv")

print(f"Profiling source: {os.path.basename(test_file)}")
v = probe_video(test_file)
duration = float(v.get("duration", 0))
print(f"Duration: {duration:.1f}s ({duration/60:.1f} min)")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

mem_0 = proc.memory_info().rss / (1024**2)
vram_0 = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
print(f"1. Baseline RSS: {mem_0:.1f} MB, VRAM: {vram_0:.1f} MB")

# 1. Chunked Audio Energy
print("\n2. Executing Chunked Audio Energy...")
t0 = time.perf_counter()
energy_scores, audio_y, sr = analyze_audio(test_file, chunk_if_long=True)
mem_audio = proc.memory_info().rss / (1024**2)
t_audio = time.perf_counter() - t0
print(f"   Audio energy done in {t_audio:.2f}s ({len(energy_scores)}s buckets)")
print(f"   RSS after audio: {mem_audio:.1f} MB (Delta: +{mem_audio - mem_0:.1f} MB)")
print(f"   Is raw audio array in memory? {audio_y is not None}")
clear_audio_cache()
gc.collect()
mem_audio_cleared = proc.memory_info().rss / (1024**2)
print(f"   RSS after clear_audio_cache(): {mem_audio_cleared:.1f} MB")

# 2. Chunked Whisper Transcription
print("\n3. Executing Chunked Whisper Transcription (300s slice across chunks)...")
t0 = time.perf_counter()
transcript, words = transcribe_audio(test_file, chunk_duration_sec=150.0)
mem_whisper = proc.memory_info().rss / (1024**2)
vram_whisper = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
t_whisper = time.perf_counter() - t0
print(f"   Whisper done in {t_whisper:.2f}s ({len(transcript)} segs, {len(words)} words)")
print(f"   RSS after Whisper: {mem_whisper:.1f} MB (Delta: +{mem_whisper - mem_audio_cleared:.1f} MB)")
print(f"   VRAM allocated: {vram_whisper:.1f} MB")

# 3. Visual Analysis (Sample 60s)
print("\n4. Executing Visual Analysis (60s slice)...")
t0 = time.perf_counter()
visual_scores = analyze_visuals(test_file, duration_sec=60.0)
mem_vis = proc.memory_info().rss / (1024**2)
vram_vis = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
t_vis = time.perf_counter() - t0
print(f"   Visual analysis done in {t_vis:.2f}s ({len(visual_scores)} frames)")
print(f"   RSS after Visual: {mem_vis:.1f} MB (Delta: +{mem_vis - mem_whisper:.1f} MB)")
print(f"   VRAM allocated: {vram_vis:.1f} MB")

# 4. Clean up
clear_model_cache()
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
mem_final = proc.memory_info().rss / (1024**2)
vram_final = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
print(f"\n5. Final RSS after cleanup: {mem_final:.1f} MB, VRAM: {vram_final:.1f} MB")
print(f"   Peak RAM increase: +{max(mem_audio, mem_whisper, mem_vis) - mem_0:.1f} MB")
