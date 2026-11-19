"""tests/audit_whisper_continuity.py
Detailed comparison of Whisper transcription before and after chunking on real footage.
Validates boundary sentence preservation, word timestamps, and lack of duplication/omission.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from modules.speech_recognition import transcribe_audio, _get_model
from modules.audio_analysis import load_audio

test_file = os.path.join(config.INPUT_DIR, "321206426716_20261118_165332_live_segment.mkv")

print(f"Loading test file: {os.path.basename(test_file)}")
y, sr = load_audio(test_file)
total_dur = len(y) / sr
print(f"Audio duration: {total_dur:.1f}s")

# 1. Monolithic transcription (Single pass, no chunking)
print("\n1. Running Monolithic Transcription (no chunking)...")
trans_mono, words_mono = transcribe_audio(y, chunk_duration_sec=99999.0, vad_filter=False)
print(f"   Monolithic: {len(trans_mono)} segments, {len(words_mono)} words")

# 2. Chunked transcription (60s chunks, cutting across sentences at 60s, 120s, 180s)
print("\n2. Running Chunked Transcription (60.0s chunks)...")
trans_chunk, words_chunk = transcribe_audio(test_file, chunk_duration_sec=60.0, vad_filter=False)
print(f"   Chunked: {len(trans_chunk)} segments, {len(words_chunk)} words")

# 3. Inspect Boundary Windows [55s - 65s], [115s - 125s], [175s - 185s]
print("\n3. Comparing Speech at Chunk Boundaries:")
for boundary in [60.0, 120.0, 180.0]:
    print(f"\n--- Checking Boundary around {boundary:.0f}s [{boundary-5:.0f}s – {boundary+5:.0f}s] ---")
    mono_words = [w for w in words_mono if (boundary - 5.0) <= w["start"] <= (boundary + 5.0)]
    chunk_words = [w for w in words_chunk if (boundary - 5.0) <= w["start"] <= (boundary + 5.0)]

    mono_txt = " ".join(w["word"] for w in mono_words)
    chunk_txt = " ".join(w["word"] for w in chunk_words)
    print(f"   Mono text:  '{mono_txt}'")
    print(f"   Chunk text: '{chunk_txt}'")

    # Check for duplicated consecutive words in chunked
    chunk_word_list = [w["word"].lower() for w in chunk_words]
    dups = [chunk_word_list[i] for i in range(len(chunk_word_list)-1) if chunk_word_list[i] == chunk_word_list[i+1]]
    if dups:
        print(f"   ⚠️ Duplicate adjacent words detected: {dups}")
    else:
        print(f"   ✅ No adjacent duplicate words.")

    # Check timestamp offsets
    if mono_words and chunk_words:
        drift = abs(mono_words[0]["start"] - chunk_words[0]["start"])
        print(f"   Boundary start timestamp delta: {drift:.2f}s")

# 4. Check Final Incomplete Chunk
last_chunk_start = 3 * 60.0 # 180.0
last_chunk_end = total_dur # 230.1s (incomplete 50s chunk)
print(f"\n4. Final Incomplete Chunk [{last_chunk_start:.0f}s – {last_chunk_end:.1f}s]:")
final_mono = [s for s in trans_mono if s["start"] >= last_chunk_start]
final_chunk = [s for s in trans_chunk if s["start"] >= last_chunk_start]
print(f"   Monolithic segments in final chunk: {len(final_mono)}")
print(f"   Chunked segments in final chunk:    {len(final_chunk)}")
print(f"   Last mono seg:  {final_mono[-1] if final_mono else None}")
print(f"   Last chunk seg: {final_chunk[-1] if final_chunk else None}")
