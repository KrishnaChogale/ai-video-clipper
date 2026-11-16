"""
tests/test_modules.py
Unit tests verifying all core modules of the GTA 6 AI Auto Clipper.
Run: cd C:\\ai-video-clipper && .\\venv\\Scripts\\python.exe -m pytest tests/test_modules.py -v
"""
import os
import sys
import numpy as np
import sqlite3
import tempfile
import pathlib

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import config
from modules.gpu_utils import get_device_info, select_device
from modules.subtitle_generator import censor_profanity, format_ass_time, format_srt_time, generate_clip_ass, generate_clip_srt
from modules.audio_analysis import compute_energy_buckets
from modules.highlight_scoring import compute_per_second_scores, select_highlights
from modules.event_classification import classify_events
from modules.title_generator import generate_titles, generate_hooks
from modules.description_generator import generate_descriptions, generate_hashtags
from modules.quality_control import evaluate_clip
from modules.metadata_manager import init_db, save_clip_metadata, get_all_clips, save_performance_metrics


def test_gpu_utils():
    info = get_device_info()
    assert isinstance(info, dict)
    assert "cuda_available" in info
    assert "vram_free_gb" in info
    assert "ffmpeg_path" in info

    device = select_device("auto")
    assert device in ("cuda", "cpu")


def test_censor_profanity():
    raw_text = "What the fuck is this shit, you motherfucker?"
    censored = censor_profanity(raw_text)
    assert "fuck" not in censored.lower() or "f*ck" in censored.lower()
    assert "shit" not in censored.lower() or "sh*t" in censored.lower()
    assert "motherfucker" not in censored.lower() or "f*ck" in censored.lower()


def test_ass_and_srt_formatting():
    assert format_ass_time(65.5) == "0:01:05.50"
    assert format_srt_time(65.5) == "00:01:05,500"

    mock_transcript = [
        {"start": 10.0, "end": 14.0, "text": "Look at that police chase!"},
        {"start": 15.0, "end": 18.0, "text": "What the fuck happened?!"},
    ]
    mock_words = [
        {"start": 10.0, "end": 11.0, "word": "Look"},
        {"start": 11.0, "end": 12.0, "word": "at"},
        {"start": 12.0, "end": 13.0, "word": "that"},
        {"start": 13.0, "end": 14.0, "word": "police"},
    ]

    ass = generate_clip_ass(words_list=mock_words, transcript=mock_transcript, clip_start=10.0, clip_end=20.0)
    assert "[Script Info]" in ass
    assert "[Events]" in ass
    assert "POLICE" in ass

    srt = generate_clip_srt(mock_transcript, clip_start=10.0, clip_end=20.0, censor=True)
    assert "00:00:" in srt
    assert "F*CK" in srt or "f*ck" in srt.lower()


def test_audio_energy_buckets():
    sr = 16000
    duration = 5
    # Synthetic sine wave (440 Hz)
    t = np.linspace(0, duration, sr * duration, endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    buckets = compute_energy_buckets(y, sr)
    assert len(buckets) == duration
    assert buckets[0]["loudness"] > 0
    assert "spike" in buckets[0]


def test_highlight_scoring_and_selection():
    # 30 seconds of mock data
    energy_buckets = [{"second": i, "loudness": 0.1, "spike": 2.0, "is_gunshot": False, "is_explosion": False} for i in range(30)]
    # Create an artificial peak at seconds 10-15
    for i in range(10, 15):
        energy_buckets[i]["loudness"] = 0.8
        energy_buckets[i]["spike"] = 15.0
        energy_buckets[i]["is_explosion"] = True

    visual_scores = [{"second": i, "visual_score": 1.0, "motion": 10.0, "scene_change": False} for i in range(30)]
    for i in range(10, 15):
        visual_scores[i]["visual_score"] = 5.0
        visual_scores[i]["motion"] = 80.0

    transcript = [{"start": 10.0, "end": 14.0, "text": "Insane explosion happening right now!"}]

    events = classify_events(energy_buckets, visual_scores, transcript)
    assert len(events) == 30
    assert "action" in [e["category"] for e in events]

    scores = compute_per_second_scores(energy_buckets, visual_scores, transcript, events)
    assert len(scores) == 30

    highlights = select_highlights(scores, event_classification=events, top_k=2, window_sec=10, min_score=5.0)
    assert len(highlights) >= 1
    # Check that winning clip contains the peak region
    assert highlights[0]["start"] <= 14


def test_title_and_hook_generation():
    clip_mock = {
        "transcript_snippet": "Look at the cops chasing that sports car through Vice City!",
        "category": "action",
        "events": ["police_chase", "high_speed"],
        "start": 10,
        "end": 35,
    }

    titles = generate_titles(clip_mock, style="action")
    assert len(titles) == 3
    assert all(isinstance(t, str) and len(t) > 0 for t in titles)

    hooks = generate_hooks(clip_mock, style="action")
    assert len(hooks) == 3
    assert all(isinstance(h, str) and len(h) > 0 for h in hooks)


def test_description_and_hashtag_generation():
    clip_mock = {
        "transcript_snippet": "The NPC just jumped off the roof for no reason at all!",
        "category": "funny",
        "events": ["npc_reaction"],
        "selected_title": "GTA 6 NPC Has No Brain",
    }

    descs = generate_descriptions(clip_mock, style="funny")
    assert "YouTube Shorts" in descs
    assert "Instagram Reels" in descs
    assert "TikTok" in descs
    assert "yt_description" in descs

    hashtags = generate_hashtags(clip_mock, count=5)
    assert len(hashtags) == 5
    assert "#GTA6" in hashtags[0]


def test_quality_control():
    # Test passing clip
    res = evaluate_clip(
        clip_path="non_existent.mp4",
        start_sec=10,
        end_sec=30,
        score=45.0,
        min_score=10.0,
        min_duration=15,
        max_duration=60,
    )
    assert isinstance(res, dict)
    assert "details" in res
    assert res["details"]["duration"] == 20.0


def test_database_and_metadata(tmp_path):
    test_db = str(tmp_path / "test_clips.db")
    conn = init_db(test_db)
    assert conn is not None

    clip_record = {
        "clip_id": "clip_test_01",
        "source_path": "test.mp4",
        "start_sec": 5.0,
        "end_sec": 25.0,
        "duration": 20.0,
        "category": "action",
        "score": 85.5,
        "hook": "WAIT FOR THE CRASH 💀",
        "alt_hooks": ["WAIT FOR IT", "NO WAY"],
        "selected_title": "GTA 6 Crazy Crash",
        "alt_titles": ["Crash 1", "Crash 2", "Crash 3"],
        "yt_description": "Insane GTA 6 crash footage!",
        "ig_description": "Unreal GTA 6 moments!",
        "tt_description": "Bro forgot how to steer 💀",
        "hashtags": ["#GTA6", "#Gaming"],
        "thumbnail_path": "thumb.jpg",
        "clip_path": "clip.mp4",
        "qc_passed": 1,
    }

    cid = save_clip_metadata(clip_record, db_path=test_db)
    assert cid == "clip_test_01"

    all_clips = get_all_clips(db_path=test_db)
    assert len(all_clips) == 1
    assert all_clips[0]["selected_title"] == "GTA 6 Crazy Crash"

    # Test performance metrics
    save_performance_metrics(
        "clip_test_01",
        "YouTube Shorts",
        {"views": 5000, "likes": 420, "shares": 50, "comments": 25, "avg_watch_sec": 18.2, "completion_rate": 91.0, "subs_gained": 12},
        db_path=test_db,
    )

    cur = conn.cursor()
    cur.execute("SELECT views, likes FROM clip_performance WHERE clip_id=?", ("clip_test_01",))
    row = cur.fetchone()
    assert row["views"] == 5000
    assert row["likes"] == 420
    conn.close()


if __name__ == "__main__":
    print("Testing gpu_utils...")
    test_gpu_utils()
    print("✓ test_gpu_utils passed")

    print("Testing censor_profanity...")
    test_censor_profanity()
    print("✓ test_censor_profanity passed")

    print("Testing ass_and_srt_formatting...")
    test_ass_and_srt_formatting()
    print("✓ test_ass_and_srt_formatting passed")

    print("Testing audio_energy_buckets...")
    test_audio_energy_buckets()
    print("✓ test_audio_energy_buckets passed")

    print("Testing highlight_scoring_and_selection...")
    test_highlight_scoring_and_selection()
    print("✓ test_highlight_scoring_and_selection passed")

    print("Testing title_and_hook_generation...")
    test_title_and_hook_generation()
    print("✓ test_title_and_hook_generation passed")

    print("Testing description_and_hashtag_generation...")
    test_description_and_hashtag_generation()
    print("✓ test_description_and_hashtag_generation passed")

    print("Testing quality_control...")
    test_quality_control()
    print("✓ test_quality_control passed")

    print("Testing database_and_metadata...")
    with tempfile.TemporaryDirectory() as td:
        test_database_and_metadata(pathlib.Path(td))
    print("✓ test_database_and_metadata passed")

    print("\n🎉 ALL 9 TEST SUITES PASSED PERFECTLY!")
