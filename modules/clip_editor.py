"""
modules/clip_editor.py
FFmpeg-based clip cutting: NVENC / libx264 fallback, ASS captions, vertical crop,
blurred background, output package (clip.mp4 + thumbnail + metadata + subtitles).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import subprocess
import json
import shutil
import re
import logging

logger = logging.getLogger(__name__)


def _find_ffmpeg() -> str:
    """Return the ffmpeg executable path."""
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    winget_path = (
        r"C:\Users\choga\AppData\Local\Microsoft\WinGet\Packages"
        r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
        r"\ffmpeg-9.0.2-full_build\bin\ffmpeg.EXE"
    )
    if os.path.isfile(winget_path):
        return winget_path
    return "ffmpeg"


FFMPEG_EXE = _find_ffmpeg()
_NVENC_AVAILABLE = None


def _use_nvenc() -> bool:
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is None:
        if not config.USE_NVENC:
            _NVENC_AVAILABLE = False
            return False
        try:
            # Test actual 1-frame NVENC encoding capability to catch driver/SDK mismatch (e.g. driver 596 vs SDK 13.1 requiring 610+)
            probe_cmd = [
                FFMPEG_EXE, "-y",
                "-f", "lavfi", "-i", "nullsrc=s=64x64:d=0.1",
                "-c:v", "h264_nvenc",
                "-f", "null", "-"
            ]
            r = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=5)
            _NVENC_AVAILABLE = (r.returncode == 0)
            if not _NVENC_AVAILABLE:
                logger.info("NVENC probed but unavailable with current NVIDIA driver; using libx264.")
        except Exception:
            _NVENC_AVAILABLE = False
    return _NVENC_AVAILABLE


def detect_dead_air(
    source_path: str,
    start_sec: float,
    end_sec: float,
    words_list: list | None = None,
    min_pause_sec: float = None,
    noise_thresh_db: float = None,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], float]:
    """
    Detects dead air and speech pauses inside [start_sec, end_sec].
    Combines Whisper word-gap analysis with FFmpeg silencedetect.
    Returns (cut_intervals, keep_intervals, total_silence_removed).
    Intervals are relative to clip start (0.0 to clip_dur).
    """
    min_pause_sec = min_pause_sec or config.DEAD_AIR_MIN_PAUSE_SEC
    noise_thresh_db = noise_thresh_db or config.DEAD_AIR_NOISE_THRESH_DB
    clip_dur = max(0.5, float(end_sec - start_sec))
    candidate_cuts = []

    # 1. Dialogue pause detection via word timestamps
    if words_list:
        sub_words = [
            w for w in words_list
            if w.get("end", 0) > start_sec and w.get("start", 0) < end_sec
        ]
        sub_words.sort(key=lambda x: x.get("start", 0))

        # Check for leading dead air before speech starts
        if sub_words and (sub_words[0].get("start", 0) - start_sec) > (min_pause_sec + 0.15):
            lead_end = max(0.0, sub_words[0]["start"] - start_sec - config.DEAD_AIR_PAD_SEC)
            if lead_end >= 0.2:
                candidate_cuts.append((0.0, round(lead_end, 3)))

        # Gaps between spoken words
        for i in range(len(sub_words) - 1):
            w_curr_end = max(0.0, sub_words[i].get("end", 0) - start_sec)
            w_next_start = min(clip_dur, sub_words[i + 1].get("start", 0) - start_sec)
            gap = w_next_start - w_curr_end
            if gap >= min_pause_sec:
                c_s = w_curr_end + config.DEAD_AIR_PAD_SEC
                c_e = w_next_start - config.DEAD_AIR_PAD_SEC
                if c_e - c_s >= 0.15:
                    candidate_cuts.append((round(c_s, 3), round(c_e, 3)))

        # Trailing dead air after speech finishes
        if sub_words and (end_sec - sub_words[-1].get("end", 0)) > (min_pause_sec + 0.25):
            trail_start = sub_words[-1]["end"] - start_sec + config.DEAD_AIR_PAD_SEC
            if clip_dur - trail_start >= 0.25:
                candidate_cuts.append((round(trail_start, 3), round(clip_dur, 3)))

    # 2. Audio-level silence detection fallback (FFmpeg)
    try:
        cmd = [
            FFMPEG_EXE, "-y",
            "-ss", str(start_sec),
            "-i", source_path,
            "-t", str(clip_dur),
            "-af", f"silencedetect=noise={noise_thresh_db}dB:d={min_pause_sec}",
            "-f", "null", "-"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=12)
        starts = [float(x) for x in re.findall(r"silence_start:\s*([\d\.]+)", res.stderr)]
        ends = [float(x) for x in re.findall(r"silence_end:\s*([\d\.]+)", res.stderr)]
        for s, e in zip(starts, ends):
            s_rel = max(0.0, min(clip_dur, s + config.DEAD_AIR_PAD_SEC))
            e_rel = max(0.0, min(clip_dur, e - config.DEAD_AIR_PAD_SEC))
            if e_rel - s_rel >= 0.2:
                candidate_cuts.append((round(s_rel, 3), round(e_rel, 3)))
    except Exception as e:
        logger.debug("silencedetect note: %s", e)

    if not candidate_cuts:
        return [], [(0.0, round(clip_dur, 3))], 0.0

    # 3. Merge overlapping cuts
    candidate_cuts.sort(key=lambda x: x[0])
    merged_cuts = []
    for s, e in candidate_cuts:
        s = max(0.0, min(clip_dur, s))
        e = max(0.0, min(clip_dur, e))
        if e <= s:
            continue
        if not merged_cuts:
            merged_cuts.append([s, e])
        elif s <= merged_cuts[-1][1]:
            merged_cuts[-1][1] = max(merged_cuts[-1][1], e)
        else:
            merged_cuts.append([s, e])

    # 4. Generate keep intervals
    keep_intervals = []
    cur = 0.0
    total_removed = 0.0
    final_cuts = []
    for s, e in merged_cuts:
        if s - cur >= 0.25:
            keep_intervals.append((round(cur, 3), round(s, 3)))
            final_cuts.append((round(s, 3), round(e, 3)))
            total_removed += (e - s)
            cur = e

    if clip_dur - cur >= 0.25:
        keep_intervals.append((round(cur, 3), round(clip_dur, 3)))
    elif keep_intervals:
        last_s, _ = keep_intervals[-1]
        keep_intervals[-1] = (last_s, round(clip_dur, 3))

    if not keep_intervals:
        keep_intervals = [(0.0, round(clip_dur, 3))]
        final_cuts = []
        total_removed = 0.0

    return final_cuts, keep_intervals, round(total_removed, 2)


def cut_clip(
    source_path: str,
    start_sec: float,
    end_sec: float,
    output_path: str,
    ass_path: str | None = None,
    vertical: bool = True,
    crop_style: str = "center",  # "center" (9:16 crop) | "blur_bg" (16:9 in 9:16 blurred bg)
    src_width: int = 0,
    src_height: int = 0,
    keep_intervals: list[tuple[float, float]] | None = None,
) -> tuple[bool, str]:
    """
    Cut a clip from source_path using FFmpeg.
    If keep_intervals has multiple segments, applies seamless jump-cuts to eliminate dead air.
    Returns (success: bool, output_path_or_error_msg: str).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    duration = max(1.0, float(end_sec - start_sec))
    out_w = config.EXPORT_WIDTH   # 1080
    out_h = config.EXPORT_HEIGHT  # 1920

    # Case A: Jump-cuts (Multiple keep intervals after dead-air removal)
    if keep_intervals and len(keep_intervals) > 1:
        f_parts = []
        concat_v_in = []
        concat_a_in = []
        for i, (k_s, k_e) in enumerate(keep_intervals):
            abs_s = start_sec + k_s
            abs_e = start_sec + k_e
            f_parts.append(f"[0:v]trim=start={abs_s}:end={abs_e},setpts=PTS-STARTPTS[v{i}]")
            f_parts.append(f"[0:a]atrim=start={abs_s}:end={abs_e},asetpts=PTS-STARTPTS[a{i}]")
            concat_v_in.append(f"[v{i}]")
            concat_a_in.append(f"[a{i}]")

        n_seg = len(keep_intervals)
        stream_in = "".join(v + a for v, a in zip(concat_v_in, concat_a_in))
        f_parts.append(f"{stream_in}concat=n={n_seg}:v=1:a=1[v_cat][a_cat]")

        # Framing on concatenated stream
        if vertical:
            if crop_style == "blur_bg" and src_width > 0 and src_height > 0:
                f_parts.append(
                    f"[v_cat]split=2[bg][fg];"
                    f"[bg]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
                    f"crop={out_w}:{out_h},boxblur=20:5[bg_b];"
                    f"[fg]scale={out_w}:-2[fg_s];"
                    f"[bg_b][fg_s]overlay=(W-w)/2:(H-h)/2[v_framed]"
                )
            else:
                f_parts.append(f"[v_cat]crop=ih*9/16:ih,scale={out_w}:{out_h},setsar=1[v_framed]")
        else:
            f_parts.append(f"[v_cat]null[v_framed]")

        # Burn ASS subtitles if provided
        if ass_path and os.path.isfile(ass_path):
            ass_escaped = os.path.abspath(ass_path).replace("\\", "/").replace(":", "\\:")
            f_parts.append(f"[v_framed]ass='{ass_escaped}'[v_final]")
            out_v_tag = "[v_final]"
        else:
            out_v_tag = "[v_framed]"

        filter_complex = ";".join(f_parts)
        cmd = [
            FFMPEG_EXE, "-y",
            "-i", source_path,
            "-filter_complex", filter_complex,
            "-map", out_v_tag,
            "-map", "[a_cat]",
        ]
        return _execute_ffmpeg(cmd, output_path, source_path, start_sec, end_sec, ass_path, vertical, crop_style, src_width, src_height, keep_intervals)

    # Case B: Standard single-segment cut
    filters = []
    if vertical:
        if crop_style == "blur_bg" and src_width > 0 and src_height > 0:
            filter_complex = (
                f"[0:v]split=2[bg][fg];"
                f"[bg]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
                f"crop={out_w}:{out_h},boxblur=20:5[bg_b];"
                f"[fg]scale={out_w}:-2[fg_s];"
                f"[bg_b][fg_s]overlay=(W-w)/2:(H-h)/2[v_out]"
            )
            if ass_path and os.path.isfile(ass_path):
                ass_escaped = os.path.abspath(ass_path).replace("\\", "/").replace(":", "\\:")
                filter_complex += f";[v_out]ass='{ass_escaped}'[final_v]"
                video_map = "[final_v]"
            else:
                video_map = "[v_out]"

            cmd = [
                FFMPEG_EXE, "-y",
                "-ss", str(start_sec),
                "-i", source_path,
                "-t", str(duration),
                "-filter_complex", filter_complex,
                "-map", video_map,
                "-map", "0:a?",
            ]
            return _execute_ffmpeg(cmd, output_path, source_path, start_sec, end_sec, ass_path, vertical, "center", src_width, src_height)
        else:
            filters.append(f"crop=ih*9/16:ih,scale={out_w}:{out_h},setsar=1")

    if ass_path and os.path.isfile(ass_path):
        ass_escaped = os.path.abspath(ass_path).replace("\\", "/").replace(":", "\\:")
        filters.append(f"ass='{ass_escaped}'")

    cmd = [
        FFMPEG_EXE, "-y",
        "-ss", str(start_sec),
        "-i", source_path,
        "-t", str(duration),
    ]
    if filters:
        cmd += ["-vf", ",".join(filters)]

    cmd += ["-map", "0:v:0", "-map", "0:a:0?"]
    return _execute_ffmpeg(cmd, output_path, source_path, start_sec, end_sec, ass_path, vertical, crop_style, src_width, src_height)


def _execute_ffmpeg(cmd, output_path, source_path, start_sec, end_sec, ass_path, vertical, crop_style, src_width, src_height, keep_intervals=None):
    use_hw = _use_nvenc()
    codec_args = (
        ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", str(config.EXPORT_CRF)]
        if use_hw else
        ["-c:v", "libx264", "-preset", config.EXPORT_PRESET, "-crf", str(config.EXPORT_CRF)]
    )

    full_cmd = cmd + codec_args + [
        "-r", str(config.EXPORT_FPS),
        "-c:a", "aac", "-b:a", config.EXPORT_AUDIO_BITRATE, "-ar", "44100",
        output_path
    ]

    try:
        res = subprocess.run(full_cmd, capture_output=True, text=True, timeout=600)
        if res.returncode != 0:
            # If NVENC failed, retry on CPU libx264
            if use_hw:
                logger.warning("NVENC failed, retrying on libx264: %s", res.stderr[-500:])
                global _NVENC_AVAILABLE
                _NVENC_AVAILABLE = False
                return cut_clip(source_path, start_sec, end_sec, output_path, ass_path, vertical, crop_style, src_width, src_height, keep_intervals=keep_intervals)
            return False, f"FFmpeg error:\n{res.stderr[-2000:]}"
        return True, output_path
    except subprocess.TimeoutExpired:
        return False, "FFmpeg timed out (>600s)"
    except Exception as e:
        return False, str(e)


def build_clip_package(
    clip_id: str,
    source_path: str,
    start_sec: float,
    end_sec: float,
    ass_content: str | None = None,
    srt_content: str | None = None,
    vertical: bool = True,
    crop_style: str = "center",
    src_width: int = 0,
    src_height: int = 0,
    metadata: dict | None = None,
    output_root: str | None = None,
    remove_dead_air: bool = True,
) -> dict:
    """
    Build complete export package for a single clip:
      output/{clip_id}/
        clip.mp4
        thumbnail.jpg
        subtitles.ass
        subtitles.srt
        metadata.json
        title.txt
        description.txt
    """
    output_root = output_root or config.OUTPUT_DIR
    pkg_dir = os.path.join(output_root, clip_id)
    os.makedirs(pkg_dir, exist_ok=True)

    result = {
        "clip_id":        clip_id,
        "pkg_dir":        pkg_dir,
        "clip_path":      None,
        "ass_path":       None,
        "srt_path":       None,
        "thumbnail_path": None,
        "metadata_path":  None,
        "silence_removed": 0.0,
        "success":        False,
        "error":          None,
    }

    # 1. Dead air / pause detection (jump-cuts)
    keep_intervals = None
    cut_intervals = []
    total_removed = 0.0
    words_list = metadata.get("words_list") if metadata else None

    if remove_dead_air:
        cut_intervals, keep_intervals, total_removed = detect_dead_air(
            source_path, start_sec, end_sec, words_list=words_list
        )
        result["silence_removed"] = total_removed
        if metadata is not None:
            metadata["silence_removed_sec"] = total_removed
            metadata["cut_intervals"] = cut_intervals

        # If silence was removed and we have words, remap subtitles
        if cut_intervals and words_list:
            from modules.subtitle_generator import remap_words_for_dead_air, generate_clip_ass
            remapped_words = remap_words_for_dead_air(words_list, cut_intervals, start_sec, end_sec)
            pos = metadata.get("caption_position", "Bottom (Shorts/TikTok)") if metadata else "Bottom (Shorts/TikTok)"
            ass_content = generate_clip_ass(
                words_list=remapped_words,
                clip_start=0.0,
                clip_end=max(1.0, (end_sec - start_sec) - total_removed),
                crop_vertical=vertical,
                position=pos,
                censor_swear_words=True,
                dynamic_highlight=True,
            )

    # 2. Write ASS & SRT files
    ass_path = None
    if ass_content:
        ass_path = os.path.join(pkg_dir, "subtitles.ass")
        with open(ass_path, "w", encoding="utf-8") as f:
            f.write(ass_content)
        result["ass_path"] = ass_path

    if srt_content:
        srt_path = os.path.join(pkg_dir, "subtitles.srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)
        result["srt_path"] = srt_path

    # 3. Cut video (single segment or seamless jump-cuts)
    clip_path = os.path.join(pkg_dir, "clip.mp4")
    success, msg = cut_clip(
        source_path, start_sec, end_sec, clip_path,
        ass_path=ass_path, vertical=vertical, crop_style=crop_style,
        src_width=src_width, src_height=src_height,
        keep_intervals=keep_intervals,
    )

    if not success:
        result["error"] = msg
        return result

    result["clip_path"] = clip_path
    result["success"]   = True

    # 4. Extract thumbnail
    from modules.thumbnail_generator import extract_thumbnail
    thumb_path = os.path.join(pkg_dir, "thumbnail.jpg")
    peak = metadata.get("peak_second") if metadata else None
    thumb = extract_thumbnail(clip_path, thumb_path, peak_second=peak)
    result["thumbnail_path"] = thumb

    # 5. Text files & metadata.json
    if metadata:
        title = metadata.get("selected_title") or metadata.get("title", "")
        if title:
            with open(os.path.join(pkg_dir, "title.txt"), "w", encoding="utf-8") as f:
                f.write(title)
        desc = metadata.get("yt_description") or metadata.get("description", "")
        if desc:
            with open(os.path.join(pkg_dir, "description.txt"), "w", encoding="utf-8") as f:
                f.write(desc)

        meta_path = os.path.join(pkg_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        result["metadata_path"] = meta_path

    return result


def zip_all_clips(output_root: str | None = None, zip_path: str | None = None) -> str | None:
    """Zip all exported clip packages into one archive."""
    output_root = output_root or config.OUTPUT_DIR
    zip_path = zip_path or os.path.join(config.BASE_DIR, "gta6_clips_export.zip")

    if not os.path.isdir(output_root):
        return None

    with shutil.ZipFile(zip_path, "w", shutil.ZIP_DEFLATED) if hasattr(shutil, "ZipFile") else __import__("zipfile").ZipFile(zip_path, "w", __import__("zipfile").ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(output_root):
            for file in files:
                abs_path = os.path.join(root, file)
                arc_name = os.path.relpath(abs_path, output_root)
                zf.write(abs_path, arc_name)

    return zip_path if os.path.isfile(zip_path) else None
