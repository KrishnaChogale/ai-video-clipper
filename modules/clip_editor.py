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
) -> tuple[bool, str]:
    """
    Cut a clip from source_path using FFmpeg.
    Returns (success: bool, output_path_or_error_msg: str).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    duration = max(1.0, float(end_sec - start_sec))

    filters = []

    if vertical:
        out_w = config.EXPORT_WIDTH   # 1080
        out_h = config.EXPORT_HEIGHT  # 1920

        if crop_style == "blur_bg" and src_width > 0 and src_height > 0:
            # Blurred background extension
            filter_complex = (
                f"[0:v]split=2[bg][fg];"
                f"[bg]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
                f"crop={out_w}:{out_h},boxblur=20:5[bg_b];"
                f"[fg]scale={out_w}:-2[fg_s];"
                f"[bg_b][fg_s]overlay=(W-w)/2:(H-h)/2[v_out]"
            )
            # If ASS subtitles present
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
            # Classic center crop (Hormozi / TikTok 9:16 fullscreen)
            filters.append("crop=ih*9/16:ih,scale=1080:1920,setsar=1")

    # Burn ASS subtitles into video filter chain
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


def _execute_ffmpeg(cmd, output_path, source_path, start_sec, end_sec, ass_path, vertical, crop_style, src_width, src_height):
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
                return cut_clip(source_path, start_sec, end_sec, output_path, ass_path, vertical, crop_style, src_width, src_height)
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
        "success":        False,
        "error":          None,
    }

    # 1. Write ASS & SRT files
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

    # 2. Cut video
    clip_path = os.path.join(pkg_dir, "clip.mp4")
    success, msg = cut_clip(
        source_path, start_sec, end_sec, clip_path,
        ass_path=ass_path, vertical=vertical, crop_style=crop_style,
        src_width=src_width, src_height=src_height,
    )

    if not success:
        result["error"] = msg
        return result

    result["clip_path"] = clip_path
    result["success"]   = True

    # 3. Extract thumbnail
    from modules.thumbnail_generator import extract_thumbnail
    thumb_path = os.path.join(pkg_dir, "thumbnail.jpg")
    peak = metadata.get("peak_second") if metadata else None
    thumb = extract_thumbnail(clip_path, thumb_path, peak_second=peak)
    result["thumbnail_path"] = thumb

    # 4. Text files & metadata.json
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
