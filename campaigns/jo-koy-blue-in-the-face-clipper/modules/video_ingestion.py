"""
modules/video_ingestion.py
YouTube download (cookie auth) + local file handling + ffprobe video metadata.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import subprocess
import json
import shutil
import re


# ─────────────────────── FFPROBE ───────────────────────

def probe_video(video_path: str) -> dict:
    """
    Use ffprobe to get video metadata.
    Returns dict with: duration, fps, width, height, codec, audio_codec, size_bytes.
    Falls back to OpenCV if ffprobe unavailable.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_streams", "-show_format", video_path,
            ],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            fmt = data.get("format", {})
            streams = data.get("streams", [])
            video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
            audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), {})

            fps_str = video_stream.get("r_frame_rate", "30/1")
            try:
                num, den = fps_str.split("/")
                fps = round(int(num) / int(den), 3)
            except Exception:
                fps = 30.0

            return {
                "duration":    float(fmt.get("duration", 0)),
                "fps":         fps,
                "width":       int(video_stream.get("width", 0)),
                "height":      int(video_stream.get("height", 0)),
                "codec":       video_stream.get("codec_name", "unknown"),
                "audio_codec": audio_stream.get("codec_name", "unknown"),
                "size_bytes":  int(fmt.get("size", 0)),
            }
    except Exception:
        pass

    # OpenCV fallback
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            return {
                "duration":    frames / fps,
                "fps":         fps,
                "width":       w,
                "height":      h,
                "codec":       "unknown",
                "audio_codec": "unknown",
                "size_bytes":  os.path.getsize(video_path) if os.path.isfile(video_path) else 0,
            }
    except Exception:
        pass

    return {"duration": 0, "fps": 30.0, "width": 0, "height": 0,
            "codec": "unknown", "audio_codec": "unknown", "size_bytes": 0}


# ─────────────────────── FORMAT REMUX ───────────────────────

def remux_to_mp4(src_path: str, dest_path: str) -> str:
    """
    If src_path is MKV/MOV/AVI/WEBM, remux (stream copy) to MP4.
    Returns dest_path on success, src_path if already MP4 or on error.
    """
    ext = os.path.splitext(src_path)[1].lower()
    if ext in (".mp4",):
        return src_path

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src_path, "-c", "copy", dest_path],
            check=True, capture_output=True, timeout=300,
        )
        return dest_path
    except Exception as e:
        print(f"[video_ingestion] remux_to_mp4 failed: {e}")
        return src_path


# ─────────────────────── YOUTUBE DOWNLOAD ───────────────────────

def download_youtube_video(
    url: str,
    output_dir: str | None = None,
    cookies_path: str | None = None,
    progress_callback=None,
) -> tuple[str | None, str | None]:
    """
    Download a YouTube video using yt-dlp.
    Returns (video_path, error_message).
    progress_callback(pct: float, status: str) is called if provided.
    """
    output_dir = output_dir or config.INPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    try:
        import yt_dlp

        def _progress_hook(d):
            if progress_callback is None:
                return
            if d.get("status") == "downloading":
                total = d.get("total_bytes") or d.get("total_bytes_estimate") or 1
                downloaded = d.get("downloaded_bytes", 0)
                pct = min(downloaded / total * 100, 100)
                progress_callback(pct, f"Downloading… {pct:.0f}%")
            elif d.get("status") == "finished":
                progress_callback(100, "Download complete")

        ydl_opts = {
            "format":           "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl":          os.path.join(output_dir, "%(id)s_%(title)s.%(ext)s"),
            "merge_output_format": "mp4",
            "quiet":            True,
            "no_warnings":      True,
            "progress_hooks":   [_progress_hook],
            "socket_timeout":   30,
        }

        # Cookie auth
        if cookies_path and os.path.isfile(cookies_path):
            ydl_opts["cookiefile"] = cookies_path
        else:
            # Try default uploaded_cookies.txt location
            default_cookies = os.path.join(config.BASE_DIR, "uploaded_cookies.txt")
            if os.path.isfile(default_cookies):
                ydl_opts["cookiefile"] = default_cookies

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            # Find the actual file on disk
            filename = ydl.prepare_filename(info)
            # yt-dlp may change extension after merging
            for ext in (".mp4", ".mkv", ".webm", ".mov"):
                candidate = os.path.splitext(filename)[0] + ext
                if os.path.isfile(candidate):
                    # Remux to MP4 if needed
                    if ext != ".mp4":
                        mp4_path = os.path.splitext(candidate)[0] + ".mp4"
                        candidate = remux_to_mp4(candidate, mp4_path)
                    return candidate, None

            # Fallback: scan output dir for newest file
            files = sorted(
                [f for f in os.listdir(output_dir) if f.endswith(".mp4")],
                key=lambda f: os.path.getmtime(os.path.join(output_dir, f)),
                reverse=True,
            )
            if files:
                return os.path.join(output_dir, files[0]), None

            return None, "Downloaded file not found"

    except Exception as e:
        return None, str(e)


def capture_live_stream_segment(url: str, duration_sec: int, output_dir: str | None = None,
                                cookies_path: str | None = None) -> tuple[str | None, str | None]:
    """Capture a bounded segment from a live yt-dlp source, then process it as a video."""
    output_dir = output_dir or config.INPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    try:
        import yt_dlp
        from modules.clip_editor import FFMPEG_EXE

        duration_sec = max(30, min(int(duration_sec), 3600))
        opts = {"quiet": True, "no_warnings": True, "skip_download": True,
                "socket_timeout": 30, "live_from_start": False}
        if cookies_path and os.path.isfile(cookies_path):
            opts["cookiefile"] = cookies_path
        elif os.path.isfile(os.path.join(config.BASE_DIR, "uploaded_cookies.txt")):
            opts["cookiefile"] = os.path.join(config.BASE_DIR, "uploaded_cookies.txt")

        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
        if info.get("is_live") is False and info.get("live_status") not in {"is_live", "is_upcoming"}:
            return None, "This URL does not appear to be a live stream. Use the regular video download instead."
        formats = [f for f in info.get("formats", []) if f.get("url") and f.get("vcodec") != "none"]
        if not formats and info.get("url"):
            formats = [info]
        if not formats:
            return None, "Could not find a playable live video stream for this URL."
        video = max(formats, key=lambda f: (f.get("acodec") != "none", f.get("height") or 0, f.get("tbr") or 0))
        headers = video.get("http_headers") or info.get("http_headers") or {}
        headers_arg = "".join(f"{key}: {value}\r\n" for key, value in headers.items())
        stream_id = re.sub(r"[^A-Za-z0-9_-]", "_", str(info.get("id") or "live"))[:48]
        output_path = os.path.join(output_dir, f"{stream_id}_live_segment.mkv")
        cmd = [FFMPEG_EXE, "-y", "-nostdin", "-loglevel", "error", "-rw_timeout", "15000000"]
        if headers_arg:
            cmd += ["-headers", headers_arg]
        cmd += ["-i", video.get("url") or info["url"], "-t", str(duration_sec),
                "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy", "-f", "matroska", output_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration_sec + 90)
        if result.returncode != 0 or not os.path.isfile(output_path) or os.path.getsize(output_path) < 100_000:
            return None, (result.stderr or "The live capture ended before enough video arrived.")[-1200:]
        return output_path, None
    except subprocess.TimeoutExpired:
        return None, "Live capture timed out. Try a shorter segment or check the stream connection."
    except Exception as e:
        return None, str(e)


def accept_local_file(src_path: str, dest_dir: str | None = None) -> tuple[str | None, str | None]:
    """
    Accept a local video file: copy it to dest_dir (INPUT_DIR) if not already there.
    Returns (dest_path, error_message).
    """
    dest_dir = dest_dir or config.INPUT_DIR
    os.makedirs(dest_dir, exist_ok=True)

    if not os.path.isfile(src_path):
        return None, f"File not found: {src_path}"

    filename = os.path.basename(src_path)
    dest_path = os.path.join(dest_dir, filename)

    ext = os.path.splitext(filename)[1].lower()

    if os.path.abspath(src_path) == os.path.abspath(dest_path):
        # Already in dest_dir; just remux if needed
        if ext != ".mp4":
            mp4_path = os.path.splitext(dest_path)[0] + ".mp4"
            dest_path = remux_to_mp4(dest_path, mp4_path)
        return dest_path, None

    # Copy to input dir
    try:
        shutil.copy2(src_path, dest_path)
    except Exception as e:
        return None, f"Copy failed: {e}"

    if ext != ".mp4":
        mp4_path = os.path.splitext(dest_path)[0] + ".mp4"
        dest_path = remux_to_mp4(dest_path, mp4_path)

    return dest_path, None
