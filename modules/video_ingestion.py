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
from urllib.parse import urlparse
from datetime import datetime


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


# ─────────────────────── FORMAT HELPERS ───────────────────────

def format_bytes(bytes_num: float | int | None) -> str:
    """Format bytes into human-readable string (KB, MB, GB)."""
    if bytes_num is None or bytes_num < 0:
        return "Unknown size"
    bytes_float = float(bytes_num)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_float < 1024.0 or unit == "TB":
            return f"{bytes_float:.1f} {unit}" if unit != "B" else f"{int(bytes_float)} B"
        bytes_float /= 1024.0
    return f"{bytes_float:.1f} GB"


def format_seconds(seconds: float | int | None) -> str:
    """Format seconds into HH:MM:SS or MM:SS."""
    if seconds is None or seconds < 0:
        return "--:--"
    total_sec = int(seconds)
    m, s = divmod(total_sec, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _notify_progress(callback, pct: float | None, message: str, details: dict | None = None):
    """Safely deliver progress to callback supporting (pct, msg, details) or (pct, msg)."""
    if callback is None:
        return
    try:
        callback(pct, message, details or {})
    except TypeError:
        try:
            callback(pct, message)
        except Exception:
            pass
    except Exception:
        pass


# ─────────────────────── YOUTUBE DOWNLOAD ───────────────────────

def download_youtube_video(
    url: str,
    output_dir: str | None = None,
    cookies_path: str | None = None,
    browser: str | None = None,
    progress_callback=None,
) -> tuple[str | None, str | None]:
    """
    Download a video from YouTube or any yt-dlp supported platform.
    Returns (video_path, error_message).
    progress_callback(pct, message, details) is called throughout download and merging.
    """
    output_dir = output_dir or config.INPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    try:
        import yt_dlp
        import time

        last_update_ts = [0.0]

        def _progress_hook(d):
            if progress_callback is None:
                return
            status = d.get("status")
            if status == "downloading":
                now = time.time()
                # Rate limit updates so Streamlit UI stays smooth
                if now - last_update_ts[0] < 0.12:
                    return
                last_update_ts[0] = now

                downloaded = d.get("downloaded_bytes", 0)
                total = d.get("total_bytes") or d.get("total_bytes_estimate")
                speed = d.get("speed")
                eta = d.get("eta")
                frag_idx = d.get("fragment_index")
                frag_cnt = d.get("fragment_count")

                info = d.get("info_dict") or {}
                vcodec = info.get("vcodec")
                acodec = info.get("acodec")
                if vcodec and vcodec != "none" and (not acodec or acodec == "none"):
                    stream_label = "Video Track"
                elif acodec and acodec != "none" and (not vcodec or vcodec == "none"):
                    stream_label = "Audio Track"
                else:
                    stream_label = "Media"

                pct = 0.0
                if total and total > 0:
                    pct = min(100.0, max(0.0, (downloaded / total) * 100.0))
                elif frag_cnt and frag_cnt > 0 and frag_idx:
                    pct = min(100.0, max(0.0, (frag_idx / frag_cnt) * 100.0))
                else:
                    pct = None

                downloaded_str = format_bytes(downloaded)
                total_str = format_bytes(total) if total else ("Unknown size" if not frag_cnt else f"{frag_cnt} frags")
                speed_str = f"{format_bytes(speed)}/s" if speed else "-- KB/s"
                eta_str = format_seconds(eta) if eta is not None else "--:--"

                if total and total > 0:
                    msg = f"📥 Downloading {stream_label}: {downloaded_str} / {total_str} ({pct:.0f}%) • {speed_str} • ETA: {eta_str}"
                elif frag_cnt:
                    msg = f"📥 Downloading {stream_label}: fragment {frag_idx}/{frag_cnt} • {downloaded_str} • {speed_str}"
                else:
                    msg = f"📥 Downloading {stream_label}: {downloaded_str} • {speed_str}"

                details = {
                    "pct": pct if pct is not None else 50.0,
                    "status": "downloading",
                    "stream_label": stream_label,
                    "downloaded_bytes": downloaded,
                    "total_bytes": total,
                    "downloaded_str": downloaded_str,
                    "total_str": total_str,
                    "speed": speed,
                    "speed_str": speed_str,
                    "eta": eta,
                    "eta_str": eta_str,
                }
                _notify_progress(progress_callback, pct if pct is not None else 50.0, msg, details)

            elif status == "finished":
                total = d.get("total_bytes") or d.get("downloaded_bytes", 0)
                details = {
                    "pct": 100.0,
                    "status": "finished",
                    "stream_label": "Media",
                    "downloaded_bytes": total,
                    "total_bytes": total,
                    "downloaded_str": format_bytes(total),
                    "total_str": format_bytes(total),
                    "speed": None,
                    "speed_str": "--",
                    "eta": 0,
                    "eta_str": "00:00",
                }
                msg = f"📥 Download finished ({format_bytes(total)}). Preparing media..."
                _notify_progress(progress_callback, 100.0, msg, details)

        def _postprocessor_hook(d):
            if progress_callback is None:
                return
            status = d.get("status")
            if status == "started":
                pp_name = d.get("postprocessor", "")
                if "Merger" in pp_name or "ffmpeg" in pp_name.lower():
                    msg = "⚙️ Merging video & audio tracks into MP4 with FFmpeg..."
                else:
                    msg = f"⚙️ Finalizing media ({pp_name})..."
                details = {
                    "pct": 98.0,
                    "status": "processing",
                    "stage": msg,
                }
                _notify_progress(progress_callback, 98.0, msg, details)
            elif status == "finished":
                details = {
                    "pct": 100.0,
                    "status": "completed",
                    "stage": "Media processing complete!",
                }
                _notify_progress(progress_callback, 100.0, "✅ Media processing complete!", details)

        ydl_opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": os.path.join(output_dir, "%(id)s_%(title)s.%(ext)s"),
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True,
            "progress_hooks": [_progress_hook],
            "postprocessor_hooks": [_postprocessor_hook],
            "socket_timeout": 30,
        }

        # Cookie auth or browser extraction
        if browser:
            ydl_opts["cookiesfrombrowser"] = (browser,)
        elif cookies_path and os.path.isfile(cookies_path):
            ydl_opts["cookiefile"] = cookies_path
        else:
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


def identify_live_platform(url: str) -> str | None:
    """Return the supported live platform for a Twitch, Kick, or YouTube URL."""
    try:
        parsed = urlparse((url or "").strip())
        host = (parsed.hostname or "").lower().rstrip(".")
    except Exception:
        return None
    if parsed.scheme not in {"http", "https"}:
        return None
    if host == "twitch.tv" or host.endswith(".twitch.tv"):
        return "Twitch"
    if host == "kick.com" or host.endswith(".kick.com"):
        return "Kick"
    if host in {"youtube.com", "youtu.be", "youtube-nocookie.com"} or host.endswith(".youtube.com"):
        return "YouTube Live"
    return None


def capture_live_stream_segment(
    url: str,
    duration_sec: int,
    output_dir: str | None = None,
    cookies_path: str | None = None,
    expected_platform: str | None = None,
    progress_callback=None,
) -> tuple[str | None, str | None]:
    """Capture a bounded live segment from Twitch, Kick, or YouTube with real-time progress."""
    output_dir = output_dir or config.INPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    platform = identify_live_platform(url)
    if platform is None:
        return None, "Use a Twitch, Kick, or YouTube Live channel/stream URL."
    if expected_platform and platform != expected_platform:
        return None, f"That link is from {platform}; choose {platform} in the platform selector."
    try:
        import yt_dlp
        import time
        from modules.clip_editor import FFMPEG_EXE

        duration_sec = max(30, min(int(duration_sec), 3600))
        opts = {"quiet": True, "no_warnings": True, "skip_download": True,
                "socket_timeout": 30, "live_from_start": False}
        if cookies_path and os.path.isfile(cookies_path):
            opts["cookiefile"] = cookies_path
        elif os.path.isfile(os.path.join(config.BASE_DIR, "uploaded_cookies.txt")):
            opts["cookiefile"] = os.path.join(config.BASE_DIR, "uploaded_cookies.txt")

        _notify_progress(progress_callback, 0.0, f"🔍 Connecting to {platform} livestream...")

        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
        live_status = info.get("live_status")
        is_live = info.get("is_live") is True or live_status == "is_live"
        if not is_live:
            if live_status == "is_upcoming":
                return None, "This stream is scheduled but not live yet. Try again after it starts."
            return None, f"No active {platform} livestream was found at this URL. Check that the channel is live."
        formats = [f for f in info.get("formats", []) if f.get("url") and f.get("vcodec") != "none"]
        if not formats and info.get("url"):
            formats = [info]
        if not formats:
            return None, "Could not find a playable live video stream for this URL."
        video = max(formats, key=lambda f: (f.get("acodec") != "none", f.get("height") or 0, f.get("tbr") or 0))
        stream_url = video.get("url")
        headers = video.get("http_headers") or info.get("http_headers") or {}
        headers_arg = "".join(f"{key}: {value}\r\n" for key, value in headers.items())
        stream_id = re.sub(r"[^A-Za-z0-9_-]", "_", str(info.get("id") or "live"))[:48]
        capture_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"{stream_id}_{capture_stamp}_live_segment.mkv")

        cmd = [
            FFMPEG_EXE, "-y", "-nostdin", "-loglevel", "error",
            "-rw_timeout", "15000000",
        ]
        if headers_arg:
            cmd += ["-headers", headers_arg]
        cmd += [
            "-i", stream_url,
            "-t", str(duration_sec),
            "-map", "0:v:0",
            "-map", "0:a:0?",
            "-c", "copy",
            "-f", "matroska",
            "-progress", "pipe:1",
            "-nostats",
            output_path,
        ]

        _notify_progress(progress_callback, 1.0, f"🔴 Starting capture of {format_seconds(duration_sec)} segment...")

        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        current_sec = 0.0
        total_size = 0
        speed = "1.0x"
        last_update_ts = 0.0

        if p.stdout:
            for line in p.stdout:
                line = line.strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    if k == "out_time_us":
                        try:
                            current_sec = float(v) / 1_000_000.0
                        except Exception:
                            pass
                    elif k == "total_size":
                        try:
                            total_size = int(v) if v.isdigit() else total_size
                        except Exception:
                            pass
                    elif k == "speed":
                        speed = v.strip()
                    elif k == "progress":
                        now = time.time()
                        if now - last_update_ts >= 0.25 or v == "end":
                            last_update_ts = now
                            pct = min(100.0, max(0.0, (current_sec / duration_sec) * 100.0))
                            if total_size <= 0 and os.path.isfile(output_path):
                                total_size = os.path.getsize(output_path)
                            details = {
                                "pct": pct,
                                "elapsed_sec": current_sec,
                                "target_sec": duration_sec,
                                "elapsed_str": format_seconds(current_sec),
                                "target_str": format_seconds(duration_sec),
                                "size_bytes": total_size,
                                "size_str": format_bytes(total_size),
                                "speed": speed,
                                "status": "recording" if v != "end" else "finished",
                            }
                            msg = f"🔴 Recording stream: {details['elapsed_str']} / {details['target_str']} ({pct:.0f}%) • {details['size_str']} captured • {speed}"
                            _notify_progress(progress_callback, pct, msg, details)

        p.wait(timeout=duration_sec + 90)
        stderr_output = p.stderr.read() if p.stderr else ""

        if p.returncode != 0 or not os.path.isfile(output_path) or os.path.getsize(output_path) < 100_000:
            return None, (stderr_output or "The live capture ended before enough video arrived.")[-1200:]

        final_size = os.path.getsize(output_path)
        _notify_progress(progress_callback, 100.0, f"✅ Stream segment captured: {format_bytes(final_size)}", {
            "pct": 100.0,
            "status": "completed",
            "size_str": format_bytes(final_size),
        })

        return output_path, None
    except subprocess.TimeoutExpired:
        if 'p' in locals():
            p.kill()
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
