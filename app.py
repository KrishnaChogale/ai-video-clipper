"""
GTA 6 AI Auto Clipper — Main Streamlit Orchestrator
Modular architecture: each step delegates to a dedicated module in modules/.
All tuneable settings come from config.py.
Run: cd C:\\ai-video-clipper && .\\venv\\Scripts\\python.exe -m streamlit run app.py
"""
import os
import sys
import re
import json
import sqlite3
import shutil
import io
import subprocess
import streamlit as st

# ─────────────────────── PATH SETUP ───────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import config
from modules.gpu_utils import get_device_info, release_gpu_memory
from modules.video_ingestion import (
    download_youtube_video, capture_live_stream_segment, identify_live_platform,
    probe_video, accept_local_file,
)
from modules.audio_analysis import analyze_audio
from modules.speech_recognition import transcribe_audio
from modules.object_detection import analyze_visuals
from modules.event_classification import classify_events
from modules.highlight_scoring import compute_per_second_scores, select_highlights
from modules.title_generator import generate_titles, generate_hooks
from modules.description_generator import generate_descriptions, generate_hashtags
from modules.subtitle_generator import generate_clip_ass, generate_clip_srt, censor_profanity
from modules.caption_templates import (
    CAPTION_TEMPLATES, get_caption_template, load_custom_templates,
    save_custom_template, reset_custom_templates,
)
from modules.clip_editor import build_clip_package, zip_all_clips, FFMPEG_EXE, _use_nvenc
from modules.thumbnail_generator import extract_thumbnail
from modules.quality_control import evaluate_clip
from modules.metadata_manager import init_db, save_clip_metadata, get_all_clips, save_performance_metrics

# ─────────────────────── PAGE CONFIG ───────────────────────
st.set_page_config(
    page_title="GTA 6 AI Auto Clipper",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────── SESSION STATE DEFAULTS ───────────────────────
_DEFAULTS = {
    "video_path":    None,
    "transcript":    None,
    "words_list":    None,
    "energy_scores": None,
    "visual_scores": None,
    "highlights":    None,
    "final_clips":   [],
    "step_done":     0,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─────────────────────── SIDEBAR: DEVICE INFO & CONTROLS ───────────────────────
with st.sidebar:
    st.header("⚙️ System & Controls")

    with st.expander("🖥️ GPU & Hardware Status", expanded=True):
        try:
            dev_info = get_device_info()
            st.markdown(f"**GPU:** {dev_info.get('gpu_name', 'N/A')}")
            st.markdown(f"**VRAM:** {dev_info.get('vram_total_gb', 0):.1f} GB total / **{dev_info.get('vram_free_gb', 0):.1f} GB free**")
            st.markdown(f"**PyTorch CUDA:** {'✅ Active (v' + str(dev_info.get('torch_cuda_version')) + ')' if dev_info.get('cuda_available') else '❌ CPU Fallback'}")
            st.markdown(f"**NVENC Encoder:** {'✅ Available' if _use_nvenc() else '❌ CPU (libx264)'}")
            st.caption(f"FFmpeg: `{os.path.basename(FFMPEG_EXE)}`")
        except Exception as e:
            st.warning(f"Device probe note: {e}")

    st.markdown("---")
    st.subheader("🎬 Highlight Generation")
    clip_count = st.slider("Clips to generate", 1, 20, config.CLIP_DEFAULT_COUNT)

    duration_preset = st.radio(
        "Duration Mode",
        ["30–45s (Viral Shorts / Reels)", "Custom Duration"],
        index=0,
        horizontal=True,
    )
    if duration_preset == "30–45s (Viral Shorts / Reels)":
        clip_min_dur = config.CLIP_VIRAL_MIN_DURATION  # 30s
        clip_max_dur = config.CLIP_VIRAL_MAX_DURATION  # 45s
        clip_length = config.CLIP_DEFAULT_DURATION     # 35s
        st.caption("🎯 **Viral Sweet Spot:** 30–45s with opening 3-second hook optimization.")
    else:
        clip_length = st.slider(
            "Target clip length (sec)",
            min_value=config.CLIP_MIN_DURATION,
            max_value=config.CLIP_MAX_DURATION,
            value=35,
            step=5,
        )
        clip_min_dur = max(15, clip_length - 5)
        clip_max_dur = min(180, clip_length + 5)

    remove_dead_air = st.checkbox(
        "✂️ Remove Long Dead Air (limited jump-cuts)",
        value=config.REMOVE_DEAD_AIR_DEFAULT,
        key="remove_dead_air_enabled_v2",
        help="Off by default to preserve natural dialogue. When enabled, only the longest pauses are removed, with a limit of about two cuts per 30 seconds.",
    )

    st.markdown("---")
    st.subheader("🎞️ Framing & Dynamic Captions")
    aspect_mode = st.radio(
        "Video Framing",
        ["9:16 Vertical (Full Screen)", "9:16 Blurred Background", "Original Aspect Ratio"],
        index=0,
    )
    export_preset = st.selectbox(
        "Export Preset",
        ["YouTube Shorts (9:16)", "Instagram Reels (9:16)", "TikTok (9:16)", "YouTube Landscape (16:9)"],
    )
    export_codec = st.selectbox("Video codec", ["H.264", "H.265"], index=0,
                                help="H.265 encoding is attempted only when FFmpeg supports it; hardware encoding is probed at runtime.")
    export_fps = st.selectbox("Export FPS", [24, 30, 60], index=1)
    export_quality = st.slider("Video quality (lower is higher quality)", 16, 28, config.EXPORT_CRF)
    export_audio_bitrate = st.selectbox("Audio quality", ["128k", "192k", "256k", "320k"], index=1)
    export_vertical = (export_preset != "YouTube Landscape (16:9)"
                       and aspect_mode != "Original Aspect Ratio")
    export_settings = {
        "width": 1080 if export_vertical else 1920,
        "height": 1920 if export_vertical else 1080,
        "fps": export_fps,
        "codec": "h265" if export_codec == "H.265" else "h264",
        "quality": export_quality,
        "audio_bitrate": export_audio_bitrate,
        "preset": export_preset,
    }
    face_track_vertical = st.checkbox(
        "Face-aware vertical framing for streamer reactions",
        value=True,
        help="Tracks the largest visible face while framing portrait clips. Turn off to keep the fixed center crop.",
    )
    burn_captions = st.checkbox("Burn Captions (Viral Kinetic Subtitles)", value=True)
    saved_caption_templates = load_custom_templates()
    available_caption_templates = {**CAPTION_TEMPLATES, **saved_caption_templates}
    caption_template_name = st.selectbox(
        "Caption Template",
        list(available_caption_templates),
        index=list(available_caption_templates).index("Gaming Style"),
        disabled=not burn_captions,
        help="Switching templates updates this preview immediately and applies the style on export.",
    )
    selected_caption_template = available_caption_templates[caption_template_name]
    caption_font_size = st.slider("Caption font size", 24, 110,
                                  int(selected_caption_template["fontsize_vertical"]),
                                  disabled=not burn_captions,
                                  key=f"caption_size_{caption_template_name}")
    caption_font_family = st.selectbox(
        "Font family", ["Arial", "Arial Black", "Impact", "Verdana"],
        index=["Arial", "Arial Black", "Impact", "Verdana"].index(selected_caption_template["fontname"])
        if selected_caption_template["fontname"] in ["Arial", "Arial Black", "Impact", "Verdana"] else 0,
        disabled=not burn_captions,
        key=f"caption_font_{caption_template_name}",
    )
    caption_base_hex = st.color_picker(
        "Caption text color", selected_caption_template["base_color"], disabled=not burn_captions,
        key=f"caption_base_{caption_template_name}",
    )
    caption_highlight_hex = st.color_picker(
        "Caption highlight color", selected_caption_template["highlight_color"], disabled=not burn_captions,
        key=f"caption_highlight_{caption_template_name}",
    )
    caption_outline = st.slider("Text outline", 0.0, 8.0,
                                float(selected_caption_template["outline"]), 0.5,
                                disabled=not burn_captions,
                                key=f"caption_outline_{caption_template_name}")
    caption_shadow = st.slider("Text shadow", 0.0, 5.0,
                               float(selected_caption_template["shadow"]), 0.5,
                               disabled=not burn_captions,
                               key=f"caption_shadow_{caption_template_name}")
    caption_max_words = st.slider("Maximum words per caption line", 1, 10,
                                  int(selected_caption_template["max_words_per_cue"]),
                                  disabled=not burn_captions,
                                  key=f"caption_words_{caption_template_name}")
    dynamic_highlight = st.checkbox(
        "✨ Dynamic Word Highlighting (Reels/Shorts Pop)",
        value=selected_caption_template["dynamic_highlight"],
        disabled=not burn_captions,
        help="Highlights each spoken word in vibrant yellow/gold with white base text.",
        key=f"caption_dynamic_{caption_template_name}",
    )
    caption_position = st.selectbox(
        "Caption Placement",
        ["Bottom (Reels/Shorts Safe Zone)", "Center-Low (Podcast style)", "Top"],
        index=0,
        disabled=not burn_captions,
    )
    censor_swear = st.checkbox(
        "Censor Profanity (e.g. F*CK, SH*T)",
        value=True,
        disabled=not burn_captions,
    )
    burn_hook_header = st.checkbox(
        "🪝 Burn On-Screen Hook Title (First 3.5s)",
        value=True,
        help="Burns the opening viral hook as a bold title banner at the top of the video to stop viewers from scrolling.",
    )
    enable_caption_emojis = st.checkbox(
        "🔥 Dynamic Subtitle Emojis",
        value=True,
        disabled=not burn_captions,
        help="Automatically injects context emojis (🔥, 💀, 🚗, 🚨, 🎯) into subtitle phrases.",
    )
    caption_style_overrides = {
        "fontsize_vertical": caption_font_size,
        "fontsize_horizontal": max(18, round(caption_font_size * 0.7)),
        "fontname": caption_font_family,
        "base_color": caption_base_hex,
        "highlight_color": caption_highlight_hex,
        "outline_color": selected_caption_template.get("outline_color", "#111111"),
        "outline": caption_outline,
        "shadow": caption_shadow,
        "max_words_per_cue": caption_max_words,
        "dynamic_highlight": dynamic_highlight,
    }
    base_style_name = caption_template_name if caption_template_name in CAPTION_TEMPLATES else "Gaming Style"
    caption_style = get_caption_template(
        base_style_name,
        {**saved_caption_templates.get(caption_template_name, {}), **caption_style_overrides},
    )
    preview_text = "That was an incredible <span style='color:{}'>clutch</span>!".format(caption_highlight_hex)
    st.markdown(
        f"<div style='background:#17191f;border-radius:10px;padding:18px;text-align:center;"
        f"font-family:{caption_font_family};font-size:{min(32, caption_font_size//2)}px;"
        f"font-weight:{'800' if selected_caption_template['bold'] else '400'};color:{caption_base_hex};"
        f"-webkit-text-stroke:{caption_outline}px {selected_caption_template['outline_color']};"
        f"text-shadow:0 2px {caption_shadow}px #000'>"
        f"{preview_text}</div>",
        unsafe_allow_html=True,
    )
    save_col, reset_col = st.columns(2)
    with save_col:
        custom_template_name = st.text_input("Save current style as", placeholder="My Caption Style")
        if st.button("Save template", disabled=not custom_template_name.strip()):
            save_custom_template(custom_template_name, {
                **selected_caption_template,
                **caption_style_overrides,
            })
            st.rerun()
    with reset_col:
        st.caption("Custom templates are saved locally on this machine.")
        if st.button("Reset custom templates"):
            reset_custom_templates()
            st.rerun()

    st.markdown("---")
    st.subheader("🎯 Content Tuning")
    content_style = st.selectbox(
        "Target Gameplay Style",
        ["Balanced", "Action", "Funny", "Cinematic", "Discovery", "Scenic Showcase"],
        index=0,
    )
    whisper_model = st.selectbox(
        "Whisper Model Size",
        ["tiny", "base", "small", "medium"],
        index=1,
        help="Base is fast and accurate on 6GB VRAM. Tiny is fastest.",
    )

# ─────────────────────── MAIN TABS ───────────────────────
st.title("🎮 GTA 6 AI Auto Clipper")
st.caption("Intelligent Local Short-Form Content Generator · Opus Clip Style · NVIDIA RTX 3050 Accelerated")

tab_main, tab_history, tab_performance, tab_about = st.tabs([
    "🎬 Generate Clips",
    "📂 Clip History",
    "📊 Performance Tracking",
    "ℹ️ About & Diagnostics",
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — GENERATE CLIPS
# ══════════════════════════════════════════════════════════════
with tab_main:
    st.header("Step 1 · Input Video")

    # YouTube URL Input
    youtube_url = st.text_input("🔗 Paste YouTube URL:")

    default_cookie_file = os.path.join(BASE_DIR, "cookies.txt")
    uploaded_cookie_file = os.path.join(BASE_DIR, "uploaded_cookies.txt")
    has_local_cookie = os.path.isfile(default_cookie_file)
    has_uploaded_cookie = os.path.isfile(uploaded_cookie_file)

    with st.expander("🔑 YouTube Authentication & Cookies (For Age-Restricted / 18+ Videos)", expanded=False):
        st.write("Use this if YouTube requires login or age confirmation:")
        cookie_mode = st.radio(
            "Cookie Mode:",
            ["Auto-detect cookies.txt", "Upload cookies.txt", "Extract from Browser", "None"],
            index=0 if (has_local_cookie or has_uploaded_cookie) else 3,
            horizontal=True,
        )
        uploaded_cookie_obj = None
        selected_browser = None
        if cookie_mode == "Upload cookies.txt":
            uploaded_cookie_obj = st.file_uploader("Upload exported cookies.txt file", type=["txt"])
        elif cookie_mode == "Extract from Browser":
            selected_browser = st.selectbox(
                "Browser (must be closed during download):",
                ["chrome", "edge", "firefox", "brave"],
            )
        elif cookie_mode == "Auto-detect cookies.txt":
            if has_local_cookie:
                st.success(f"Found `cookies.txt` in project folder: `{default_cookie_file}`")
            elif has_uploaded_cookie:
                st.success(f"Using uploaded cookie file: `{uploaded_cookie_file}`")
            else:
                st.info("No `cookies.txt` detected yet. Upload one or place it in the project root.")

    with st.expander("🔴 Capture from an ongoing livestream", expanded=False):
        live_platform = st.selectbox(
            "Livestream platform", ["Twitch", "Kick", "YouTube Live"],
            key="live_stream_platform",
        )
        live_url = st.text_input(
            f"{live_platform} channel or livestream URL",
            key="live_stream_url",
            placeholder={"Twitch": "https://twitch.tv/channel", "Kick": "https://kick.com/channel",
                         "YouTube Live": "https://youtube.com/@channel/live"}[live_platform],
        )
        live_minutes = st.slider("Capture duration (minutes)", 1, 30, 5, key="live_capture_minutes")
        st.caption("Captures the next segment from a currently live channel, then sends it through the normal clipping pipeline. Scheduled streams and replays are not captured here.")
        detected_platform = identify_live_platform(live_url) if live_url else None
        if live_url and detected_platform is None:
            st.warning("Enter a Twitch, Kick, or YouTube Live URL.")
        elif live_url and detected_platform != live_platform:
            st.warning(f"This URL looks like {detected_platform}. Select that platform above.")
        if live_url and st.button("⏺️ Record live segment", key="capture_live_segment"):
            capture_cookies = uploaded_cookie_file if has_uploaded_cookie else default_cookie_file
            if cookie_mode == "Upload cookies.txt" and uploaded_cookie_obj is not None:
                with open(uploaded_cookie_file, "wb") as cookie_out:
                    cookie_out.write(uploaded_cookie_obj.getbuffer())
                capture_cookies = uploaded_cookie_file
            with st.spinner(f"Recording {live_minutes} minutes from the live stream…"):
                live_path, live_error = capture_live_stream_segment(
                    live_url, live_minutes * 60, cookies_path=capture_cookies,
                    expected_platform=live_platform,
                )
            if live_error:
                st.error(f"Live capture failed: {live_error}")
            else:
                st.session_state["video_path"] = live_path
                st.success(f"Captured live segment: `{os.path.basename(live_path)}`. You can now run the clipper.")

    if youtube_url and st.button("⬇️ Download Video", key="btn_yt_dl"):
        cookie_target = None
        if cookie_mode == "Auto-detect cookies.txt" and has_local_cookie:
            cookie_target = default_cookie_file
        elif cookie_mode == "Auto-detect cookies.txt" and has_uploaded_cookie:
            cookie_target = uploaded_cookie_file
        elif cookie_mode == "Upload cookies.txt" and uploaded_cookie_obj:
            with open(uploaded_cookie_file, "wb") as f:
                f.write(uploaded_cookie_obj.getbuffer())
            cookie_target = uploaded_cookie_file

        dl_bar = st.progress(0, "Connecting to YouTube...")

        def _on_dl_prog(pct, msg):
            dl_bar.progress(int(pct), msg)

        with st.spinner("Downloading video..."):
            if selected_browser:
                import yt_dlp
                try:
                    ydl_opts = {
                        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                        "outtmpl": os.path.join(config.INPUT_DIR, "%(id)s_%(title)s.%(ext)s"),
                        "merge_output_format": "mp4",
                        "quiet": True,
                        "cookiesfrombrowser": (selected_browser,),
                    }
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        inf = ydl.extract_info(youtube_url, download=True)
                        f_name = ydl.prepare_filename(inf)
                        for ext in (".mp4", ".mkv", ".webm"):
                            c_path = os.path.splitext(f_name)[0] + ext
                            if os.path.isfile(c_path):
                                st.session_state["video_path"] = c_path
                                break
                    dl_bar.progress(100, "Download complete!")
                    st.success("Download complete!")
                except Exception as e:
                    st.error(f"Browser download error: {e}")
            else:
                v_path, err = download_youtube_video(youtube_url, cookies_path=cookie_target, progress_callback=_on_dl_prog)
                if err:
                    err_l = err.lower()
                    if "age" in err_l or "sign in" in err_l:
                        st.error("🔒 **Age-Restricted Video:** Please supply cookies.txt or extract from browser above.")
                    else:
                        st.error(f"Download failed: {err}")
                else:
                    st.session_state["video_path"] = v_path
                    dl_bar.progress(100, "Complete!")
                    st.success(f"Video downloaded: `{os.path.basename(v_path)}`")

    st.markdown("---")

    # Local file upload (up to 50 GB chunked)
    st.markdown("**Or upload a local video file** (supports up to 50 GB via chunked streaming):")
    uploaded_file = st.file_uploader("Upload video (.mp4, .mkv, .mov, .avi):", type=["mp4", "mkv", "mov", "avi"])
    if uploaded_file is not None:
        dest_path = os.path.join(config.INPUT_DIR, uploaded_file.name)
        if not os.path.isfile(dest_path) or os.path.getsize(dest_path) != uploaded_file.size:
            with st.spinner(f"Saving uploaded video ({uploaded_file.size / (1024**2):.1f} MB)..."):
                with open(dest_path, "wb") as f:
                    while chunk := uploaded_file.read(16 * 1024 * 1024):
                        f.write(chunk)
        st.session_state["video_path"] = dest_path
        st.success(f"Loaded: `{uploaded_file.name}`")

    # Direct local path input
    local_path_input = st.text_input("Or enter local file path directly (instant for large files):")
    if local_path_input:
        clean_p = local_path_input.strip("\"' ")
        if os.path.isfile(clean_p):
            st.session_state["video_path"] = clean_p
            st.success(f"Using local video: `{clean_p}` ({os.path.getsize(clean_p) / (1024**2):.1f} MB)")
        else:
            st.warning(f"File not found: `{clean_p}`")

    video_path = st.session_state.get("video_path")

    # Video preview & probe info
    if video_path and os.path.isfile(video_path):
        st.video(video_path)
        try:
            v_meta = probe_video(video_path)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Duration", f"{v_meta['duration']:.0f}s ({v_meta['duration']/60:.1f}m)")
            c2.metric("Resolution", f"{v_meta['width']}×{v_meta['height']}")
            c3.metric("Framerate", f"{v_meta['fps']:.1f} fps")
            c4.metric("File Size", f"{v_meta['size_bytes'] / (1024**2):.1f} MB")
        except Exception:
            pass

    # ── PIPELINE TRIGGER ─────────────────────────────────────────
    st.markdown("---")
    st.header("▶️ Run AI Pipeline")

    btn_col1, btn_col2 = st.columns([2, 1])
    with btn_col1:
        if not video_path or not os.path.isfile(video_path):
            st.info("⬆️ Select or upload a video above to begin.")
            run_btn = False
        else:
            run_btn = st.button(
                f"✂️ Find, Cut & Caption {clip_count} Clips ({clip_length}s each)",
                type="primary",
                use_container_width=True,
            )
    with btn_col2:
        if st.button("🔄 Reset / Clear Session", use_container_width=True):
            for k in _DEFAULTS:
                st.session_state[k] = _DEFAULTS[k]
            st.rerun()

    if video_path and os.path.isfile(video_path) and run_btn:
        # STEP 2: Audio & Speech Analysis
        st.markdown("---")
        st.subheader("Step 2/5 · Audio & Speech Analysis")
        p2 = st.progress(0, "Loading audio...")

        try:
            p2.progress(15, "Analyzing audio energy, onset spikes, and transients...")
            energy_scores, audio_data, sr = analyze_audio(video_path)
            st.session_state["energy_scores"] = energy_scores

            p2.progress(45, f"Transcribing dialogue with Faster-Whisper ({whisper_model})...")
            transcript, words_list = transcribe_audio(audio_data, model_size=whisper_model)
            st.session_state["transcript"] = transcript
            st.session_state["words_list"] = words_list

            p2.progress(100, f"✅ Speech analysis complete ({len(transcript)} segments, {len(words_list)} words)")
            st.success(f"Transcribed {len(transcript)} dialogue segments with word-level timestamps.")
        except Exception as e:
            st.error(f"Step 2 failed: {e}")
            st.stop()
        finally:
            release_gpu_memory()

        # STEP 3: Visual Object Detection (YOLO nano)
        st.subheader("Step 3/5 · Visual Object Detection & Motion Analysis (YOLO)")
        p3 = st.progress(0, "Starting YOLO inference on GPU...")

        try:
            def _yolo_cb(pct, msg):
                p3.progress(int(pct), msg)

            visual_scores = analyze_visuals(video_path, progress_callback=_yolo_cb)
            st.session_state["visual_scores"] = visual_scores

            p3.progress(100, f"✅ Processed {len(visual_scores)} seconds of visual gameplay")
            st.success(f"YOLO detection & motion tracking complete across {len(visual_scores)} seconds.")
        except Exception as e:
            st.error(f"Step 3 failed: {e}")
            st.stop()
        finally:
            release_gpu_memory()

        # STEP 4: GTA Event Classification & AI Title/Metadata Generation
        st.subheader("Step 4/5 · GTA Event Classification, Highlight Scoring & AI Titles")
        p4 = st.progress(0, "Classifying GTA gameplay events...")

        try:
            events = classify_events(energy_scores, visual_scores, transcript)

            p4.progress(25, "Computing multi-factor highlight scores...")
            per_sec_scores = compute_per_second_scores(
                energy_scores, visual_scores, transcript, events
            )

            p4.progress(45, f"Selecting top {clip_count} highlights ({content_style} style, {clip_min_dur}–{clip_max_dur}s)...")
            highlights = select_highlights(
                per_sec_scores,
                event_classification=events,
                top_k=clip_count,
                min_dur=clip_min_dur,
                max_dur=clip_max_dur,
                style=content_style,
            )

            if not highlights:
                st.warning("No highlight windows passed the minimum score threshold. Try a longer video or lower duration.")
                st.stop()

            p4.progress(65, "Generating viral titles, hooks & platform metadata with Ollama...")
            for idx, h in enumerate(highlights):
                snippet = " ".join(
                    s["text"] for s in transcript
                    if s["end"] >= h["start"] and s["start"] <= h["end"]
                ).strip()
                h["transcript_snippet"] = snippet

                if h.get("category") == "scenic" and not snippet:
                    h["alt_titles"] = ["Take In This GTA View", "A Quiet Look at GTA's World", "GTA's World in Its Best Light"]
                    h["selected_title"] = h["alt_titles"][0]
                    h["alt_hooks"] = ["Take a second to look at this view", "A calmer moment in the GTA world", "Here is a scenic moment from the game"]
                    h["hook"] = h["alt_hooks"][0]
                    h["hashtags"] = ["#GTA", "#GTAGameplay", "#GamingClips", "#GameWorld"]
                    h.update({
                        "yt_description": "A scenic showcase moment from GTA gameplay.\n\n#GTA #GTAGameplay #GamingClips",
                        "ig_description": "A scenic moment from the GTA world 🎮\n\n#GTA #GTAGameplay #GamingClips",
                        "tt_description": "A quiet look at the GTA game world 🎮\n#GTA #GTAGameplay #GamingClips",
                    })
                elif h.get("category") == "unknown" and not snippet:
                    # The current local models cannot describe silent visuals.
                    # Keep metadata factual instead of inventing a chase/crash.
                    h["alt_titles"] = ["A GTA Moment Worth Replaying",
                                        "Watch This GTA Scene Unfold",
                                        "A Moment From This GTA Session"]
                    h["selected_title"] = h["alt_titles"][0]
                    h["alt_hooks"] = ["Watch this moment unfold",
                                      "Here is a moment from the session",
                                      "Take a closer look at this scene"]
                    h["hook"] = h["alt_hooks"][0]
                    h["hashtags"] = ["#GTA", "#GTAGameplay", "#GamingClips"]
                    h.update({
                        "yt_description": "A moment from this GTA session. What stood out to you?\n\n#GTA #GTAGameplay #GamingClips",
                        "ig_description": "A moment from this GTA session 🎮 Which detail caught your eye?\n\n#GTA #GTAGameplay #GamingClips",
                        "tt_description": "One moment from the session 🎮 What did you notice?\n#GTA #GTAGameplay #GamingClips",
                    })
                else:
                    h["alt_titles"] = generate_titles(h, style=content_style.lower())
                    h["selected_title"] = h["alt_titles"][0] if h["alt_titles"] else f"GTA Highlight {idx+1}"
                    h["alt_hooks"] = generate_hooks(h, style=content_style.lower())
                    h["hook"] = h["alt_hooks"][0] if h["alt_hooks"] else ""
                    h.update(generate_descriptions(h, title=h["selected_title"], style=content_style.lower()))
                    h["hashtags"] = generate_hashtags(h, snippet=snippet)

                p4.progress(65 + int(30 * (idx + 1) / len(highlights)), f"Processed metadata for Clip {idx+1}/{len(highlights)}")

            st.session_state["highlights"] = highlights
            p4.progress(100, f"✅ Scored and titled {len(highlights)} clips")
            st.success(f"Selected {len(highlights)} moments. Cutting clips and adding captions automatically.")
        except Exception as e:
            st.error(f"Step 4 failed: {e}")
            st.stop()

        # STEP 5: Clip Cutting, Burning Subtitles & Package Export
        st.subheader("Step 5/5 · Video Cutting, Subtitles & Export Packages")
        p5 = st.progress(0, "Initializing FFmpeg exporter...")

        v_probe = probe_video(video_path)
        src_w = v_probe.get("width", 0)
        src_h = v_probe.get("height", 0)

        is_vertical = export_vertical
        crop_style_choice = "blur_bg" if "Blurred Background" in aspect_mode else (
            "face" if is_vertical and face_track_vertical else "center"
        )

        final_clips = []
        for i, h in enumerate(highlights):
            clip_title_safe = re.sub(r"[^a-zA-Z0-9_-]", "_", h["selected_title"])[:30].strip("_")
            clip_id = f"clip_{i+1:02d}_{clip_title_safe}"

            p5.progress(
                int((i / len(highlights)) * 90),
                f"Exporting clip {i+1}/{len(highlights)}: {h['selected_title'][:35]}...",
            )

            # Subtitles & On-Screen Hook Banner
            ass_content = None
            srt_content = None
            hook_banner_text = (h.get("hook") or h.get("selected_title") or "").strip()
            if (burn_captions and transcript) or (burn_hook_header and hook_banner_text):
                ass_content = generate_clip_ass(
                    words_list=words_list or [],
                    transcript=transcript if burn_captions else None,
                    clip_start=h["start"],
                    clip_end=h["end"],
                    crop_vertical=is_vertical,
                    position=caption_position,
                    censor_swear_words=censor_swear,
                    dynamic_highlight=dynamic_highlight,
                    caption_style=caption_style,
                    hook_text=hook_banner_text,
                    burn_hook=burn_hook_header,
                    add_emojis=enable_caption_emojis,
                )
            if burn_captions and transcript:
                srt_content = generate_clip_srt(
                    transcript=transcript,
                    clip_start=h["start"],
                    clip_end=h["end"],
                    censor=censor_swear,
                )

            metadata_dict = {
                "clip_id":          clip_id,
                "source_path":      video_path,
                "start_sec":        h["start"],
                "end_sec":          h["end"],
                "duration":         h["end"] - h["start"],
                "category":         h.get("category", content_style.lower()),
                "score":            h.get("score", 0.0),
                "highlight_reason": h.get("reason", ""),
                "score_factors":    h.get("score_factors", {}),
                "peak_second":      h.get("peak_second"),
                "hook_score":       h.get("hook_score", 0.0),
                "hook_type":        h.get("hook_type", "action_onset"),
                "hook":             h.get("hook", ""),
                "alt_hooks":        h.get("alt_hooks", []),
                "selected_title":   h.get("selected_title", ""),
                "alt_titles":       h.get("alt_titles", []),
                "yt_description":   h.get("yt_description", ""),
                "ig_description":   h.get("ig_description", ""),
                "tt_description":   h.get("tt_description", ""),
                "hashtags":         h.get("hashtags", []),
                "words_list":       words_list or [],
                "transcript":       transcript or [],
                "caption_position": caption_position,
                "caption_template": caption_template_name,
                "caption_style":    caption_style,
                "burn_hook":        burn_hook_header,
                "add_emojis":       enable_caption_emojis,
                "export_settings":  export_settings,
                "crop_style":       crop_style_choice,
                "remove_dead_air":  remove_dead_air,
            }

            pkg = build_clip_package(
                clip_id=clip_id,
                source_path=video_path,
                start_sec=h["start"],
                end_sec=h["end"],
                ass_content=ass_content,
                srt_content=srt_content,
                vertical=is_vertical,
                crop_style=crop_style_choice,
                src_width=src_w,
                src_height=src_h,
                metadata=metadata_dict,
                remove_dead_air=remove_dead_air,
                export_settings=export_settings,
            )

            if not pkg["success"]:
                st.warning(f"Clip {i+1} export note: {pkg.get('error')}")
                continue

            # Quality Control Check
            qc = evaluate_clip(
                pkg["clip_path"],
                h["start"],
                h["end"],
                h.get("score", 0.0),
                min_score=5.0,
            )
            metadata_dict["qc_passed"] = qc["passed"]
            metadata_dict["qc_issues"] = qc["issues"]
            metadata_dict["clip_path"] = pkg["clip_path"]
            metadata_dict["thumbnail_path"] = pkg.get("thumbnail_path", "")

            # Save in SQLite DB
            save_clip_metadata(metadata_dict)

            final_clips.append({**h, **pkg, "metadata": metadata_dict, "qc": qc})

        st.session_state["final_clips"] = final_clips
        p5.progress(100, f"✅ Exported {len(final_clips)} ready-to-post clips!")
        st.success(f"🎉 All {len(final_clips)} clips generated successfully with captions, thumbnails, and metadata!")

    # ── RESULTS DISPLAY ──────────────────────────────────────────
    final_clips = st.session_state.get("final_clips", [])
    if final_clips:
        st.markdown("---")
        st.header(f"📦 Generated Clips ({len(final_clips)})")

        for idx, clip in enumerate(final_clips):
            meta = clip.get("metadata", {})
            qc = clip.get("qc", {})
            passed = qc.get("passed", True)
            score = meta.get("score", 0.0)
            category = meta.get("category", "gaming").upper()

            with st.expander(
                f"{'✅' if passed else '⚠️'} #{idx+1} · [{category}] {meta.get('selected_title', 'Clip')} "
                f"(Score: {score:.1f} | {meta.get('start_sec', 0):.0f}s–{meta.get('end_sec', 0):.0f}s)",
                expanded=(idx == 0),
            ):
                col_preview, col_details = st.columns([1, 1])

                with col_preview:
                    cp = clip.get("clip_path")
                    if cp and os.path.isfile(cp):
                        st.video(cp)
                    th = clip.get("thumbnail_path")
                    if th and os.path.isfile(th):
                        st.image(th, caption="Suggested Cover Thumbnail", use_container_width=True)

                with col_details:
                    # Retention & Hook Metrics
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Virality Score", f"{score:.1f}")
                    m2.metric("3s Hook Strength", f"{meta.get('hook_score', 0):.0f}%")
                    silence_rem = meta.get("silence_removed_sec", 0.0)
                    m3.metric("Dead Air Cut", f"{silence_rem:.1f}s" if silence_rem > 0 else "Tight")
                    if meta.get("highlight_reason"):
                        st.caption(meta["highlight_reason"])
                    if meta.get("score_factors"):
                        with st.expander("Why this moment was selected"):
                            st.json(meta["score_factors"])

                    with st.expander("Transcript and clip trim"):
                        duration_info = probe_video(meta.get("source_path", "")) if meta.get("source_path") else {}
                        source_duration = max(float(duration_info.get("duration", meta.get("end_sec", 0))), 1.0)
                        clip_start_edit = st.number_input(
                            "Clip start (seconds)", min_value=0.0, max_value=source_duration,
                            value=float(meta.get("start_sec", 0)), step=0.1, key=f"clip_start_{idx}",
                        )
                        clip_end_edit = st.number_input(
                            "Clip end (seconds)", min_value=0.1, max_value=source_duration,
                            value=float(meta.get("end_sec", 1)), step=0.1, key=f"clip_end_{idx}",
                        )
                        transcript_excerpt = [
                            {"Start": round(float(seg.get("start", 0)), 2),
                             "End": round(float(seg.get("end", 0)), 2),
                             "Transcript": seg.get("text", "")}
                            for seg in meta.get("transcript", [])
                            if float(seg.get("end", 0)) > clip_start_edit
                            and float(seg.get("start", 0)) < clip_end_edit
                        ]
                        if transcript_excerpt:
                            st.dataframe(transcript_excerpt, hide_index=True, use_container_width=True)
                        else:
                            st.caption("No recognized speech falls inside this time range.")
                        if st.button("Apply trim and re-render", key=f"apply_trim_{idx}"):
                            if clip_end_edit <= clip_start_edit:
                                st.error("Clip end must be later than clip start.")
                            else:
                                edited_meta = dict(meta)
                                edited_meta.update({
                                    "start_sec": float(clip_start_edit),
                                    "end_sec": float(clip_end_edit),
                                    "duration": float(clip_end_edit - clip_start_edit),
                                    "clip_id": f"{meta.get('clip_id', f'clip_{idx+1:02d}')}_edit",
                                })
                                edited_hook = edited_meta.get("hook") or edited_meta.get("selected_title")
                                edited_burn_hook = edited_meta.get("burn_hook", True)
                                edited_emojis = edited_meta.get("add_emojis", True)
                                edited_ass = generate_clip_ass(
                                    words_list=edited_meta.get("words_list", []),
                                    transcript=edited_meta.get("transcript", []),
                                    clip_start=clip_start_edit, clip_end=clip_end_edit,
                                    crop_vertical=bool((edited_meta.get("export_settings") or {}).get("height", 1920)
                                                       > (edited_meta.get("export_settings") or {}).get("width", 1080)),
                                    position=edited_meta.get("caption_position", "Bottom (Shorts/TikTok)"),
                                    caption_style=edited_meta.get("caption_style"),
                                    hook_text=edited_hook,
                                    burn_hook=edited_burn_hook,
                                    add_emojis=edited_emojis,
                                ) if (edited_meta.get("words_list") or edited_meta.get("transcript") or (edited_burn_hook and edited_hook)) else None
                                edited_srt = generate_clip_srt(
                                    edited_meta.get("transcript", []), clip_start_edit, clip_end_edit
                                ) if edited_meta.get("transcript") else None
                                export_opts = edited_meta.get("export_settings") or {}
                                edited_pkg = build_clip_package(
                                    clip_id=edited_meta["clip_id"],
                                    source_path=edited_meta["source_path"],
                                    start_sec=clip_start_edit, end_sec=clip_end_edit,
                                    ass_content=edited_ass, srt_content=edited_srt,
                                    vertical=bool(export_opts.get("height", 1920) > export_opts.get("width", 1080)),
                                    crop_style=edited_meta.get("crop_style", "center"),
                                    src_width=int(duration_info.get("width", 0)),
                                    src_height=int(duration_info.get("height", 0)),
                                    metadata=edited_meta,
                                    remove_dead_air=bool(edited_meta.get("remove_dead_air", False)),
                                    export_settings=export_opts,
                                )
                                if edited_pkg.get("success"):
                                    edited_qc = evaluate_clip(
                                        edited_pkg["clip_path"], clip_start_edit, clip_end_edit,
                                        edited_meta.get("score", 0), min_score=5.0,
                                    )
                                    edited_meta.update({"clip_path": edited_pkg["clip_path"],
                                                        "thumbnail_path": edited_pkg.get("thumbnail_path", ""),
                                                        "qc_passed": edited_qc["passed"], "qc_issues": edited_qc["issues"]})
                                    save_clip_metadata(edited_meta)
                                    updated_clip = {**clip, **edited_pkg, "metadata": edited_meta, "qc": edited_qc}
                                    current_clips = list(st.session_state.get("final_clips", []))
                                    if idx < len(current_clips):
                                        current_clips[idx] = updated_clip
                                        st.session_state["final_clips"] = current_clips
                                    st.success("Updated trim rendered and checked.")
                                    st.rerun()
                                else:
                                    st.error(f"Could not re-render trim: {edited_pkg.get('error')}")

                    if passed:
                        st.success("✅ Quality Check: Passed all retention & technical checks")
                    else:
                        st.warning("⚠️ QC Notes: " + ", ".join(qc.get("issues", [])))

                    # 3 Title alternatives
                    alt_titles = meta.get("alt_titles", [meta.get("selected_title", "")])
                    selected_title = st.selectbox(
                        "🏷️ Selected Title (3 Alternatives)",
                        alt_titles,
                        key=f"title_sel_{idx}",
                    )

                    # 3 Hook alternatives
                    alt_hooks = meta.get("alt_hooks", [meta.get("hook", "")])
                    if alt_hooks:
                        st.selectbox("🪝 Opening Hook Text (3 Alternatives)", alt_hooks, key=f"hook_sel_{idx}")

                    # Descriptions per platform
                    desc_tabs = st.tabs(["📺 YouTube Shorts", "📸 Instagram Reels", "🎵 TikTok"])
                    with desc_tabs[0]:
                        st.text_area("YouTube Description", meta.get("yt_description", ""), height=110, key=f"desc_yt_{idx}")
                    with desc_tabs[1]:
                        st.text_area("Instagram Description", meta.get("ig_description", ""), height=110, key=f"desc_ig_{idx}")
                    with desc_tabs[2]:
                        st.text_area("TikTok Description", meta.get("tt_description", ""), height=110, key=f"desc_tt_{idx}")

                    # Hashtags
                    tags = meta.get("hashtags", [])
                    if tags:
                        tag_str = " ".join(f"#{t.lstrip('#')}" for t in tags)
                        st.markdown("**Hashtags:**")
                        st.code(tag_str, language="text")

                    # Download Buttons
                    dl_c1, dl_c2 = st.columns(2)
                    with dl_c1:
                        if cp and os.path.isfile(cp):
                            with open(cp, "rb") as vf:
                                st.download_button(
                                    "⬇️ Download MP4",
                                    vf,
                                    file_name=os.path.basename(cp),
                                    mime="video/mp4",
                                    key=f"dl_vid_{idx}",
                                    use_container_width=True,
                                )
                    with dl_c2:
                        srt_p = clip.get("srt_path")
                        if srt_p and os.path.isfile(srt_p):
                            with open(srt_p, "rb") as sf:
                                st.download_button(
                                    "📄 Download SRT",
                                    sf,
                                    file_name=os.path.basename(srt_p),
                                    key=f"dl_srt_{idx}",
                                    use_container_width=True,
                                )

        # Batch Export All
        st.markdown("---")
        if st.button("📦 Export All Clips as ZIP Package", type="secondary", use_container_width=True):
            with st.spinner("Bundling all clips, thumbnails, subtitles, and metadata into ZIP..."):
                zip_file = zip_all_clips()
                if zip_file and os.path.isfile(zip_file):
                    with open(zip_file, "rb") as zf:
                        st.download_button(
                            "⬇️ Download Full ZIP Archive (MP4 + Thumbnails + Metadata + SRT)",
                            zf,
                            file_name="gta6_clips_bundle.zip",
                            mime="application/zip",
                            use_container_width=True,
                        )
                else:
                    st.error("Failed to package ZIP archive.")

# ══════════════════════════════════════════════════════════════
# TAB 2 — CLIP HISTORY
# ══════════════════════════════════════════════════════════════
with tab_history:
    st.header("📂 Saved Clips Database")
    try:
        all_stored = get_all_clips()
        if not all_stored:
            st.info("No clips recorded in the database yet. Generate clips in the first tab!")
        else:
            st.markdown(f"Found **{len(all_stored)}** clip records in `{config.DB_PATH}`:")
            for item in all_stored:
                with st.expander(
                    f"[{item.get('category', 'clip').upper()}] {item.get('selected_title', 'Clip')} "
                    f"· Score: {item.get('score', 0):.1f} · {item.get('created_at', '')[:10]}"
                ):
                    c_col1, c_col2 = st.columns([1, 1])
                    with c_col1:
                        c_path = item.get("clip_path")
                        if c_path and os.path.isfile(c_path):
                            st.video(c_path)
                        else:
                            st.warning(f"File moved or unavailable: `{c_path}`")
                    with c_col2:
                        st.write(f"**Duration:** {item.get('duration', 0):.1f}s ({item.get('start_sec', 0):.0f}s to {item.get('end_sec', 0):.0f}s)")
                        st.write(f"**Hook:** {item.get('hook', 'N/A')}")
                        st.write(f"**YouTube Description:**")
                        st.text(item.get("yt_description", ""))
                        st.json({
                            "clip_id": item.get("clip_id"),
                            "source": item.get("source_path"),
                            "qc_passed": bool(item.get("qc_passed")),
                            "score": item.get("score"),
                        })
    except Exception as e:
        st.error(f"Error accessing database: {e}")

# ══════════════════════════════════════════════════════════════
# TAB 3 — PERFORMANCE TRACKING
# ══════════════════════════════════════════════════════════════
with tab_performance:
    st.header("📊 Social Media Content Performance")
    st.caption("Track real-world retention, views, and likes across platforms to learn which GTA 6 moments perform best.")

    all_stored = get_all_clips()
    if not all_stored:
        st.info("Generate clips first to start logging performance stats.")
    else:
        clip_options = {c["clip_id"]: f"[{c.get('category','clip').upper()}] {c.get('selected_title','')}" for c in all_stored}
        selected_clip_id = st.selectbox(
            "Select Clip to Log or Review Performance:",
            options=list(clip_options.keys()),
            format_func=lambda x: clip_options[x],
        )

        perf_col1, perf_col2, perf_col3 = st.columns(3)
        with perf_col1:
            p_platform = st.selectbox("Platform", ["YouTube Shorts", "Instagram Reels", "TikTok"])
            p_views = st.number_input("Views", min_value=0, value=0, step=100)
            p_likes = st.number_input("Likes", min_value=0, value=0, step=10)
        with perf_col2:
            p_shares = st.number_input("Shares / Reposts", min_value=0, value=0, step=5)
            p_comments = st.number_input("Comments", min_value=0, value=0, step=5)
            p_subs = st.number_input("Followers / Subscribers Gained", min_value=0, value=0, step=1)
        with perf_col3:
            p_watch_sec = st.number_input("Average Watch Time (seconds)", min_value=0.0, value=0.0, step=1.0)
            p_completion = st.slider("Completion Rate (%)", min_value=0, max_value=100, value=50)

        if st.button("💾 Save Performance Record", type="primary"):
            save_performance_metrics(
                selected_clip_id,
                p_platform,
                {
                    "views": p_views,
                    "likes": p_likes,
                    "shares": p_shares,
                    "comments": p_comments,
                    "avg_watch_sec": p_watch_sec,
                    "completion_rate": p_completion,
                    "subs_gained": p_subs,
                },
            )
            st.success(f"Saved stats for `{selected_clip_id}` on {p_platform}!")

# ══════════════════════════════════════════════════════════════
# TAB 4 — ABOUT & DIAGNOSTICS
# ══════════════════════════════════════════════════════════════
with tab_about:
    st.header("ℹ️ GTA 6 AI Auto Clipper — Architecture & Guide")
    st.markdown("""
### 🏗️ Complete Modular Architecture
The application runs 100% locally on your machine with zero paid API dependencies:

- **Hardware Acceleration:** Native PyTorch with CUDA 12.1 for NVIDIA RTX 3050 Laptop GPU (6GB VRAM) + NVENC hardware video encoding.
- **Audio Processing (`modules/audio_analysis.py`):** Librosa 16kHz mono audio energy and onset transient detection to pinpoint gunshots, explosions, and sirens.
- **Speech Recognition (`modules/speech_recognition.py`):** Faster-Whisper with word-level timestamps, automated CUDA-to-CPU fallback, and profanity masking.
- **Visual Detection (`modules/object_detection.py`):** Ultralytics YOLO nano model + OpenCV motion difference to spot high-speed chases, collisions, and NPC crowds.
- **Event Classification (`modules/event_classification.py`):** Heuristic detector for Police Chases, Crashes, Explosions, Stunts, Cinematic views, and NPC reactions.
- **Highlight Scoring (`modules/highlight_scoring.py`):** 6-factor multi-modal scoring (0–100 scale) with non-overlapping dynamic window selection.
- **Metadata Generator (`modules/title_generator.py` & `description_generator.py`):** Local Ollama (`llama3.2`) generating 3 titles, 3 hooks, platform descriptions, and hashtags.
- **Video Subtitling (`modules/subtitle_generator.py`):** ASS subtitles formatted in viral lower-third safe zones with profanity masking.
- **Video Clipper (`modules/clip_editor.py`):** FFmpeg NVENC/libx264 cutting with 9:16 vertical center crop or blurred background.
- **Thumbnail Extractor (`modules/thumbnail_generator.py`):** Laplacian variance and brightness filters to extract crisp, non-blurry cover frames.
- **Quality Control (`modules/quality_control.py`):** Automated checklist verification before exporting.

### 🚀 Running the App
Always run via your virtual environment python module:
```powershell
cd C:\\ai-video-clipper
.\\venv\\Scripts\\python.exe -m streamlit run app.py
```
""")
