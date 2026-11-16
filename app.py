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
from modules.video_ingestion import download_youtube_video, probe_video, accept_local_file
from modules.audio_analysis import analyze_audio
from modules.speech_recognition import transcribe_audio
from modules.object_detection import analyze_visuals
from modules.event_classification import classify_events
from modules.highlight_scoring import compute_per_second_scores, select_highlights
from modules.title_generator import generate_titles, generate_hooks
from modules.description_generator import generate_descriptions, generate_hashtags
from modules.subtitle_generator import generate_clip_ass, generate_clip_srt, censor_profanity
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
    clip_min = st.slider("Min duration (sec)", 10, 60, config.CLIP_MIN_DURATION, step=5)
    clip_max = st.slider("Max duration (sec)", 15, 120, config.CLIP_MAX_DURATION, step=5)
    if clip_min > clip_max:
        clip_min, clip_max = clip_max, clip_min

    st.markdown("---")
    st.subheader("🎞️ Aspect Ratio & Subtitles")
    aspect_mode = st.radio(
        "Video Framing",
        ["9:16 Vertical (Full Screen)", "9:16 Blurred Background", "Original Aspect Ratio"],
        index=0,
    )
    burn_captions = st.checkbox("Burn Captions (Viral Animated Subtitles)", value=True)
    caption_position = st.selectbox(
        "Caption Placement",
        ["Bottom (Shorts/TikTok)", "Center-Low (Podcast style)", "Top"],
        index=0,
        disabled=not burn_captions,
    )
    censor_swear = st.checkbox(
        "Censor Profanity (e.g. F*CK, SH*T)",
        value=True,
        disabled=not burn_captions,
    )

    st.markdown("---")
    st.subheader("🎯 Content Tuning")
    content_style = st.selectbox(
        "Target Gameplay Style",
        ["Balanced", "Action", "Funny", "Cinematic", "Discovery"],
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
                f"🚀 Generate {clip_count} Viral Highlights ({clip_min}–{clip_max}s each)",
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

            p4.progress(45, f"Selecting top {clip_count} highlights ({content_style} style)...")
            highlights = select_highlights(
                per_sec_scores,
                event_classification=events,
                top_k=clip_count,
                min_dur=clip_min,
                max_dur=clip_max,
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
                ).strip() or "(High-action GTA gameplay moment)"
                h["transcript_snippet"] = snippet

                # Titles & Hooks
                titles = generate_titles(h, style=content_style.lower())
                h["alt_titles"] = titles
                h["selected_title"] = titles[0] if titles else f"GTA 6 Highlight {idx+1}"

                hooks = generate_hooks(h, style=content_style.lower())
                h["alt_hooks"] = hooks
                h["hook"] = hooks[0] if hooks else ""

                # Platform Descriptions & Hashtags
                descs = generate_descriptions(h, title=h["selected_title"], style=content_style.lower())
                h.update(descs)
                h["hashtags"] = generate_hashtags(h, snippet=snippet)

                p4.progress(65 + int(30 * (idx + 1) / len(highlights)), f"Processed metadata for Clip {idx+1}/{len(highlights)}")

            st.session_state["highlights"] = highlights
            p4.progress(100, f"✅ Scored and titled {len(highlights)} clips")
            st.success(f"Selected {len(highlights)} winning highlight moments.")
        except Exception as e:
            st.error(f"Step 4 failed: {e}")
            st.stop()

        # STEP 5: Clip Cutting, Burning Subtitles & Package Export
        st.subheader("Step 5/5 · Video Cutting, Subtitles & Export Packages")
        p5 = st.progress(0, "Initializing FFmpeg exporter...")

        v_probe = probe_video(video_path)
        src_w = v_probe.get("width", 0)
        src_h = v_probe.get("height", 0)

        is_vertical = (aspect_mode != "Original Aspect Ratio")
        crop_style_choice = "blur_bg" if "Blurred Background" in aspect_mode else "center"

        final_clips = []
        for i, h in enumerate(highlights):
            clip_title_safe = re.sub(r"[^a-zA-Z0-9_-]", "_", h["selected_title"])[:30].strip("_")
            clip_id = f"clip_{i+1:02d}_{clip_title_safe}"

            p5.progress(
                int((i / len(highlights)) * 90),
                f"Exporting clip {i+1}/{len(highlights)}: {h['selected_title'][:35]}...",
            )

            # Subtitles
            ass_content = None
            srt_content = None
            if burn_captions and transcript:
                ass_content = generate_clip_ass(
                    words_list=words_list or [],
                    transcript=transcript,
                    clip_start=h["start"],
                    clip_end=h["end"],
                    crop_vertical=is_vertical,
                    position=caption_position,
                    censor_swear_words=censor_swear,
                )
                srt_content = generate_clip_srt(
                    transcript=transcript,
                    clip_start=h["start"],
                    clip_end=h["end"],
                    censor=censor_swear,
                )

            metadata_dict = {
                "clip_id":        clip_id,
                "source_path":    video_path,
                "start_sec":      h["start"],
                "end_sec":        h["end"],
                "duration":       h["end"] - h["start"],
                "category":       h.get("category", content_style.lower()),
                "score":          h.get("score", 0.0),
                "peak_second":    h.get("peak_second"),
                "hook":           h.get("hook", ""),
                "alt_hooks":      h.get("alt_hooks", []),
                "selected_title": h.get("selected_title", ""),
                "alt_titles":     h.get("alt_titles", []),
                "yt_description": h.get("yt_description", ""),
                "ig_description": h.get("ig_description", ""),
                "tt_description": h.get("tt_description", ""),
                "hashtags":       h.get("hashtags", []),
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
