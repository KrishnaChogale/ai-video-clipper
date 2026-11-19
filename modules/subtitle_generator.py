"""
modules/subtitle_generator.py
Profanity censoring, ASS subtitle generation (lower-third placement), and SRT export.
Supports word-level dynamic animated cues and segment fallbacks.
"""
import sys, os
import re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ─────────────────────── PROFANITY MAP ───────────────────────
PROFANITY_MAP = {
    r"\bmotherfucking\b": "MOTHERF*CKING",
    r"\bmotherfuckers\b": "MOTHERF*CKERS",
    r"\bmotherfucker\b":  "MOTHERF*CKER",
    r"\bfucking\b":       "F*CKING",
    r"\bfucked\b":        "F*CKED",
    r"\bfucker\b":        "F*CKER",
    r"\bfuckers\b":       "F*CKERS",
    r"\bfucks\b":         "F*CKS",
    r"\bfuck\b":          "F*CK",
    r"\bbullshit\b":      "BULLSH*T",
    r"\bshitty\b":        "SH*TTY",
    r"\bshitting\b":      "SH*TTING",
    r"\bshits\b":         "SH*TS",
    r"\bshit\b":          "SH*T",
    r"\bbitches\b":       "B*TCHES",
    r"\bbitching\b":      "B*TCHING",
    r"\bbitch\b":         "B*TCH",
    r"\bassholes\b":      "A**HOLES",
    r"\basshole\b":       "A**HOLE",
    r"\bdumbass\b":       "DUMBA**",
    r"\bjackass\b":       "JACKA**",
    r"\bbadass\b":        "BADA**",
    r"\bass\b":           "A**",
    r"\bdickhead\b":      "D*CKHEAD",
    r"\bdicks\b":         "D*CKS",
    r"\bdick\b":          "D*CK",
    r"\bpussies\b":       "P*SSIES",
    r"\bpussy\b":         "P*SSY",
    r"\bcunts\b":         "C*NTS",
    r"\bcunt\b":          "C*NT",
    r"\bbastards\b":      "B*STARDS",
    r"\bbastard\b":       "B*STARD",
    r"\bgoddamn\b":       "GODD*MN",
    r"\bdammit\b":        "D*MMIT",
    r"\bdamned\b":        "D*MNED",
    r"\bdamn\b":          "D*MN",
    r"\bcrap\b":          "CR*P",
    r"\bcocksucker\b":    "C*CKSUCKER",
    r"\bcocks\b":         "C*CKS",
    r"\bcock\b":          "C*CK",
    r"\bwhores\b":        "WH*RES",
    r"\bwhore\b":         "WH*RE",
    r"\bsluts\b":         "SL*TS",
    r"\bslut\b":          "SL*T",
    r"\bniggers\b":       "N*GGERS",
    r"\bnigger\b":        "N*GGER",
    r"\bniggas\b":        "N*GGAS",
    r"\bnigga\b":         "N*GGA",
}

# ─────────────────────── DYNAMIC SUBTITLE EMOJIS ───────────────────────
EMOJI_KEYWORD_MAP = {
    r"\b(crazy|insane|wild|unreal|fire|lit|heat|hype|cracked|clutch)\b": "🔥",
    r"\b(dead|die|died|killing|killed|kill|rip|ded|rekt|busted|wasted)\b": "💀",
    r"\b(car|cars|drive|driving|speed|drift|drifting|race|racing|vehicle|whip)\b": "🚗",
    r"\b(cop|cops|police|feds|siren|sirens|wanted|arrested|escape|escaping)\b": "🚨",
    r"\b(shoot|shot|gun|guns|weapon|sniper|bullet|bullets|aim)\b": "🎯",
    r"\b(boom|explode|explosion|bomb|blast|crash|crashed|smash|hit)\b": "💥",
    r"\b(money|cash|rich|bank|heist|dollar|dollars|pay|paid|rich|bag)\b": "💰",
    r"\b(lol|lmao|haha|hahaha|funny|joke|laughing|laugh)\b": "😂",
    r"\b(omg|wow|what|bro|dude|wait|shocked|look)\b": "😱",
    r"\b(win|won|goat|king|god|legend)\b": "👑",
}


def detect_cue_emoji(text: str) -> str:
    """Detect matching context emoji for subtitle phrases based on gaming/action keywords."""
    for pattern, emoji in EMOJI_KEYWORD_MAP.items():
        if re.search(pattern, text, re.IGNORECASE):
            return emoji
    return ""


def wrap_hook_text(text: str, max_chars_per_line: int = 24) -> str:
    """Wrap hook text onto 2-3 lines using ASS \\N tag."""
    words = text.strip().split()
    if not words:
        return ""
    lines = []
    current_line = []
    current_len = 0
    for w in words:
        if current_len + len(w) + 1 > max_chars_per_line and current_line:
            lines.append(" ".join(current_line))
            current_line = [w]
            current_len = len(w)
        else:
            current_line.append(w)
            current_len += len(w) + 1
    if current_line:
        lines.append(" ".join(current_line))
    return r"\N".join(lines)


def format_hook_ass_text(text: str) -> str:
    """Format hook title text with yellow highlight on lead words and white on remainder."""
    clean = text.strip().upper().replace("{", "(").replace("}", ")")
    words = clean.split()
    if len(words) <= 2:
        return r"{\c&H0000FFFF&}" + clean
    lead = " ".join(words[:2])
    rest = " ".join(words[2:])
    wrapped_rest = wrap_hook_text(rest, max_chars_per_line=24)
    return r"{\c&H0000FFFF&}" + lead + r" {\c&H00FFFFFF&}" + wrapped_rest


def censor_profanity(text: str) -> str:
    """Replace profanity using PROFANITY_MAP (case-insensitive regex)."""
    for pattern, rep in PROFANITY_MAP.items():
        text = re.sub(pattern, rep, text, flags=re.IGNORECASE)
    return text


def format_ass_time(seconds: float) -> str:
    """Convert float seconds to ASS timestamp H:MM:SS.cc"""
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = min(99, int(round((seconds - int(seconds)) * 100)))
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def format_srt_time(seconds: float) -> str:
    """Convert float seconds to SRT timestamp HH:MM:SS,mmm"""
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = min(999, int(round((seconds - int(seconds)) * 1000)))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def remap_words_for_dead_air(
    words_list: list,
    cut_intervals: list[tuple[float, float]],
    clip_start: float,
    clip_end: float,
) -> list[dict]:
    """
    Remaps word timestamps when dead air / pauses are trimmed from the video.
    Subtitles remain frame-accurately synchronized with the jump-cut video.
    """
    if not words_list:
        return []

    remapped = []
    for w in words_list:
        if w.get("end", 0) <= clip_start or w.get("start", 0) >= clip_end:
            continue

        orig_s = max(0.0, w["start"] - clip_start)
        orig_e = min(clip_end - clip_start, w["end"] - clip_start)
        if orig_e <= orig_s or not w.get("word"):
            continue

        # Calculate time removed before this word
        removed_before_s = 0.0
        removed_before_e = 0.0
        word_dropped = False

        for (c_start, c_end) in cut_intervals:
            # If word is completely inside a cut interval, drop it
            if orig_s >= c_start and orig_e <= c_end:
                word_dropped = True
                break
            # Time removed before start
            if orig_s >= c_end:
                removed_before_s += (c_end - c_start)
            elif orig_s > c_start:
                removed_before_s += (orig_s - c_start)

            # Time removed before end
            if orig_e >= c_end:
                removed_before_e += (c_end - c_start)
            elif orig_e > c_start:
                removed_before_e += (orig_e - c_start)

        if not word_dropped:
            new_s = max(0.0, orig_s - removed_before_s)
            new_e = max(new_s + 0.05, orig_e - removed_before_e)
            remapped.append({
                "start": round(new_s, 3),
                "end":   round(new_e, 3),
                "word":  w["word"],
            })

    return remapped


def generate_clip_ass(
    words_list: list = None,
    transcript: list = None,
    clip_start: float = 0.0,
    clip_end: float = 30.0,
    crop_vertical: bool = True,
    position: str = "Bottom (Shorts/TikTok)",
    censor_swear_words: bool = True,
    max_words_per_cue: int = None,
    dynamic_highlight: bool = True,
    highlight_color: str = None,
    caption_style: dict | None = None,
    hook_text: str | None = None,
    hook_duration: float = None,
    burn_hook: bool = None,
    add_emojis: bool = None,
    hook_style: str = None,
    facecam_split: bool = False,
) -> str:
    """
    Generate pixel-perfect ASS subtitles for viral shorts.
    Supports dynamic word-by-word kinetic highlighting (Opus Clip / CapCut style),
    context-aware emoji triggers, top hook title banners,
    and safe zones optimized for Instagram Reels, YouTube Shorts, and TikTok.
    """
    style = caption_style or {}
    max_words_per_cue = max_words_per_cue or style.get("max_words_per_cue") or config.CAPTION_MAX_WORDS_PER_CUE
    dynamic_highlight = bool(style.get("dynamic_highlight", dynamic_highlight))
    highlight_color = highlight_color or style.get("highlight_color") or config.CAPTION_COLOR_HIGHLIGHT
    base_color = style.get("base_color") or config.CAPTION_COLOR_BASE
    burn_hook = getattr(config, "HOOK_BANNER_ENABLED", True) if burn_hook is None else bool(burn_hook)
    add_emojis = getattr(config, "CAPTION_EMOJIS_ENABLED", True) if add_emojis is None else bool(add_emojis)
    hook_duration = float(hook_duration or getattr(config, "HOOK_BANNER_DURATION_SEC", 3.5))
    hook_style = str(hook_style or getattr(config, "HOOK_BANNER_STYLE", "pill")).lower()
    clip_words = []

    if words_list:
        for w in words_list:
            # If start is already relative (e.g. from remap_words_for_dead_air)
            if "start" in w and "end" in w:
                if w.get("end", 0) > clip_start and w.get("start", 0) < clip_end:
                    rel_start = max(0.0, w["start"] - clip_start)
                    rel_end   = min(clip_end - clip_start, w["end"] - clip_start)
                else:
                    rel_start = w["start"]
                    rel_end   = w["end"]

                if rel_end > rel_start and w.get("word"):
                    clip_words.append({"start": rel_start, "end": rel_end, "word": w["word"]})

    cues = []
    if clip_words:
        current_chunk = []
        for w in clip_words:
            current_chunk.append(w)
            ends_sentence = any(w["word"].endswith(p) for p in [".", "!", "?", ","])
            if len(current_chunk) >= max_words_per_cue or ends_sentence:
                cues.append(current_chunk)
                current_chunk = []
        if current_chunk:
            cues.append(current_chunk)
    elif transcript:
        for seg in transcript:
            if seg.get("end", 0) > clip_start and seg.get("start", 0) < clip_end:
                rel_start = max(0.0, seg["start"] - clip_start)
                rel_end   = min(clip_end - clip_start, seg["end"] - clip_start)
                words = seg.get("text", "").split()
                if not words:
                    continue
                dur = max(0.5, rel_end - rel_start)
                wps = len(words) / dur
                for i in range(0, len(words), max_words_per_cue):
                    chunk = words[i:i + max_words_per_cue]
                    c_start = rel_start + (i / wps)
                    c_end   = min(rel_end, rel_start + ((i + len(chunk)) / wps))
                    cues.append([{"start": c_start, "end": c_end, "word": " ".join(chunk)}])

    if not cues and not (burn_hook and hook_text):
        return ""

    # Canvas & safe margin placement (optimized for Reels & Shorts UI overlays)
    pos_lower = (position or "").lower()
    if crop_vertical:
        res_x, res_y = config.EXPORT_WIDTH, config.EXPORT_HEIGHT  # 1080, 1920
        fontsize = style.get("fontsize_vertical", config.CAPTION_FONT_SIZE_VERTICAL)
        outline  = style.get("outline", config.CAPTION_OUTLINE)
        shadow   = style.get("shadow", config.CAPTION_SHADOW)
        margin_l = config.CAPTION_SAFE_MARGIN_H_REELS             # 80
        margin_r = config.CAPTION_SAFE_MARGIN_H_REELS + 20        # 100 (space for right UI buttons)

        if "center" in pos_lower:
            alignment = 2
            margin_v = 800   # Middle / Action zone
        elif "top" in pos_lower:
            alignment = 8
            margin_v = 240   # Header safe zone
        else:  # Bottom
            alignment = 2
            if facecam_split:
                # Safe zone for split screen: upper area of gameplay panel, below the banner
                # Leaves streamer face (top) and game HUD (bottom) completely clear
                margin_v = 780
            else:
                margin_v = config.CAPTION_SAFE_MARGIN_V_REELS  # 440px (Safe above Reels/Shorts bottom bar)

        hook_fontsize = getattr(config, "HOOK_BANNER_FONT_SIZE_VERTICAL", 48)
        hook_margin_v = getattr(config, "HOOK_BANNER_MARGIN_V_VERTICAL", 260)
    else:
        res_x, res_y = 1920, 1080
        fontsize = style.get("fontsize_horizontal", config.CAPTION_FONT_SIZE_HORIZONTAL)
        outline = style.get("outline", 3.5)
        shadow = style.get("shadow", 2.0)
        margin_l = 40
        margin_r = 40
        if "center" in pos_lower:
            alignment = 2
            margin_v = 220
        elif "top" in pos_lower:
            alignment = 8
            margin_v = 90
        else:
            alignment = 2
            margin_v = 100

        hook_fontsize = getattr(config, "HOOK_BANNER_FONT_SIZE_HORIZONTAL", 36)
        hook_margin_v = getattr(config, "HOOK_BANNER_MARGIN_V_HORIZONTAL", 70)

    fontname = str(style.get("fontname", "Arial"))
    if not re.fullmatch(r"[A-Za-z0-9 _-]{1,40}", fontname):
        fontname = "Arial"
    bold = -1 if style.get("bold", True) else 0
    outline_color = style.get("outline_color", config.CAPTION_COLOR_OUTLINE)
    back_color = style.get("back_color", "&H80000000&")
    border_style = int(style.get("border_style", 1))

    # Hook banner style definition (Pill box vs Bold Outline)
    if hook_style == "outline":
        hook_style_def = f"Style: HookBanner,Arial Black,{hook_fontsize},&H00FFFFFF,&H000000FF,&H00000000,&H90000000,-1,0,0,0,100,100,2,0,1,6,3,8,60,60,{hook_margin_v},1"
    else:
        # Default "pill": Translucent rounded box
        hook_style_def = f"Style: HookBanner,Arial Black,{hook_fontsize},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,1,0,3,14,0,8,60,60,{hook_margin_v},1"

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {res_x}
PlayResY: {res_y}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: ViralShorts,{fontname},{fontsize},{base_color},&H000000FF,{outline_color},{back_color},{bold},0,0,0,100,100,0,0,{border_style},{outline},{shadow},{alignment},{margin_l},{margin_r},{margin_v},1
{hook_style_def}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []

    # 1. On-Screen Hook Title Banner Dialogue (Layer 1, displayed for first hook_duration seconds)
    if burn_hook and hook_text:
        hook_end = min(max(0.5, clip_end - clip_start), hook_duration)
        if hook_end > 0.4:
            styled_hook = format_hook_ass_text(hook_text)
            events.append(
                f"Dialogue: 1,0:00:00.00,{format_ass_time(hook_end)},HookBanner,,0,0,0,,{{\\fad(100,250)}}{styled_hook}"
            )

    # 2. Dynamic Kinetic Subtitles
    for cue in cues:
        # Check if we should inject context emoji for this cue
        cue_full_text = " ".join(item["word"] for item in cue)
        detected_emoji = detect_cue_emoji(cue_full_text) if add_emojis else ""
        has_existing_emoji = any(char in cue_full_text for char in ["🔥", "💀", "🚗", "🚨", "🎯", "💥", "💰", "😂", "😱", "👑"])
        cue_emoji = detected_emoji if (detected_emoji and not has_existing_emoji) else ""

        # Check if we have individual word timing in this cue
        has_word_timing = dynamic_highlight and len(cue) > 1 and all("start" in w and "end" in w for w in cue)

        if has_word_timing:
            # Word-by-word kinetic highlighting (Karaoke style)
            clean_words = []
            for item in cue:
                w_str = item["word"].strip().upper()
                if censor_swear_words:
                    w_str = censor_profanity(w_str)
                w_str = w_str.replace("{", "(").replace("}", ")")
                clean_words.append(w_str)

            # Append emoji to the final word of the cue
            if cue_emoji and clean_words:
                clean_words[-1] = f"{clean_words[-1]} {cue_emoji}"

            for j, active_item in enumerate(cue):
                w_start = active_item["start"]
                # Display each word until next word begins, or natural duration
                if j < len(cue) - 1:
                    w_end = max(w_start + 0.1, cue[j + 1]["start"])
                else:
                    w_end = max(active_item["end"], w_start + 0.35)

                # Format current line: active word kinetic pop & highlight, others in base white
                line_parts = []
                for k, w_text in enumerate(clean_words):
                    if k == j:
                        is_high_arousal = any(em in w_text for em in ["!", "?", "INSANE", "CRAZY", "CLUTCH", "DEAD", "WHAT", "GOD", "HYPE", "OMG", "BRO"])
                        scale_tag = r"\fscx116\fscy116" if is_high_arousal else r"\fscx112\fscy112"
                        active_color = "&H0037E6FF&" if (is_high_arousal and highlight_color == "&H0000FFFF&") else highlight_color
                        line_parts.append(f"{{{scale_tag}\\c{active_color}}}{w_text}{{\\fscx100\\fscy100}}{{\\c{base_color}}}")
                    else:
                        line_parts.append(w_text)

                line_str = " ".join(line_parts)
                events.append(
                    f"Dialogue: 0,{format_ass_time(w_start)},{format_ass_time(w_end)},ViralShorts,,0,0,0,,{line_str}"
                )
        else:
            # Fallback chunk display
            cue_start = cue[0]["start"]
            cue_end   = max(cue[-1]["end"], cue_start + 0.4)
            text = " ".join(item["word"] for item in cue).strip().upper()
            if censor_swear_words:
                text = censor_profanity(text)
            text = text.replace("{", "(").replace("}", ")")
            if cue_emoji:
                text = f"{text} {cue_emoji}"
            # Apply highlight color to entire phrase if specified
            styled_text = f"{{\\c{highlight_color}}}{text}" if highlight_color else text
            events.append(
                f"Dialogue: 0,{format_ass_time(cue_start)},{format_ass_time(cue_end)},ViralShorts,,0,0,0,,{styled_text}"
            )

    return header + "\n".join(events) + "\n"


def generate_clip_srt(
    transcript: list,
    clip_start: float,
    clip_end: float,
    censor: bool = True,
    max_words_per_cue: int = None,
) -> str:
    """Generate SRT subtitle content for a clip."""
    max_words_per_cue = max_words_per_cue or config.CAPTION_MAX_WORDS_PER_CUE
    lines = []
    idx = 1

    for seg in (transcript or []):
        s_start = seg.get("start", 0.0)
        s_end   = seg.get("end", s_start + 1.0)
        if s_end <= clip_start or s_start >= clip_end:
            continue

        rel_start = max(0.0, s_start - clip_start)
        rel_end   = min(clip_end - clip_start, s_end - clip_start)

        text = seg.get("text", "").strip()
        if not text:
            continue
        if censor:
            text = censor_profanity(text)

        words = text.split()
        dur = max(0.5, rel_end - rel_start)
        wps = len(words) / dur

        for i in range(0, len(words), max_words_per_cue):
            chunk = " ".join(words[i:i + max_words_per_cue])
            c_start = rel_start + (i / wps)
            c_end   = min(rel_end, rel_start + ((i + len(chunk.split())) / wps))

            lines.append(str(idx))
            lines.append(f"{format_srt_time(c_start)} --> {format_srt_time(c_end)}")
            lines.append(chunk)
            lines.append("")
            idx += 1

    return "\n".join(lines)
