"""Caption presets and color conversion helpers for ASS subtitle rendering."""

import json
import os

import config

CAPTION_TEMPLATES = {
    "MrBeast Style": {
        "fontname": "Arial", "fontsize_vertical": 72, "fontsize_horizontal": 50,
        "base_color": "#FFFFFF", "highlight_color": "#FFE600", "outline_color": "#101010",
        "bold": True, "outline": 6.0, "shadow": 2.0, "max_words_per_cue": 3,
        "dynamic_highlight": True,
    },
    "Hormozi Style": {
        "fontname": "Arial", "fontsize_vertical": 66, "fontsize_horizontal": 46,
        "base_color": "#FFFFFF", "highlight_color": "#FFD24A", "outline_color": "#080808",
        "bold": True, "outline": 5.0, "shadow": 1.0, "max_words_per_cue": 4,
        "dynamic_highlight": True,
    },
    "Podcast Style": {
        "fontname": "Arial", "fontsize_vertical": 52, "fontsize_horizontal": 38,
        "base_color": "#FFFFFF", "highlight_color": "#58D6C7", "outline_color": "#171717",
        "bold": False, "outline": 3.0, "shadow": 1.0, "max_words_per_cue": 7,
        "dynamic_highlight": False,
    },
    "Gaming Style": {
        "fontname": "Arial", "fontsize_vertical": 68, "fontsize_horizontal": 48,
        "base_color": "#FFFFFF", "highlight_color": "#72FF4D", "outline_color": "#080808",
        "bold": True, "outline": 5.5, "shadow": 2.0, "max_words_per_cue": 3,
        "dynamic_highlight": True,
    },
    "Minimal Style": {
        "fontname": "Arial", "fontsize_vertical": 54, "fontsize_horizontal": 40,
        "base_color": "#FFFFFF", "highlight_color": "#FFFFFF", "outline_color": "#111111",
        "bold": False, "outline": 2.5, "shadow": 0.5, "max_words_per_cue": 6,
        "dynamic_highlight": False,
    },
    "Karaoke Style": {
        "fontname": "Arial", "fontsize_vertical": 64, "fontsize_horizontal": 46,
        "base_color": "#FFFFFF", "highlight_color": "#37E6FF", "outline_color": "#080808",
        "bold": True, "outline": 5.0, "shadow": 1.5, "max_words_per_cue": 4,
        "dynamic_highlight": True,
    },
}


def hex_to_ass(value: str, alpha: str = "00") -> str:
    """Convert #RRGGBB to ASS's &HAABBGGRR color format."""
    raw = str(value or "#FFFFFF").lstrip("#")
    if len(raw) != 6 or any(c not in "0123456789abcdefABCDEF" for c in raw):
        raw = "FFFFFF"
    red, green, blue = raw[0:2], raw[2:4], raw[4:6]
    return f"&H{alpha}{blue}{green}{red}&"


def get_caption_template(name: str, overrides: dict | None = None) -> dict:
    """Return a fresh template merged with validated UI overrides."""
    template = dict(CAPTION_TEMPLATES.get(name, CAPTION_TEMPLATES["Gaming Style"]))
    template.update(overrides or {})
    template["fontsize_vertical"] = max(24, min(120, int(template["fontsize_vertical"])))
    template["fontsize_horizontal"] = max(18, min(90, int(template["fontsize_horizontal"])))
    template["max_words_per_cue"] = max(1, min(10, int(template["max_words_per_cue"])))
    for key in ("base_color", "highlight_color", "outline_color"):
        template[key] = hex_to_ass(template[key])
    return template


def load_custom_templates(path: str | None = None) -> dict:
    """Load user templates from the app's local database directory."""
    path = path or os.path.join(config.DATABASE_DIR, "caption_templates.json")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def save_custom_template(name: str, style: dict, path: str | None = None) -> None:
    """Persist a named user template, keeping built-in names reserved."""
    name = " ".join(str(name).split())[:40]
    if not name or name in CAPTION_TEMPLATES:
        raise ValueError("Choose a non-empty name that does not replace a built-in template.")
    path = path or os.path.join(config.DATABASE_DIR, "caption_templates.json")
    templates = load_custom_templates(path)
    templates[name] = {k: style[k] for k in (
        "fontname", "fontsize_vertical", "fontsize_horizontal", "base_color",
        "highlight_color", "outline_color", "bold", "outline", "shadow",
        "max_words_per_cue", "dynamic_highlight",
    ) if k in style}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(templates, handle, indent=2)
    os.replace(temp_path, path)


def reset_custom_templates(path: str | None = None) -> None:
    """Remove custom templates while leaving built-in presets intact."""
    path = path or os.path.join(config.DATABASE_DIR, "caption_templates.json")
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
