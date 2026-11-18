import os
import uuid

import config
from modules.caption_templates import (
    CAPTION_TEMPLATES, get_caption_template, hex_to_ass,
    load_custom_templates, reset_custom_templates, save_custom_template,
)
from modules.subtitle_generator import generate_clip_ass


def test_hex_to_ass_uses_ass_bgr_order():
    assert hex_to_ass("#123456") == "&H00563412&"


def test_all_caption_templates_render_with_distinct_styles():
    outputs = {}
    words = [
        {"start": 1.0, "end": 1.3, "word": "that"},
        {"start": 1.3, "end": 1.6, "word": "was"},
        {"start": 1.6, "end": 2.0, "word": "great!"},
    ]
    for name in CAPTION_TEMPLATES:
        style = get_caption_template(name)
        ass = generate_clip_ass(
            words_list=words, clip_start=0, clip_end=4,
            crop_vertical=True, caption_style=style,
        )
        assert "[Events]" in ass
        assert "Dialogue:" in ass
        assert f"Style: ViralShorts,{style['fontname']},{style['fontsize_vertical']}" in ass
        outputs[name] = ass
    assert len(set(outputs.values())) == len(CAPTION_TEMPLATES)


def test_template_overrides_are_bounded_and_applied():
    style = get_caption_template("Gaming Style", {
        "fontsize_vertical": 999,
        "max_words_per_cue": 50,
        "highlight_color": "#123456",
    })
    assert style["fontsize_vertical"] == 120
    assert style["max_words_per_cue"] == 10
    assert style["highlight_color"] == "&H00563412&"


def test_custom_template_save_and_reset():
    path = os.path.join(config.TEMP_DIR, f"caption_templates_test_{uuid.uuid4().hex}.json")
    try:
        save_custom_template("My Style", CAPTION_TEMPLATES["Gaming Style"], path)
        saved = load_custom_templates(path)
        assert saved["My Style"]["fontname"] == "Arial"
        reset_custom_templates(path)
        assert load_custom_templates(path) == {}
    finally:
        reset_custom_templates(path)

