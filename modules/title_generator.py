"""
modules/title_generator.py
Generates 3 viral title alternatives per clip using Ollama (llama3.2).
Falls back to template-based titles when Ollama is unavailable.
"""
import logging
import re
from typing import Union

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)

# Template-based fallback titles per category
TITLE_TEMPLATES = {
    "action": [
        "GTA 6 This Was INSANE 🔥",
        "GTA 6 Action You WON'T Believe 💀",
        "Nobody Expected This GTA 6 Moment 😱",
    ],
    "funny": [
        "GTA 6 NPCs Have NO Survival Instinct 😂",
        "GTA 6 Physics Are Completely Broken 💀",
        "GTA 6 Is Wild Right Now 😂",
    ],
    "cinematic": [
        "GTA 6 Looks Like Real Life",
        "This GTA 6 Scene Hits Different 🎬",
        "GTA 6 Is Actually Beautiful",
    ],
    "dialogue": [
        "GTA 6 Writing Is On Another Level",
        "This GTA 6 Dialogue Got Me 👀",
        "GTA 6 Characters Are So Real",
    ],
    "discovery": [
        "Nobody Talks About This GTA 6 Detail",
        "GTA 6 Hidden Detail You Missed",
        "GTA 6 Easter Egg Found 👀",
    ],
    "unknown": [
        "GTA 6 Moment You NEED To See",
        "GTA 6 Gameplay Is Different Now",
        "GTA 6 Just Hit Different 🔥",
    ],
}

HOOK_TEMPLATES = {
    "action": [
        "WHAT HAPPENED NEXT WAS INSANE 💀",
        "GTA 6 POLICE CHASE GONE WRONG 🚨",
        "BRO FORGOT HOW TO DRIVE 😂",
    ],
    "funny": [
        "THE NPC SAID WHAT?! 😂",
        "GTA 6 PHYSICS SAID NO 💀",
        "I WAS NOT EXPECTING THAT 😂",
    ],
    "cinematic": [
        "THIS LOOKS LIKE A REAL MOVIE 🎬",
        "GTA 6 IS GORGEOUS 😮",
        "ROCKSTAR REALLY DID THIS 🤯",
    ],
    "dialogue": [
        "WAIT FOR WHAT THEY SAY 👀",
        "THE DIALOGUE IN GTA 6 IS CRAZY",
        "THIS LINE HIT DIFFERENT 💀",
    ],
    "discovery": [
        "NOBODY NOTICED THIS 👀",
        "HIDDEN GTA 6 DETAIL FOUND 🕵️",
        "THEY HID THIS IN GTA 6 😱",
    ],
    "unknown": [
        "YOU NEED TO SEE THIS 🔥",
        "WAIT FOR IT... 😱",
        "GTA 6 IS DIFFERENT 💀",
    ],
}


def _extract_clip_context(clip_input: Union[dict, str]) -> tuple[str, str, str, float]:
    """Helper to safely extract (snippet, category, events_str, duration)."""
    if isinstance(clip_input, dict):
        snippet = clip_input.get("transcript_snippet") or clip_input.get("snippet") or ""
        category = clip_input.get("category", "unknown")
        events = ", ".join(clip_input.get("events", [])) or "general gameplay"
        start = clip_input.get("start", 0)
        end = clip_input.get("end", 30)
        duration = max(1.0, float(end - start))
    else:
        snippet = str(clip_input)
        category = "unknown"
        events = "general gameplay"
        duration = 30.0
    return snippet, category, events, duration


def _build_prompt(snippet: str, category: str, events: str, duration: float, style: str) -> str:
    return (
        f"You are a viral short-form gaming content creator specialising in GTA 6.\n"
        f"Generate EXACTLY 3 alternative titles for a {duration:.0f}-second GTA 6 clip.\n"
        f"Category: {category} | Events detected: {events} | Style: {style}\n"
        f"Transcript snippet: \"{snippet[:300]}\"\n\n"
        f"Rules:\n"
        f"- Each title under 60 characters\n"
        f"- Create genuine curiosity without clickbait lies\n"
        f"- Use emojis sparingly (max 1-2 per title)\n"
        f"- Different angles: one suspenseful, one descriptive, one reaction-based\n"
        f"- Base titles on ACTUAL events in the clip, not invented ones\n"
        f"- Output as JSON: {{\"titles\": [\"title1\", \"title2\", \"title3\"]}}"
    )


def generate_titles(clip_input: Union[dict, str], style: str = "balanced") -> list[str]:
    """
    Returns 3 title strings. Uses Ollama if available, templates as fallback.
    Accepts either a clip dict or transcript string snippet.
    """
    snippet, category, events, duration = _extract_clip_context(clip_input)
    prompt = _build_prompt(snippet, category, events, duration, style)

    try:
        import ollama
        from pydantic import BaseModel

        class TitleOutput(BaseModel):
            titles: list[str]

        response = ollama.chat(
            model=config.OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format=TitleOutput.model_json_schema(),
            options={"temperature": 0.7, "num_predict": 200},
        )
        result = TitleOutput.model_validate_json(response.message.content)
        titles = [t.strip() for t in result.titles if t.strip()][:3]
        while len(titles) < 3:
            idx = len(titles) % 3
            titles.append(TITLE_TEMPLATES.get(category, TITLE_TEMPLATES["unknown"])[idx])
        return titles
    except Exception as e:
        logger.debug("Ollama title generation fallback: %s", e)
        return list(TITLE_TEMPLATES.get(category, TITLE_TEMPLATES["unknown"]))


def generate_hooks(clip_input: Union[dict, str], style: str = "balanced") -> list[str]:
    """
    Returns 3 on-screen hook text alternatives.
    Accepts either a clip dict or transcript string snippet.
    """
    _, category, _, _ = _extract_clip_context(clip_input)
    return list(HOOK_TEMPLATES.get(category, HOOK_TEMPLATES["unknown"]))
