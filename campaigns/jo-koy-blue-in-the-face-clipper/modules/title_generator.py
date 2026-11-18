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

# Campaign fallback titles. Keep unknown moments neutral rather than inventing a joke.
TITLE_TEMPLATES = {
    "punchline": [
        "That Jo Koy Punchline Landed",
        "The Setup Made This Line Better",
        "Jo Koy Had the Room on This One",
    ],
    "crowd_reaction": [
        "Listen to the Room After That Line",
        "The Crowd Loved This Jo Koy Bit",
        "That Audience Reaction Says It All",
    ],
    "storytelling": [
        "Jo Koy's Story Takes a Turn",
        "The Detail That Makes This Story",
        "A Jo Koy Story Worth Hearing",
    ],
    "physical_comedy": [
        "Jo Koy Acts This Story Out",
        "The Physical Bit Makes the Story",
        "Jo Koy's Expression Says It All",
    ],
    "unknown": [
        "A Jo Koy Moment Worth Replaying",
        "Jo Koy: Blue in the Face",
        "A Moment From Blue in the Face",
    ],
}

HOOK_TEMPLATES = {
    "punchline": [
        "Wait for the line that follows",
        "Listen to how Jo Koy tells this",
        "Watch the room after the punchline",
    ],
    "crowd_reaction": [
        "Wait for the crowd's reaction",
        "The way he tells this story",
        "Listen to the punchline",
    ],
    "storytelling": [
        "See how the story comes together",
        "Listen to this Jo Koy bit",
        "Wait for the turn in this story",
    ],
    "physical_comedy": [
        "Wait for the next line",
        "Listen to how this lands",
        "This line turns the story",
    ],
    "unknown": [
        "Listen to this moment",
        "A moment from Blue in the Face",
        "Watch how this story unfolds",
    ],
}


def _extract_clip_context(clip_input: Union[dict, str]) -> tuple[str, str, str, float]:
    """Helper to safely extract (snippet, category, events_str, duration)."""
    if isinstance(clip_input, dict):
        snippet = clip_input.get("transcript_snippet") or clip_input.get("snippet") or ""
        category = clip_input.get("category", "unknown")
        events = ", ".join(clip_input.get("events", [])) or "stand-up performance"
        start = clip_input.get("start", 0)
        end = clip_input.get("end", 30)
        duration = max(1.0, float(end - start))
    else:
        snippet = str(clip_input)
        category = "unknown"
        events = "stand-up performance"
        duration = 30.0
    return snippet, category, events, duration


def _build_prompt(snippet: str, category: str, events: str, duration: float, style: str) -> str:
    return (
        f"You write accurate short-form titles for Jo Koy: Blue in the Face stand-up clips.\n"
        f"Generate EXACTLY 3 alternative titles for a {duration:.0f}-second clip.\n"
        f"Category: {category} | Events detected: {events} | Style: {style}\n"
        f"Transcript snippet: \"{snippet[:300]}\"\n\n"
        f"Rules:\n"
        f"- Each title under 60 characters\n"
        f"- Create genuine curiosity without clickbait lies\n"
        f"- Use emojis sparingly (max 1-2 per title)\n"
        f"- Different angles: one suspenseful, one descriptive, one reaction-based\n"
        f"- Base titles only on the supplied transcript and supported event evidence\n"
        f"- Do not invent a joke, quote, audience reaction, or story detail\n"
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


def generate_pov_hook(clip_input: dict) -> str:
    """Create one short POV-style opening grounded in the clip transcript."""
    snippet = (clip_input or {}).get("transcript_snippet", "").strip()
    if snippet:
        prompt = (
            "Write one short on-screen POV hook for a Jo Koy stand-up clip. "
            "It must start with 'POV:' and create curiosity without giving away the payoff. "
            "Use only details in the transcript; do not invent a person, joke, or reaction. "
            "Maximum 80 characters. Return only the hook.\n\n"
            f"Transcript: {snippet[:500]}"
        )
        try:
            import ollama
            response = ollama.generate(
                model=config.OLLAMA_MODEL,
                prompt=prompt,
                options={"temperature": 0.45, "num_predict": 48},
            )
            hook = (response.get("response", "") if isinstance(response, dict)
                    else getattr(response, "response", "")).strip().strip('"“”')
            hook = " ".join(hook.split())
            if hook and len(hook) <= 100:
                return hook if hook.lower().startswith("pov:") else f"POV: {hook}"
        except Exception as e:
            logger.debug("POV hook fallback: %s", e)
        phrase = " ".join(snippet.split()[:8]).strip(".,!?;:")
        if phrase:
            return f"POV: Jo Koy says, ‘{phrase}…’"
    return "POV: You're watching Jo Koy's set"
