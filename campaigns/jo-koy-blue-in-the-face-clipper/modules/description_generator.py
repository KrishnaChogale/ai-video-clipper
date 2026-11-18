"""
modules/description_generator.py
Generates platform-specific social media descriptions and hashtags per clip.
Ollama-powered with template fallbacks.
"""
import logging
from typing import Union

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)

HASHTAG_POOLS = {
    "punchline": ["#JoKoy", "#BlueInTheFace", "#StandUpComedy", "#ComedyClips", "#Punchline"],
    "crowd_reaction": ["#JoKoy", "#BlueInTheFace", "#StandUpComedy", "#CrowdWork", "#ComedyClips"],
    "storytelling": ["#JoKoy", "#BlueInTheFace", "#StandUpComedy", "#ComedyStory", "#ComedyClips"],
    "physical_comedy": ["#JoKoy", "#BlueInTheFace", "#StandUpComedy", "#PhysicalComedy", "#ComedyClips"],
    "unknown": ["#JoKoy", "#BlueInTheFace", "#StandUpComedy", "#ComedyClips"],
}

PLATFORM_TONE = {
    "YouTube Shorts": "conversational and accurate to the transcript",
    "Instagram Reels": "warm, concise, and accurate to the clip",
    "TikTok": "casual, concise, and accurate to the clip",
}

PLATFORM_CTAS = {
    "YouTube Shorts": "Follow for more campaign clips.",
    "Instagram Reels": "Follow for more campaign clips.",
    "TikTok": "Follow for more campaign clips.",
}

FALLBACK_DESCRIPTIONS = {
    "YouTube Shorts": (
        "A moment from Jo Koy: Blue in the Face. Which part stood out to you?\n\n"
        "{hashtags}"
    ),
    "Instagram Reels": (
        "A moment from Jo Koy: Blue in the Face 🎤 Which part landed for you?\n\n"
        "{hashtags}"
    ),
    "TikTok": (
        "A moment from Blue in the Face 🎤 What stood out to you?\n{hashtags}"
    ),
}


def _extract_clip_context(clip_input: Union[dict, str], title_override: str = ""):
    if isinstance(clip_input, dict):
        snippet = clip_input.get("transcript_snippet") or clip_input.get("snippet") or ""
        category = clip_input.get("category", "unknown")
        events = ", ".join(clip_input.get("events", [])) or "stand-up performance"
        title = clip_input.get("selected_title") or clip_input.get("title") or title_override or "Jo Koy: Blue in the Face"
    else:
        snippet = str(clip_input)
        category = "unknown"
        events = "stand-up performance"
        title = title_override or "Jo Koy: Blue in the Face"
    return snippet, category, events, title


def _build_desc_prompt(snippet: str, category: str, events: str, title: str, platform: str, style: str) -> str:
    tone = PLATFORM_TONE.get(platform, "engaging")
    return (
        f"Write a {platform} description for a Jo Koy: Blue in the Face stand-up clip.\n"
        f"Clip title: \"{title}\"\n"
        f"Category: {category} | Events: {events} | Style: {style}\n"
        f"Transcript: \"{snippet[:250]}\"\n"
        f"Tone: {tone}\n"
        f"Rules:\n"
        f"- 2–4 sentences max\n"
        f"- Include 1 question or conversational hook\n"
        f"- Include a CTA: \"{PLATFORM_CTAS.get(platform, '')}\"\n"
        f"- End with 4–6 relevant hashtags on a new line\n"
        f"- Base description on REAL clip content only\n"
        f"- Do not invent a joke, quote, audience reaction, or story detail\n"
        f"- Output as JSON: {{\"description\": \"...\"}}"
    )


def generate_descriptions(
    clip_input: Union[dict, str],
    title: str = "",
    style: str = "balanced",
) -> dict[str, str]:
    """
    Returns dict containing both platform display names and standard keys:
    {
      "YouTube Shorts": "...",
      "Instagram Reels": "...",
      "TikTok": "...",
      "yt_description": "...",
      "ig_description": "...",
      "tt_description": "...",
    }
    """
    snippet, category, events, title = _extract_clip_context(clip_input, title_override=title)
    n_tags = config.HASHTAG_COUNT_DEFAULT
    tag_pool = HASHTAG_POOLS.get(category, HASHTAG_POOLS["unknown"])
    hashtags = " ".join(tag_pool[:n_tags])

    descriptions = {}

    for platform in config.PLATFORMS:
        prompt = _build_desc_prompt(snippet, category, events, title, platform, style)
        try:
            import ollama
            from pydantic import BaseModel

            class DescOutput(BaseModel):
                description: str

            response = ollama.chat(
                model=config.OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                format=DescOutput.model_json_schema(),
                options={"temperature": 0.6, "num_predict": 250},
            )
            result = DescOutput.model_validate_json(response.message.content)
            desc_text = result.description.strip()
        except Exception as e:
            logger.debug("Ollama description fallback for %s: %s", platform, e)
            tmpl = FALLBACK_DESCRIPTIONS.get(platform, "{hashtags}")
            desc_text = tmpl.format(category=category, hashtags=hashtags)

        descriptions[platform] = desc_text

    # Populate standard aliases
    descriptions["yt_description"] = descriptions.get("YouTube Shorts", "")
    descriptions["ig_description"] = descriptions.get("Instagram Reels", "")
    descriptions["tt_description"] = descriptions.get("TikTok", "")

    return descriptions


def generate_hashtags(
    clip_input: Union[dict, str] = None,
    snippet: str = "",
    count: int = None,
) -> list[str]:
    """Returns a list of relevant hashtag strings."""
    n = count or config.HASHTAG_COUNT_DEFAULT
    category = "unknown"
    if isinstance(clip_input, dict):
        category = clip_input.get("category", "unknown")
    elif isinstance(clip_input, str):
        # Check if clip_input is a category name
        if clip_input.lower() in HASHTAG_POOLS:
            category = clip_input.lower()

    pool = HASHTAG_POOLS.get(category, HASHTAG_POOLS["unknown"])
    return pool[:n]
