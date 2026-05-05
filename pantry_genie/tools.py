import os
import json
import requests
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

load_dotenv()

# ── Load Streamlit secrets if on cloud ─────────────────────
try:
    import streamlit as st
    for key, value in st.secrets.items():
        os.environ.setdefault(key, value)
except:
    pass

# ── Memory directory ───────────────────────────────────────
MEMORY_DIR = "/tmp/pantry_genie_memory"
os.makedirs(MEMORY_DIR, exist_ok=True)

def _get_thread_id(config: RunnableConfig) -> str:
    return config.get("configurable", {}).get("thread_id", "default")

def _pantry_file(thread_id: str) -> str:
    path = f"{MEMORY_DIR}/pantry_{thread_id}.json"
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump({"ingredients": []}, f)
    return path

def _profile_file(thread_id: str) -> str:
    path = f"{MEMORY_DIR}/profile_{thread_id}.json"
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump({}, f)
    return path


# ── Tool 1: Get Pantry Contents ────────────────────────────
@tool
def get_pantry_contents(config: RunnableConfig) -> str:
    """Read the current contents of the user's pantry/fridge.
    Use this when the user asks what they can cook with what they have.
    """
    with open(_pantry_file(_get_thread_id(config))) as f:
        data = json.load(f)
    ingredients = data.get("ingredients", [])
    if not ingredients:
        return "Pantry is empty."
    return f"Current pantry ingredients: {', '.join(ingredients)}"


# ── Tool 2: Update Pantry ──────────────────────────────────
@tool
def update_pantry(ingredients: str, config: RunnableConfig) -> str:
    """Update the pantry with a comma-separated list of ingredients.
    Use this when the user tells you what ingredients they have.
    Always pass ingredients as a single comma-separated string like:
    'chickpeas, spinach, tomatoes'
    """
    items = [i.strip() for i in ingredients.split(",") if i.strip()]
    with open(_pantry_file(_get_thread_id(config)), "w") as f:
        json.dump({"ingredients": items}, f, indent=2)
    return f"✅ Pantry updated with: {', '.join(items)}"


# ── Tool 3: Get User Preferences ──────────────────────────
@tool
def get_user_preferences(config: RunnableConfig) -> str:
    """Read the user's stored taste preferences, dislikes and dietary needs.
    Use this before making recipe suggestions to personalize recommendations.
    """
    with open(_profile_file(_get_thread_id(config))) as f:
        data = json.load(f)
    if not data:
        return "No preferences saved yet."
    return json.dumps(data, indent=2)


# ── Tool 4: Update User Preferences ───────────────────────
class UpdatePreferencesInput(BaseModel):
    spice_level: str = Field(default="", description="Spice level: 'low', 'medium', or 'high'")
    dislikes: List[str] = Field(default_factory=list, description="Ingredients the user dislikes, e.g. ['coriander', 'mushrooms']")
    favorite_cuisines: List[str] = Field(default_factory=list, description="Cuisines the user likes, e.g. ['Indian', 'Thai']")

@tool(args_schema=UpdatePreferencesInput)
def update_user_preferences(
    config: RunnableConfig,
    spice_level: str = "",
    dislikes: List[str] = None,
    favorite_cuisines: List[str] = None,
) -> str:
    """Update the user's taste preferences.
    Use this when user mentions they like/dislike something, their spice level, or favourite cuisines.
    """
    dislikes = dislikes or []
    favorite_cuisines = favorite_cuisines or []
    profile_path = _profile_file(_get_thread_id(config))
    with open(profile_path) as f:
        existing = json.load(f)
    if spice_level:
        existing["spice_level"] = spice_level
    if dislikes:
        existing["dislikes"] = list(set(existing.get("dislikes", []) + dislikes))
    if favorite_cuisines:
        existing["favorite_cuisines"] = list(set(existing.get("favorite_cuisines", []) + favorite_cuisines))
    with open(profile_path, "w") as f:
        json.dump(existing, f, indent=2)
    return f"✅ Preferences updated: {json.dumps(existing, indent=2)}"


# ── Tool 5: Search YouTube ─────────────────────────────────
@tool
def search_youtube(recipe_name: str) -> str:
    """Find a YouTube video for a given vegan recipe name.
    Use this after suggesting a recipe to provide a video link.
    """
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        return "YouTube search unavailable (no API key configured)."
    resp = requests.get(
        "https://www.googleapis.com/youtube/v3/search",
        params={
            "part": "snippet",
            "q": f"{recipe_name} vegan recipe",
            "type": "video",
            "maxResults": 1,
            "key": api_key,
        },
        timeout=5,
    )
    resp.raise_for_status()
    items = resp.json().get("items", [])
    if not items:
        return "No YouTube video found for this recipe."
    video_id = items[0]["id"]["videoId"]
    title = items[0]["snippet"]["title"]
    return f"▶️ [{title}](https://www.youtube.com/watch?v={video_id})"


# ── Export all tools ───────────────────────────────────────
TOOLS = [
    get_pantry_contents,
    update_pantry,
    get_user_preferences,
    update_user_preferences,
    search_youtube,
]
