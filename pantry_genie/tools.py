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

# ── Lazy Supabase client ───────────────────────────────────
_supabase = None

def get_supabase():
    global _supabase
    if _supabase is None:
        from supabase import create_client
        _supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    return _supabase

def _get_user_id(config: RunnableConfig) -> str:
    return config.get("configurable", {}).get("user_id", "default")


# ── Tool 1: Get Pantry Contents ────────────────────────────
@tool
def get_pantry_contents(config: RunnableConfig) -> str:
    """Read the current contents of the user's pantry/fridge.
    Use this when the user asks what they can cook with what they have.
    """
    user_id = _get_user_id(config)
    result = get_supabase().table("pantry").select("ingredients").eq("user_id", user_id).execute()
    if not result.data or not result.data[0].get("ingredients"):
        return "Pantry is empty."
    return f"Current pantry ingredients: {', '.join(result.data[0]['ingredients'])}"


# ── Tool 2: Update Pantry ──────────────────────────────────
@tool
def update_pantry(ingredients: str, config: RunnableConfig) -> str:
    """Update the pantry with a comma-separated list of ingredients.
    Use this when the user tells you what ingredients they have.
    Always pass ingredients as a single comma-separated string like:
    'chickpeas, spinach, tomatoes'
    """
    user_id = _get_user_id(config)
    items = [i.strip() for i in ingredients.split(",") if i.strip()]
    get_supabase().table("pantry").upsert({"user_id": user_id, "ingredients": items}).execute()
    return f"✅ Pantry updated with: {', '.join(items)}"


# ── Tool 3: Get User Preferences ──────────────────────────
@tool
def get_user_preferences(config: RunnableConfig) -> str:
    """Read the user's stored taste preferences, dislikes and dietary needs.
    Use this before making recipe suggestions to personalize recommendations.
    """
    user_id = _get_user_id(config)
    result = get_supabase().table("preferences").select("*").eq("user_id", user_id).execute()
    if not result.data:
        return "No preferences saved yet."
    row = result.data[0]
    prefs = {k: v for k, v in row.items() if k not in ("user_id", "updated_at") and v}
    return json.dumps(prefs, indent=2) if prefs else "No preferences saved yet."


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
    user_id = _get_user_id(config)
    dislikes = dislikes or []
    favorite_cuisines = favorite_cuisines or []

    result = get_supabase().table("preferences").select("*").eq("user_id", user_id).execute()
    existing = result.data[0] if result.data else {}

    update = {"user_id": user_id}
    if spice_level:
        update["spice_level"] = spice_level
    if dislikes:
        update["dislikes"] = list(set(existing.get("dislikes", []) + dislikes))
    if favorite_cuisines:
        update["favorite_cuisines"] = list(set(existing.get("favorite_cuisines", []) + favorite_cuisines))

    get_supabase().table("preferences").upsert(update).execute()
    return f"✅ Preferences updated."


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
            "q": f"{recipe_name} vegetarian recipe",
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
