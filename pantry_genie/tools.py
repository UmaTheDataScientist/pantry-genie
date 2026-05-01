import os
import json
import threading
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

# ── Load Streamlit secrets if on cloud ─────────────────────
try:
    import streamlit as st
    for key, value in st.secrets.items():
        os.environ.setdefault(key, value)
except:
    pass

# ── Thread-local user session ──────────────────────────────
_thread_local = threading.local()

# ── Lazy Pinecone init ─────────────────────────────────────
_pinecone_index = None

def get_index():
    global _pinecone_index
    if _pinecone_index is None:
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        _pinecone_index = pc.Index(os.getenv("PINECONE_INDEX"))
    return _pinecone_index

# ── Memory directory ───────────────────────────────────────
MEMORY_DIR = "/tmp/pantry_genie_memory"
os.makedirs(MEMORY_DIR, exist_ok=True)

def get_pantry_file() -> str:
    thread_id = getattr(_thread_local, "thread_id", "default")
    path = f"{MEMORY_DIR}/pantry_{thread_id}.json"
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump({"ingredients": []}, f)
    return path

def get_profile_file() -> str:
    thread_id = getattr(_thread_local, "thread_id", "default")
    path = f"{MEMORY_DIR}/profile_{thread_id}.json"
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump({}, f)
    return path


# ── Tool 1: Search Recipes ─────────────────────────────────
@tool
def search_recipes(query: str) -> str:
    """Search for vegan recipes in Pinecone based on ingredients or description.
    Use this when the user mentions ingredients or asks for recipe suggestions.
    """
    results = get_index().search(
        namespace="recipes",
        query={
            "inputs": {"text": query},
            "top_k": 3
        },
        fields=["recipe_name", "ingredients", "directions", "cuisine", "total_time", "nutrition"]
    )

    if not results or not results.get("result", {}).get("hits"):
        return "No matching recipes found."

    hits = results["result"]["hits"]
    output = []
    for hit in hits:
        fields = hit.get("fields", {})
        output.append(f"""
🍽️  {fields.get('recipe_name', 'Unknown')}
⏱️  Time: {fields.get('total_time', 'N/A')}
🌍  Cuisine: {fields.get('cuisine', 'N/A')}
🥕  Ingredients: {fields.get('ingredients', 'N/A')}
📋  Directions: {fields.get('directions', 'N/A')[:300]}...
🥗  Nutrition: {fields.get('nutrition', 'N/A')}
""")

    return "\n---\n".join(output)


# ── Tool 2: Get Pantry Contents ────────────────────────────
@tool
def get_pantry_contents() -> str:
    """Read the current contents of the user's pantry/fridge.
    Use this when the user asks what they can cook with what they have.
    """
    pantry_file = get_pantry_file()
    with open(pantry_file, "r") as f:
        data = json.load(f)
    ingredients = data.get("ingredients", [])
    if not ingredients:
        return "Pantry is empty."
    return f"Current pantry ingredients: {', '.join(ingredients)}"


# ── Tool 3: Update Pantry ──────────────────────────────────
@tool
def update_pantry(ingredients: str) -> str:
    """Update the pantry with a comma-separated list of ingredients.
    Use this when the user tells you what ingredients they have.
    Always pass ingredients as a single comma-separated string like:
    'chickpeas, spinach, tomatoes'
    """
    items = [i.strip() for i in ingredients.split(",") if i.strip()]
    data = {"ingredients": items}
    with open(get_pantry_file(), "w") as f:
        json.dump(data, f, indent=2)
    return f"✅ Pantry updated with: {', '.join(items)}"


# ── Tool 4: Get User Preferences ──────────────────────────
@tool
def get_user_preferences() -> str:
    """Read the user's stored taste preferences, dislikes and dietary needs.
    Use this before making recipe suggestions to personalize recommendations.
    """
    profile_file = get_profile_file()
    with open(profile_file, "r") as f:
        data = json.load(f)
    if not data:
        return "No preferences saved yet."
    return json.dumps(data, indent=2)


# ── Tool 5: Update User Preferences ───────────────────────
@tool
def update_user_preferences(spice_level: str = "", dislikes: list = None, favorite_cuisines: list = None) -> str:
    """Update the user's taste preferences.
    Use this when user mentions they like/dislike something, their spice level, or favorite cuisines.
    Args:
        spice_level: e.g. 'low', 'medium', 'high'
        dislikes: list of ingredients the user dislikes e.g. ['coriander', 'mushrooms']
        favorite_cuisines: list of cuisines e.g. ['Indian', 'Thai']
    """
    dislikes = dislikes or []
    favorite_cuisines = favorite_cuisines or []
    profile_file = get_profile_file()
    with open(profile_file, "r") as f:
        existing = json.load(f)
    if spice_level:
        existing["spice_level"] = spice_level
    if dislikes:
        existing["dislikes"] = list(set(existing.get("dislikes", []) + dislikes))
    if favorite_cuisines:
        existing["favorite_cuisines"] = list(set(existing.get("favorite_cuisines", []) + favorite_cuisines))
    with open(profile_file, "w") as f:
        json.dump(existing, f, indent=2)
    return f"✅ Preferences updated: {json.dumps(existing, indent=2)}"


# ── Export all tools ───────────────────────────────────────
TOOLS = [
    search_recipes,
    get_pantry_contents,
    update_pantry,
    get_user_preferences,
    update_user_preferences,
]