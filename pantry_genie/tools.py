import os
import json
import requests
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

load_dotenv()

try:
    import streamlit as st
    for key, value in st.secrets.items():
        os.environ.setdefault(key, value)
except:
    pass

_supabase = None

def get_supabase():
    global _supabase
    if _supabase is None:
        from supabase import create_client
        _supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    return _supabase

def _get_user_id(config: RunnableConfig) -> str:
    return config.get("configurable", {}).get("user_id", "default")


# ── Tool 1: Get Pantry Contents ────────────────────────────────────────────
@tool
def get_pantry_contents(config: RunnableConfig) -> str:
    """Read the full contents of the user's pantry.
    Always call this before suggesting recipes or analysing what to cook.
    """
    user_id = _get_user_id(config)
    result = get_supabase().table("pantry").select("ingredients").eq("user_id", user_id).execute()
    if not result.data or not result.data[0].get("ingredients"):
        return "Pantry is empty."
    ingredients = result.data[0]["ingredients"]
    return f"Pantry contains {len(ingredients)} ingredients: {', '.join(ingredients)}"


# ── Tool 2: Update Pantry ─────────────────────────────────────────────────
@tool
def update_pantry(ingredients: str, config: RunnableConfig) -> str:
    """Replace the pantry with a new comma-separated list of ingredients.
    Use when the user says they bought something new or want to reset their pantry.
    Pass ALL ingredients including existing ones you want to keep.
    """
    user_id = _get_user_id(config)
    items = [i.strip().lower() for i in ingredients.split(",") if i.strip()]
    get_supabase().table("pantry").upsert({"user_id": user_id, "ingredients": items}).execute()
    return f"✅ Pantry updated with {len(items)} ingredients: {', '.join(items)}"


# ── Tool 3: Get User Preferences ──────────────────────────────────────────
@tool
def get_user_preferences(config: RunnableConfig) -> str:
    """Read the user's saved taste preferences, equipment, dislikes, and favourite cuisines.
    Always call this before suggesting recipes to personalise recommendations.
    """
    user_id = _get_user_id(config)
    result = get_supabase().table("preferences").select("*").eq("user_id", user_id).execute()
    if not result.data:
        return "No preferences saved yet."
    row = result.data[0]
    prefs = {k: v for k, v in row.items() if k not in ("user_id", "updated_at") and v}
    return json.dumps(prefs, indent=2) if prefs else "No preferences saved yet."


# ── Tool 4: Update User Preferences ──────────────────────────────────────
class UpdatePreferencesInput(BaseModel):
    spice_level: str = Field(default="", description="'low', 'medium', or 'high'")
    dislikes: List[str] = Field(default_factory=list, description="Ingredients to avoid")
    favorite_cuisines: List[str] = Field(default_factory=list, description="Preferred cuisines")

@tool(args_schema=UpdatePreferencesInput)
def update_user_preferences(
    config: RunnableConfig,
    spice_level: str = "",
    dislikes: List[str] = None,
    favorite_cuisines: List[str] = None,
) -> str:
    """Update the user's taste preferences in the database.
    Use when the user mentions likes, dislikes, spice tolerance, or favourite cuisines.
    """
    user_id = _get_user_id(config)
    dislikes = dislikes or []
    favorite_cuisines = favorite_cuisines or []
    existing = (get_supabase().table("preferences").select("*").eq("user_id", user_id).execute().data or [{}])[0]
    update = {"user_id": user_id}
    if spice_level:
        update["spice_level"] = spice_level
    if dislikes:
        update["dislikes"] = list(set((existing.get("dislikes") or []) + dislikes))
    if favorite_cuisines:
        update["favorite_cuisines"] = list(set((existing.get("favorite_cuisines") or []) + favorite_cuisines))
    get_supabase().table("preferences").upsert(update).execute()
    return "✅ Preferences updated."


# ── Tool 5: Search YouTube ────────────────────────────────────────────────
@tool
def search_youtube(recipe_name: str) -> str:
    """Find a YouTube cooking video for a recipe. Call once per recipe, after the recipe is designed.
    recipe_name: the full descriptive name of the dish, e.g. 'spicy Thai chickpea curry'
    """
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        return "YouTube search unavailable."
    try:
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
        if items:
            vid   = items[0]["id"]["videoId"]
            title = items[0]["snippet"]["title"]
            return f"[▶️ {title}](https://www.youtube.com/watch?v={vid})"
    except Exception:
        pass
    return "No video found."


# ── Tool 6: Find Vegetarian Substitution ──────────────────────────────────
@tool
def find_vegetarian_substitution(ingredient: str, role: str = "general") -> str:
    """Find the best vegetarian substitute for a missing or non-vegetarian ingredient.
    ingredient: what to replace (e.g. 'chicken', 'fish sauce', 'parmesan')
    role: its purpose in the dish — 'protein', 'umami', 'binding', 'richness', 'flavour'
    Returns substitutes with usage notes so the recipe still works.
    """
    SUBS = {
        "chicken":         "tofu (pressed, cubed), tempeh, chickpeas, or jackfruit for shredded texture",
        "beef":            "lentils (for bulk), mushrooms + soy sauce (for umami), black beans, or tempeh",
        "lamb":            "jackfruit (braised), eggplant, or mushrooms with rosemary",
        "fish":            "banana blossom or hearts of palm (flaky texture), tofu, or marinated jackfruit",
        "shrimp":          "king oyster mushrooms sliced into rounds, or hearts of palm",
        "bacon":           "smoked tofu, tempeh bacon, or sun-dried tomatoes for savoury depth",
        "fish sauce":      "soy sauce + a pinch of nori flakes (or 1 tsp miso + lime juice)",
        "oyster sauce":    "hoisin sauce, or soy sauce + sugar + a dash of mushroom sauce",
        "worcestershire":  "soy sauce + apple cider vinegar + a drop of molasses",
        "anchovies":       "capers, olives, or miso paste — all deliver that briny umami hit",
        "shrimp paste":    "miso paste + nori powder, or fermented black bean paste",
        "gelatin":         "agar-agar (1:1 swap), pectin, or cornstarch for thickening",
        "lard":            "coconut oil or refined coconut butter (same fat ratio)",
        "parmesan":        "nutritional yeast + salt + toasted breadcrumbs, or cashew parmesan",
        "honey":           "maple syrup or agave nectar (1:1)",
        "beef stock":      "mushroom stock or dark vegetable stock with a splash of soy sauce",
        "chicken stock":   "light vegetable stock or water with nutritional yeast and a bay leaf",
    }
    low = ingredient.lower().strip()
    for key, sub in SUBS.items():
        if key in low or low in key:
            return f"**Substitute for {ingredient}:** {sub}"

    role_map = {
        "protein":  "tofu (pressed), tempeh, chickpeas, lentils, or paneer",
        "umami":    "mushrooms, miso paste, soy sauce, nutritional yeast, or sun-dried tomatoes",
        "binding":  "flax egg (1 tbsp ground flax + 3 tbsp water), chia egg, or mashed banana",
        "richness": "coconut cream, cashew cream, or full-fat coconut milk",
        "flavour":  "fresh herbs, lemon zest, toasted spices, or a splash of vinegar",
    }
    fallback = role_map.get(role.lower(), "tofu, tempeh, or mushrooms depending on context")
    return (
        f"No exact match for '{ingredient}'. "
        f"For a **{role}** role, try: {fallback}."
    )


# ── Tool 7: Analyse What to Buy ───────────────────────────────────────────
@tool
def analyze_shopping_gaps(config: RunnableConfig) -> str:
    """Identify the highest-value ingredients the user should buy next.
    Analyses the current pantry and preferred cuisines to find the 5-6 items
    that would unlock the greatest number of new recipes.
    Call this when the user asks what to buy, or to provide proactive shopping advice.
    """
    user_id = _get_user_id(config)

    pantry_res = get_supabase().table("pantry").select("ingredients").eq("user_id", user_id).execute()
    pantry = {i.lower() for i in (pantry_res.data[0]["ingredients"] if pantry_res.data else [])}

    prefs_res = get_supabase().table("preferences").select("*").eq("user_id", user_id).execute()
    prefs     = prefs_res.data[0] if prefs_res.data else {}
    cuisines  = [c.lower() for c in (prefs.get("favorite_cuisines") or [])]
    equip     = [e.lower() for e in (prefs.get("equipment") or [])]

    # (ingredient, base_score, cuisine_tags, equipment_needed, what_it_unlocks)
    STAPLES = [
        ("garlic",           18, ["all"],                         [],               "virtually every cuisine"),
        ("onion",            18, ["all"],                         [],               "virtually every cuisine"),
        ("olive oil",        15, ["italian","mediterranean"],      [],               "sautéing, roasting, dressings"),
        ("canned tomatoes",  14, ["italian","indian","mexican"],   [],               "pasta sauces, curries, stews"),
        ("coconut milk",     13, ["thai","indian","vietnamese"],   [],               "curries, soups, desserts"),
        ("soy sauce",        13, ["chinese","japanese","korean","thai"], [],          "stir-fries, marinades, umami boost"),
        ("cumin",            12, ["indian","mexican","middle eastern"], [],           "spice base for 12+ dishes"),
        ("ginger",           11, ["indian","thai","chinese","korean"], [],            "curries, stir-fries, soups"),
        ("chickpeas",        10, ["indian","mediterranean","middle eastern"], [],     "curries, salads, hummus, soups"),
        ("lentils",          10, ["indian","middle eastern","mediterranean"], [],     "dals, soups, salads"),
        ("nutritional yeast", 9, ["all"],                         [],               "cheesy flavour, B12 source"),
        ("tahini",            8, ["middle eastern","mediterranean"], [],              "hummus, dressings, sauces"),
        ("garam masala",      9, ["indian"],                      [],               "unlocks all Indian curries"),
        ("instant pot",       0, ["all"],   ["instant pot"],                          "cuts cook time by 60%"),
        ("tofu",             10, ["chinese","japanese","thai","korean"], [],          "protein in Asian dishes"),
        ("pasta",             9, ["italian"],                     [],               "quick weeknight meals"),
        ("rice",             10, ["all"],                         [],               "pairs with nearly any dish"),
        ("miso paste",        8, ["japanese","korean"],           [],               "soups, marinades, dressings"),
        ("lemon",             8, ["mediterranean","middle eastern","greek"], [],      "brightness in any dish"),
    ]

    gaps = []
    for item, score, cui_tags, equip_tags, unlocks in STAPLES:
        if item in pantry:
            continue
        # Equipment items: skip if user doesn't have that equipment listed
        if equip_tags and not any(e in equip for e in equip_tags):
            continue
        boost = 0
        if "all" in cui_tags:
            boost = 4
        elif cuisines:
            matches = sum(1 for c in cuisines if any(t in c or c in t for t in cui_tags))
            boost = matches * 3
        gaps.append((item, score + boost, unlocks))

    gaps.sort(key=lambda x: -x[1])
    top = gaps[:6]

    if not top:
        return "🎉 Your pantry already has all the high-value staples — you're well-stocked!"

    lines = ["**Buy these to unlock the most new recipes:**\n"]
    for item, _, unlocks in top:
        lines.append(f"• **{item.capitalize()}** — {unlocks}")
    return "\n".join(lines)


# ── Tool 8: Estimate Meal Nutrition ───────────────────────────────────────
@tool
def estimate_meal_nutrition(ingredients: str) -> str:
    """Estimate the nutritional profile of a meal from its main ingredients.
    ingredients: comma-separated ingredient list from a recipe
    Returns protein/carb/fat/fibre estimates and flags any major nutritional gaps.
    Call this for every suggested recipe so users know if a meal is balanced.
    """
    # (protein_g, carb_g, fat_g, fibre_g) per typical serving contribution
    NUTRITION = {
        "chickpeas":        (9, 22,  2, 6),
        "lentils":          (9, 20,  1, 8),
        "black beans":      (8, 21,  1, 8),
        "kidney beans":     (8, 22,  1, 7),
        "tofu":             (10, 2,  5, 0),
        "tempeh":           (16, 9,  9, 0),
        "paneer":           (14, 3, 20, 0),
        "eggs":             (12, 1, 10, 0),
        "yogurt":           (6,  8,  3, 0),
        "pasta":            (8, 43,  1, 2),
        "rice":             (4, 45,  0, 1),
        "quinoa":           (8, 39,  4, 5),
        "oats":             (6, 27,  3, 4),
        "spinach":          (3,  4,  0, 2),
        "broccoli":         (3,  6,  0, 2),
        "kale":             (3,  6,  0, 2),
        "sweet potato":     (2, 27,  0, 4),
        "potatoes":         (3, 37,  0, 3),
        "mushrooms":        (2,  3,  0, 1),
        "walnuts":          (5,  4, 19, 2),
        "almonds":          (6,  6, 14, 3),
        "cashews":          (5,  9, 12, 1),
        "coconut milk":     (1,  6, 14, 0),
        "olive oil":        (0,  0, 14, 0),
        "butter":           (0,  0, 12, 0),
    }

    items = [i.strip().lower() for i in ingredients.split(",")]
    protein = carbs = fat = fibre = 0
    for item in items:
        for key, (p, c, f, fi) in NUTRITION.items():
            if key in item or item in key:
                protein += p; carbs += c; fat += f; fibre += fi
                break

    flags = []
    if protein < 12:
        flags.append("⚠️ **Low protein** — add legumes, tofu, eggs, or paneer to make it more filling")
    if fibre < 5:
        flags.append("⚠️ **Low fibre** — add more vegetables, legumes, or swap to whole-grain pasta/rice")
    if fat < 5:
        flags.append("⚠️ **Very low fat** — a drizzle of olive oil or handful of nuts aids vitamin absorption")

    lines = [
        f"**Nutrition estimate** (per serving):",
        f"Protein ~{protein}g · Carbs ~{carbs}g · Fat ~{fat}g · Fibre ~{fibre}g",
    ]
    if flags:
        lines += [""] + flags
    else:
        lines.append("✅ Nutritionally well-rounded meal!")

    return "\n".join(lines)


# ── Exports ───────────────────────────────────────────────────────────────
ANALYST_TOOLS = [get_pantry_contents, get_user_preferences]

RECIPE_TOOLS = [find_vegetarian_substitution, search_youtube]

WELLNESS_TOOLS = [estimate_meal_nutrition, analyze_shopping_gaps]

SUPERVISOR_TOOLS = [update_pantry, update_user_preferences]
