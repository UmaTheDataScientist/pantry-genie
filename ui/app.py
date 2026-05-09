import streamlit as st
import sys
import os
import json
import base64
import re
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

try:
    for key, value in st.secrets.items():
        if not isinstance(value, dict):
            os.environ.setdefault(key, value)
except:
    pass

from pantry_genie.agent import build_agent, chat

COMMON_INGREDIENTS = [
    "olive oil", "garlic", "onion", "tomatoes", "potatoes", "carrots",
    "spinach", "chickpeas", "lentils", "black beans", "kidney beans",
    "tofu", "tempeh", "paneer", "eggs", "butter", "milk", "yogurt",
    "rice", "pasta", "bread", "oats", "flour", "quinoa",
    "bell peppers", "mushrooms", "zucchini", "broccoli", "cauliflower",
    "cucumber", "celery", "corn", "peas", "sweet potato", "eggplant",
    "coconut milk", "vegetable broth", "soy sauce", "cumin", "turmeric",
    "paprika", "chili flakes", "ginger", "coriander", "basil", "oregano",
    "lemon", "lime", "walnuts", "almonds", "cashews", "sunflower seeds",
]

# ── Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="PantryGenie 🧞",
    page_icon="🧞",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    /* ── Base layout ─────────────────────────── */
    .block-container {
        padding-top: 1.2rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 1100px !important;
    }

    /* ── Mobile overrides ────────────────────── */
    @media (max-width: 640px) {
        .block-container {
            padding-left: 0.6rem !important;
            padding-right: 0.6rem !important;
            padding-top: 0.6rem !important;
        }
        h1 { font-size: 1.55rem !important; }
        h2 { font-size: 1.15rem !important; }
        h3 { font-size: 1.05rem !important; }
        /* Stack tab labels smaller so they fit on one line */
        button[data-baseweb="tab"] { font-size: 0.78rem !important; padding: 6px 8px !important; }
    }

    /* ── Typography ──────────────────────────── */
    h1 { text-align: center; }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 0.9em;
        margin-top: -12px;
        margin-bottom: 16px;
    }
    .empty-state {
        text-align: center;
        color: #bbb;
        margin-top: 40px;
        margin-bottom: 40px;
    }

    /* ── Tap targets: all buttons at least 40px tall ── */
    div[data-testid="stButton"] > button {
        min-height: 40px !important;
        border-radius: 8px !important;
    }

    /* ── Pill chips for ingredient suggestions ── */
    .chip-row div[data-testid="stButton"] > button {
        border-radius: 999px !important;
        background-color: #f0f7f0 !important;
        border: 1px solid #b5d9b5 !important;
        color: #2e7d32 !important;
        font-size: 0.80em !important;
        min-height: 34px !important;
        padding: 0 10px !important;
        transition: background-color 0.15s;
    }
    .chip-row div[data-testid="stButton"] > button:hover {
        background-color: #d4edda !important;
        border-color: #4caf50 !important;
    }

    /* ── Recipe cards ────────────────────────── */
    .recipe-card {
        background: #f7faf7;
        border-radius: 14px;
        padding: 18px 20px 14px;
        border: 1px solid #cfe8cf;
        margin-bottom: 16px;
    }

    /* ── User header row ─────────────────────── */
    .user-row {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        gap: 8px;
        margin-bottom: 4px;
    }
    .user-row img {
        border-radius: 50%;
        width: 28px;
        height: 28px;
    }
    .user-name {
        font-size: 0.85em;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)


def _extract_field(text: str, label: str) -> str:
    m = re.search(rf'\*\*{label}:\*\*\s*(.+?)(?=\n\*\*|\Z)', text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _render_recipe_cards(text: str):
    sections = re.split(r'\n?---\n?', text)
    rendered = 0
    for section in sections:
        section = section.strip()
        if not section or '##' not in section:
            continue
        name_m = re.search(r'##\s+(.+)', section)
        name = name_m.group(1).strip() if name_m else "Recipe"
        ingredients = _extract_field(section, 'Ingredients')
        directions = _extract_field(section, 'Directions')
        cook_time = _extract_field(section, 'Cook time')
        watch = _extract_field(section, 'Watch')

        with st.container(border=True):
            st.markdown(f"### {name}")
            if ingredients:
                st.markdown(f"**Ingredients:** {ingredients}")
            if directions:
                st.markdown(f"**Directions:** {directions}")
            col_time, col_watch = st.columns([1, 2])
            with col_time:
                if cook_time:
                    st.markdown(f"**Cook time:** {cook_time}")
            with col_watch:
                if watch:
                    st.markdown(f"**Watch:** {watch}")
        rendered += 1

    if not rendered:
        st.markdown(text)


# ── OAuth setup ─────────────────────────────────────────────
from streamlit_oauth import OAuth2Component

def _secret(key):
    try:
        return os.getenv(key) or st.secrets.get(key) or ""
    except Exception:
        return os.getenv(key) or ""

_client_id = _secret("GOOGLE_CLIENT_ID")
if not _client_id:
    st.error("Google OAuth is not configured. Add GOOGLE_CLIENT_ID to Streamlit secrets.")
    st.stop()

oauth2 = OAuth2Component(
    client_id=_client_id,
    client_secret=_secret("GOOGLE_CLIENT_SECRET"),
    authorize_endpoint="https://accounts.google.com/o/oauth2/auth",
    token_endpoint="https://oauth2.googleapis.com/token",
    refresh_token_endpoint="https://oauth2.googleapis.com/token",
)

def _decode_id_token(id_token: str) -> dict:
    payload = id_token.split(".")[1]
    payload += "=" * (4 - len(payload) % 4)
    return json.loads(base64.urlsafe_b64decode(payload))

# ── Login gate ──────────────────────────────────────────────
if "user_info" not in st.session_state:
    st.title("🧞 PantryGenie")
    st.markdown('<p class="subtitle">Your personal vegetarian recipe assistant 🌱</p>', unsafe_allow_html=True)
    st.divider()
    _, col, _ = st.columns([1, 2, 1])
    with col:
        result = oauth2.authorize_button(
            name="Sign in with Google",
            redirect_uri=_secret("REDIRECT_URI") or "https://pantry-genie.streamlit.app",
            scope="openid email profile",
            key="google_login",
            use_container_width=True,
        )
    if result and "token" in result:
        id_token = result["token"].get("id_token", "")
        if id_token:
            st.session_state.user_info = _decode_id_token(id_token)
            st.session_state.token = result["token"]
            st.rerun()
    st.stop()

# ── User identity ────────────────────────────────────────────
user_info = st.session_state.user_info
user_id = user_info.get("email", "")
user_name = (user_info.get("name") or "there").split()[0]
user_picture = user_info.get("picture", "") or ""
user_full_name = user_info.get("name") or ""

# ── Supabase ─────────────────────────────────────────────────
@st.cache_resource
def get_supabase():
    from supabase import create_client
    return create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

supabase = get_supabase()

# ── Session state ─────────────────────────────────────────────
if "agent" not in st.session_state:
    with st.spinner("🧞 Waking up PantryGenie..."):
        st.session_state.agent = build_agent()

if "editing_pantry" not in st.session_state:
    st.session_state.editing_pantry = None
if "editing_shop" not in st.session_state:
    st.session_state.editing_shop = None
if "pantry_suggestions" not in st.session_state:
    st.session_state.pantry_suggestions = random.sample(COMMON_INGREDIENTS, 12)

# ── Header ───────────────────────────────────────────────────
hcol_title, hcol_user = st.columns([3, 1])
with hcol_title:
    st.markdown("## 🧞 PantryGenie")
with hcol_user:
    if user_picture:
        st.markdown(
            f'<div class="user-row"><img src="{user_picture}" />'
            f'<span class="user-name">{user_name}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f'<div class="user-row"><span class="user-name">{user_name}</span></div>', unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────
tab_recipes, tab_shop, tab_prefs = st.tabs(["🍽️ Recipes", "🛒 Shopping", "💛 Prefs"])

# ════════════════════════════════════════════════════════════
# RECIPES TAB  —  pantry panel left, recipes right
# ════════════════════════════════════════════════════════════
with tab_recipes:
    pantry_result = supabase.table("pantry").select("ingredients").eq("user_id", user_id).execute()
    ingredients = list(pantry_result.data[0]["ingredients"]) if pantry_result.data else []

    col_pantry, col_recipes = st.columns([2, 3], gap="large")

    # ── LEFT: pantry panel ────────────────────────────────────
    with col_pantry:
        st.markdown("#### 🥕 What's in your pantry?")

        # Quick-add chips
        suggestions = [s for s in st.session_state.pantry_suggestions if s not in ingredients]
        if suggestions:
            st.markdown('<div class="chip-row">', unsafe_allow_html=True)
            chip_cols = st.columns(2)
            for i, suggestion in enumerate(suggestions):
                with chip_cols[i % 2]:
                    if st.button(suggestion.capitalize(), key=f"suggest_{suggestion}", use_container_width=True):
                        updated = list(dict.fromkeys(ingredients + [suggestion]))
                        supabase.table("pantry").upsert({"user_id": user_id, "ingredients": updated}).execute()
                        st.session_state.pantry_suggestions = random.sample(COMMON_INGREDIENTS, 12)
                        st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # Custom add
        with st.form("pantry_add", clear_on_submit=True):
            new_item = st.text_input("", placeholder="Add ingredient...", label_visibility="collapsed")
            if st.form_submit_button("➕ Add", use_container_width=True) and new_item.strip():
                new_items = [i.strip().lower() for i in new_item.split(",") if i.strip()]
                updated = list(dict.fromkeys(ingredients + new_items))
                supabase.table("pantry").upsert({"user_id": user_id, "ingredients": updated}).execute()
                st.rerun()

        # Current pantry items
        if ingredients:
            st.caption(f"{len(ingredients)} item{'s' if len(ingredients) != 1 else ''}")
            for idx, item in enumerate(ingredients):
                if st.session_state.editing_pantry == idx:
                    edit_key = f"edit_pantry_val_{idx}"
                    if edit_key not in st.session_state:
                        st.session_state[edit_key] = item
                    ec1, ec2, ec3 = st.columns([4, 1, 1])
                    with ec1:
                        st.text_input("", key=edit_key, label_visibility="collapsed")
                    with ec2:
                        if st.button("✓", key=f"save_pantry_{idx}"):
                            new_val = st.session_state[edit_key].strip().lower()
                            if new_val:
                                updated = [new_val if j == idx else x for j, x in enumerate(ingredients)]
                                supabase.table("pantry").upsert({"user_id": user_id, "ingredients": updated}).execute()
                            st.session_state.editing_pantry = None
                            st.rerun()
                    with ec3:
                        if st.button("✕", key=f"cancel_pantry_{idx}"):
                            st.session_state.editing_pantry = None
                            st.rerun()
                else:
                    ic1, ic2, ic3, ic4 = st.columns([3, 1, 1, 1])
                    with ic1:
                        st.markdown(f"• {item.strip().capitalize()}")
                    with ic2:
                        if st.button("✏️", key=f"edit_pantry_{idx}"):
                            st.session_state.editing_pantry = idx
                            st.session_state[f"edit_pantry_val_{idx}"] = item
                            st.rerun()
                    with ic3:
                        if st.button("🛒", key=f"shop_pantry_{idx}", help="Move to shopping list"):
                            updated_pantry = [x for j, x in enumerate(ingredients) if j != idx]
                            supabase.table("pantry").upsert({"user_id": user_id, "ingredients": updated_pantry}).execute()
                            shop_res = supabase.table("shopping_list").select("items").eq("user_id", user_id).execute()
                            shop_items = list(shop_res.data[0]["items"]) if shop_res.data else []
                            if item.strip().lower() not in shop_items:
                                shop_items.append(item.strip().lower())
                            supabase.table("shopping_list").upsert({"user_id": user_id, "items": shop_items}).execute()
                            st.rerun()
                    with ic4:
                        if st.button("✕", key=f"del_pantry_{idx}"):
                            updated = [x for j, x in enumerate(ingredients) if j != idx]
                            supabase.table("pantry").upsert({"user_id": user_id, "ingredients": updated}).execute()
                            st.rerun()

    # ── RIGHT: recipe panel ───────────────────────────────────
    with col_recipes:
        st.markdown("#### 🍽️ Recipes")
        special_request = st.text_input(
            "Any special request?",
            placeholder="e.g. quick meal, Italian tonight...",
        )
        suggest_clicked = st.button("✨ Suggest Recipes", use_container_width=True, type="primary")

        if suggest_clicked:
            prompt = "Suggest recipes from my pantry."
            if special_request.strip():
                prompt += f" Special request: {special_request.strip()}"
            with st.spinner("🧞 Cooking up some ideas..."):
                reply = chat(
                    user_input=prompt,
                    agent=st.session_state.agent,
                    thread_id=user_id,
                    user_id=user_id,
                )
            st.session_state.last_recipes = reply

        if st.session_state.get("last_recipes"):
            _render_recipe_cards(st.session_state.last_recipes)
            if st.button("🔄 Clear & start over", use_container_width=True):
                st.session_state.pop("last_recipes", None)
                st.rerun()
        else:
            st.markdown("""
            <div class="empty-state">
                <p style="font-size:2.5em; margin-bottom:6px;">🥗</p>
                <p style="font-size:0.95em;">
                    Pick some ingredients,<br>then hit <b>Suggest Recipes</b>!
                </p>
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# SHOPPING TAB
# ════════════════════════════════════════════════════════════
with tab_shop:
    try:
        shop_res = supabase.table("shopping_list").select("items").eq("user_id", user_id).execute()
        shop_items = list(shop_res.data[0]["items"]) if shop_res.data else []
    except Exception:
        shop_items = []

    with st.form("shop_add", clear_on_submit=True):
        new_shop = st.text_input("", placeholder="e.g. oat milk, paneer", label_visibility="collapsed")
        if st.form_submit_button("➕ Add to List", use_container_width=True) and new_shop.strip():
            new_shop_items = [i.strip().lower() for i in new_shop.split(",") if i.strip()]
            updated_shop = list(dict.fromkeys(shop_items + new_shop_items))
            supabase.table("shopping_list").upsert({"user_id": user_id, "items": updated_shop}).execute()
            st.rerun()

    if shop_items:
        st.caption(f"{len(shop_items)} item{'s' if len(shop_items) != 1 else ''}")
        for idx, item in enumerate(shop_items):
            if st.session_state.editing_shop == idx:
                edit_key = f"edit_shop_val_{idx}"
                if edit_key not in st.session_state:
                    st.session_state[edit_key] = item
                c1, c2, c3 = st.columns([4, 1, 1])
                with c1:
                    st.text_input("", key=edit_key, label_visibility="collapsed")
                with c2:
                    if st.button("✓", key=f"save_shop_{idx}"):
                        new_val = st.session_state[edit_key].strip().lower()
                        if new_val:
                            updated_shop = [new_val if j == idx else x for j, x in enumerate(shop_items)]
                            supabase.table("shopping_list").upsert({"user_id": user_id, "items": updated_shop}).execute()
                        st.session_state.editing_shop = None
                        st.rerun()
                with c3:
                    if st.button("✕", key=f"cancel_shop_{idx}"):
                        st.session_state.editing_shop = None
                        st.rerun()
            else:
                c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
                with c1:
                    st.markdown(f"• {item.strip().capitalize()}")
                with c2:
                    if st.button("✏️", key=f"edit_shop_{idx}"):
                        st.session_state.editing_shop = idx
                        st.session_state[f"edit_shop_val_{idx}"] = item
                        st.rerun()
                with c3:
                    if st.button("✓", key=f"bought_{idx}", help="Purchased — move to pantry"):
                        updated_shop = [x for j, x in enumerate(shop_items) if j != idx]
                        supabase.table("shopping_list").upsert({"user_id": user_id, "items": updated_shop}).execute()
                        pantry_res = supabase.table("pantry").select("ingredients").eq("user_id", user_id).execute()
                        pantry_items = list(pantry_res.data[0]["ingredients"]) if pantry_res.data else []
                        if item.strip().lower() not in pantry_items:
                            pantry_items.append(item.strip().lower())
                        supabase.table("pantry").upsert({"user_id": user_id, "ingredients": pantry_items}).execute()
                        st.rerun()
                with c4:
                    if st.button("✕", key=f"del_shop_{idx}"):
                        updated_shop = [x for j, x in enumerate(shop_items) if j != idx]
                        supabase.table("shopping_list").upsert({"user_id": user_id, "items": updated_shop}).execute()
                        st.rerun()
    else:
        st.info("Shopping list is empty.")

# ════════════════════════════════════════════════════════════
# PREFERENCES TAB
# ════════════════════════════════════════════════════════════
with tab_prefs:
    prefs_result = supabase.table("preferences").select("*").eq("user_id", user_id).execute()
    prefs = prefs_result.data[0] if prefs_result.data else {}

    # Account
    if user_picture:
        ac1, ac2 = st.columns([1, 4])
        with ac1:
            st.image(user_picture, width=48)
        with ac2:
            st.markdown(f"**{user_full_name}**")
            st.caption(user_id)
    else:
        st.markdown(f"**{user_full_name}**")
        st.caption(user_id)

    st.divider()

    # Spice level
    spice_options = ["", "low", "medium", "high"]
    current_spice = prefs.get("spice_level") or ""
    spice_idx = spice_options.index(current_spice) if current_spice in spice_options else 0
    new_spice = st.selectbox("🌶️ Spice level", spice_options, index=spice_idx,
                             format_func=lambda x: x.capitalize() if x else "Not set")
    if new_spice != current_spice:
        supabase.table("preferences").upsert({"user_id": user_id, "spice_level": new_spice}).execute()
        st.rerun()

    # Dislikes
    dislikes = prefs.get("dislikes") or []
    st.markdown("**🚫 Dislikes**")
    for idx, d in enumerate(dislikes):
        c1, c2 = st.columns([5, 1])
        with c1:
            st.markdown(f"• {d.capitalize()}")
        with c2:
            if st.button("✕", key=f"del_dislike_{idx}"):
                supabase.table("preferences").upsert({"user_id": user_id, "dislikes": [x for j, x in enumerate(dislikes) if j != idx]}).execute()
                st.rerun()
    with st.form("dislike_add", clear_on_submit=True):
        new_dislike = st.text_input("", placeholder="Add dislike...", label_visibility="collapsed")
        if st.form_submit_button("➕ Add Dislike", use_container_width=True) and new_dislike.strip():
            supabase.table("preferences").upsert({"user_id": user_id, "dislikes": dislikes + [new_dislike.strip().lower()]}).execute()
            st.rerun()

    # Cuisines
    cuisines = prefs.get("favorite_cuisines") or []
    st.markdown("**❤️ Favourite Cuisines**")
    for idx, cuisine in enumerate(cuisines):
        c1, c2 = st.columns([5, 1])
        with c1:
            st.markdown(f"• {cuisine.capitalize()}")
        with c2:
            if st.button("✕", key=f"del_cuisine_{idx}"):
                supabase.table("preferences").upsert({"user_id": user_id, "favorite_cuisines": [x for j, x in enumerate(cuisines) if j != idx]}).execute()
                st.rerun()
    with st.form("cuisine_add", clear_on_submit=True):
        new_cuisine = st.text_input("", placeholder="Add cuisine...", label_visibility="collapsed")
        if st.form_submit_button("➕ Add Cuisine", use_container_width=True) and new_cuisine.strip():
            supabase.table("preferences").upsert({"user_id": user_id, "favorite_cuisines": cuisines + [new_cuisine.strip().lower()]}).execute()
            st.rerun()

    st.divider()
    if st.button("🚪 Sign out", use_container_width=True):
        st.session_state.pop("user_info", None)
        st.session_state.pop("token", None)
        st.rerun()

    st.caption("Built with LangGraph · Groq · Supabase 🌱")
