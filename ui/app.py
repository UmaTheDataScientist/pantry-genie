import streamlit as st
import sys
import os
import json
import base64
import re
from datetime import datetime, timedelta

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

# ── Config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="PantryGenie 🧞",
    page_icon="🧞",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main { max-width: 700px; margin: auto; }
    h1 { text-align: center; }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 0.9em;
        margin-top: -15px;
        margin-bottom: 20px;
    }
    .empty-state {
        text-align: center;
        color: #bbb;
        margin-top: 48px;
        margin-bottom: 48px;
    }
    .recipe-card {
        background: #f7faf7;
        border-radius: 14px;
        padding: 20px 24px 16px 24px;
        border: 1px solid #cfe8cf;
        margin-bottom: 18px;
    }
    .recipe-name {
        font-size: 1.25em;
        font-weight: 700;
        color: #1e5c1e;
        margin-bottom: 10px;
    }
    .recipe-field {
        margin: 6px 0;
        line-height: 1.55;
    }
    .recipe-label {
        font-weight: 600;
        color: #2e7d32;
    }
    .recipe-meta {
        display: flex;
        gap: 24px;
        margin-top: 10px;
        flex-wrap: wrap;
    }
    .recipe-watch a {
        color: #d32f2f;
        text-decoration: none;
        font-weight: 500;
    }
    .recipe-watch a:hover { text-decoration: underline; }
    .recipe-cooktime { color: #555; font-size: 0.92em; }
</style>
""", unsafe_allow_html=True)


def _extract_field(text: str, label: str) -> str:
    """Pull the value of a bold-labeled field out of markdown text."""
    m = re.search(rf'\*\*{label}:\*\*\s*(.+?)(?=\n\*\*|\Z)', text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _render_recipe_cards(text: str):
    """Parse recipe markdown and render each recipe as a styled card."""
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

# ── OAuth setup ────────────────────────────────────────────
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

# ── Cookie-based session (survives page refresh and server restarts) ──────────
import extra_streamlit_components as stx

_cm = stx.CookieManager(key="pg_cm")
_cookies = _cm.get_all()

# get_all() returns None while the JS component is initialising.
# It triggers an automatic rerun once ready — stop here so we never
# flash the login screen during that brief initialisation window.
if _cookies is None:
    st.stop()

if "user_info" not in st.session_state:
    _raw = _cookies.get("pg_user_info", "")
    if _raw:
        try:
            st.session_state.user_info = json.loads(_raw)
        except Exception:
            pass

# ── Login gate ─────────────────────────────────────────────
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
            user_info = _decode_id_token(id_token)
            st.session_state.user_info = user_info
            st.session_state.token = result["token"]
            _cm.set("pg_user_info", json.dumps(user_info),
                    expires_at=datetime.now() + timedelta(days=30))
            st.rerun()
    st.stop()

# ── User identity ──────────────────────────────────────────
user_info = st.session_state.user_info
user_id = user_info.get("email", "")
user_name = (user_info.get("name") or "there").split()[0]
user_picture = user_info.get("picture", "") or ""
user_full_name = user_info.get("name") or ""

# ── Supabase ───────────────────────────────────────────────
@st.cache_resource
def get_supabase():
    from supabase import create_client
    return create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

supabase = get_supabase()

# ── Session state ──────────────────────────────────────────
if "agent" not in st.session_state:
    with st.spinner("🧞 Waking up PantryGenie..."):
        st.session_state.agent = build_agent()

# ── Header ─────────────────────────────────────────────────
st.title("🧞 PantryGenie")
st.markdown(f'<p class="subtitle">Hey {user_name}! Let\'s see what we can cook today 🌱</p>', unsafe_allow_html=True)
st.divider()

# ── Special request input ──────────────────────────────────
special_request = st.text_input(
    "Any special request?",
    placeholder="e.g. quick meal, Italian tonight, high protein, light lunch...",
)

# ── Suggest button ─────────────────────────────────────────
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    suggest_clicked = st.button("✨ Suggest Recipes", use_container_width=True, type="primary")

# ── Run agent on click ─────────────────────────────────────
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

# ── Recipe display ─────────────────────────────────────────
if st.session_state.get("last_recipes"):
    st.divider()
    st.subheader("🍽️ Your Recipe Suggestions")
    _render_recipe_cards(st.session_state.last_recipes)
else:
    st.markdown("""
    <div class="empty-state">
        <p style="font-size:3em; margin-bottom:8px;">🥗</p>
        <p style="font-size:1.05em;">
            Add ingredients to your pantry in the sidebar,<br>
            then hit <b>Suggest Recipes</b>!
        </p>
    </div>
    """, unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    if user_picture:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(user_picture, width=48)
        with col2:
            st.markdown(f"**{user_full_name}**")
            st.caption(user_id)
    else:
        st.markdown(f"**{user_full_name}**")
        st.caption(user_id)

    st.divider()

    st.header("🥕 Your Pantry")
    pantry_result = supabase.table("pantry").select("ingredients").eq("user_id", user_id).execute()
    ingredients = list(pantry_result.data[0]["ingredients"]) if pantry_result.data else []

    with st.form("pantry_add", clear_on_submit=True):
        new_item = st.text_input("", placeholder="e.g. tofu, lentils", label_visibility="collapsed")
        if st.form_submit_button("➕ Add to Pantry", use_container_width=True) and new_item.strip():
            new_items = [i.strip().lower() for i in new_item.split(",") if i.strip()]
            updated = list(dict.fromkeys(ingredients + new_items))
            supabase.table("pantry").upsert({"user_id": user_id, "ingredients": updated}).execute()
            st.rerun()

    if ingredients:
        for idx, item in enumerate(ingredients):
            c1, c2, c3 = st.columns([4, 1, 1])
            with c1:
                st.markdown(f"• {item.strip().capitalize()}")
            with c2:
                if st.button("🛒", key=f"shop_pantry_{idx}", help="Move to shopping list"):
                    updated_pantry = [x for j, x in enumerate(ingredients) if j != idx]
                    supabase.table("pantry").upsert({"user_id": user_id, "ingredients": updated_pantry}).execute()
                    shop_res = supabase.table("shopping_list").select("items").eq("user_id", user_id).execute()
                    shop_items = list(shop_res.data[0]["items"]) if shop_res.data else []
                    if item.strip().lower() not in shop_items:
                        shop_items.append(item.strip().lower())
                    supabase.table("shopping_list").upsert({"user_id": user_id, "items": shop_items}).execute()
                    st.rerun()
            with c3:
                if st.button("✕", key=f"del_pantry_{idx}"):
                    updated = [x for j, x in enumerate(ingredients) if j != idx]
                    supabase.table("pantry").upsert({"user_id": user_id, "ingredients": updated}).execute()
                    st.rerun()
    else:
        st.info("Pantry is empty.")

    st.divider()

    st.header("💛 Your Preferences")
    prefs_result = supabase.table("preferences").select("*").eq("user_id", user_id).execute()
    prefs = prefs_result.data[0] if prefs_result.data else {}

    spice_options = ["", "low", "medium", "high"]
    current_spice = prefs.get("spice_level") or ""
    spice_idx = spice_options.index(current_spice) if current_spice in spice_options else 0
    new_spice = st.selectbox("🌶️ Spice level", spice_options, index=spice_idx,
                             format_func=lambda x: x.capitalize() if x else "Not set")
    if new_spice != current_spice:
        supabase.table("preferences").upsert({"user_id": user_id, "spice_level": new_spice}).execute()
        st.rerun()

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

    cuisines = prefs.get("favorite_cuisines") or []
    st.markdown("**❤️ Cuisines**")
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

    st.header("🛒 Shopping List")
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
        for idx, item in enumerate(shop_items):
            c1, c2, c3 = st.columns([4, 1, 1])
            with c1:
                st.markdown(f"• {item.strip().capitalize()}")
            with c2:
                if st.button("✓", key=f"bought_{idx}", help="Purchased — move to pantry"):
                    updated_shop = [x for j, x in enumerate(shop_items) if j != idx]
                    supabase.table("shopping_list").upsert({"user_id": user_id, "items": updated_shop}).execute()
                    pantry_res = supabase.table("pantry").select("ingredients").eq("user_id", user_id).execute()
                    pantry_items = list(pantry_res.data[0]["ingredients"]) if pantry_res.data else []
                    if item.strip().lower() not in pantry_items:
                        pantry_items.append(item.strip().lower())
                    supabase.table("pantry").upsert({"user_id": user_id, "ingredients": pantry_items}).execute()
                    st.rerun()
            with c3:
                if st.button("✕", key=f"del_shop_{idx}"):
                    updated_shop = [x for j, x in enumerate(shop_items) if j != idx]
                    supabase.table("shopping_list").upsert({"user_id": user_id, "items": updated_shop}).execute()
                    st.rerun()
    else:
        st.info("Shopping list is empty.")

    st.divider()

    if st.button("🔄 New Suggestions", help="Clear current recipes"):
        st.session_state.pop("last_recipes", None)
        st.rerun()

    if st.button("🚪 Sign out"):
        _cm.delete("pg_user_info")
        st.session_state.pop("user_info", None)
        st.session_state.pop("token", None)
        st.rerun()

    st.divider()
    st.caption("Built with LangGraph + Groq + Supabase")
    st.caption("🌱 Vegetarian recipes")
