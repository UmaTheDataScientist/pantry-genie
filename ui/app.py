import streamlit as st
import uuid
import sys
import os
import json
import base64

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
    .stChatMessage { border-radius: 12px; margin-bottom: 8px; }
    h1 { text-align: center; }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 0.9em;
        margin-top: -15px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ── Google OAuth ───────────────────────────────────────────
from streamlit_oauth import OAuth2Component
from streamlit_cookies_controller import CookieController

cookie_manager = CookieController()

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

# Restore session from cookie
# The cookie component needs one render cycle to report values back to Python.
# On the first render we force a silent rerun; on the second render the cookie is available.
if "user_info" not in st.session_state:
    try:
        _cookie = cookie_manager.get("pg_user")
    except Exception:
        _cookie = None
    if _cookie:
        try:
            st.session_state.user_info = json.loads(_cookie)
        except Exception:
            pass
    elif "cookie_loaded" not in st.session_state:
        st.session_state.cookie_loaded = True
        st.rerun()

if "user_info" not in st.session_state:
    st.title("🧞 PantryGenie")
    st.markdown('<p class="subtitle">Your personal vegetarian recipe assistant 🌱</p>', unsafe_allow_html=True)
    st.divider()
    _, col, _ = st.columns([1, 2, 1])
    with col:
        result = oauth2.authorize_button(
            name="Sign in with Google",
            redirect_uri=os.getenv("REDIRECT_URI", "https://pantry-genie.streamlit.app"),
            scope="openid email profile",
            key="google_login",
            extras_params={"prompt": "consent", "access_type": "offline"},
            use_container_width=True,
            icon="https://www.google.com/favicon.ico",
        )
    if result and "token" in result:
        _user = _decode_id_token(result["token"]["id_token"])
        cookie_manager.set("pg_user", json.dumps(_user), max_age=30*24*3600)
        st.session_state.user_info = _user
        st.rerun()
    st.stop()

# ── User identity ──────────────────────────────────────────
user_info = st.session_state.user_info
user_id = user_info["email"]
user_name = user_info.get("given_name") or user_info.get("name", "").split()[0] or "there"
user_picture = user_info.get("picture", "")

# ── Supabase ───────────────────────────────────────────────
@st.cache_resource
def get_supabase():
    from supabase import create_client
    return create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

supabase = get_supabase()

# ── Session state ──────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    with st.spinner("🧞 Waking up PantryGenie..."):
        st.session_state.agent = build_agent()

# ── Header ─────────────────────────────────────────────────
st.title("🧞 PantryGenie")
st.markdown(f'<p class="subtitle">Hey {user_name}! Tell me what\'s in your pantry — I\'ll grant your recipe wish 🌱</p>', unsafe_allow_html=True)
st.divider()

# ── Starter message ────────────────────────────────────────
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(f"Hey {user_name}! 👋 I'm PantryGenie. Tell me what ingredients you have and I'll suggest some delicious vegetarian recipes. What's in your pantry today?")

# ── Chat history ───────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ─────────────────────────────────────────────
if prompt := st.chat_input("e.g. I have chickpeas, spinach and coconut milk..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🧞 thinking..."):
            reply = chat(
                user_input=prompt,
                agent=st.session_state.agent,
                thread_id=st.session_state.thread_id,
                user_id=user_id,
            )
        st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    if user_picture:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(user_picture, width=48)
        with col2:
            st.markdown(f"**{user_info.get('name', '')}**")
            st.caption(user_id)
    else:
        st.markdown(f"**{user_info.get('name', '')}**")
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

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    if st.button("🚪 Sign out"):
        cookie_manager.delete("pg_user")
        del st.session_state.user_info
        st.rerun()

    st.divider()
    st.caption("Built with LangGraph + Groq + Supabase")
    st.caption("🌱 Vegetarian recipes")
