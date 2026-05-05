import streamlit as st
import uuid
import sys
import os

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

# ── Auth ───────────────────────────────────────────────────
if not st.user.is_logged_in:
    st.title("🧞 PantryGenie")
    st.markdown('<p class="subtitle">Your personal vegan recipe assistant 🌱</p>', unsafe_allow_html=True)
    st.divider()
    _, col, _ = st.columns([1, 2, 1])
    with col:
        if st.button("Sign in with Google 🔐", use_container_width=True):
            st.login("google")
    st.stop()

# ── User identity ──────────────────────────────────────────
user_id = st.user.email
user_name = (st.user.name or "").split()[0] or "there"
user_picture = getattr(st.user, "picture", "")

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
        st.markdown(f"Hey {user_name}! 👋 I'm PantryGenie. Tell me what ingredients you have and I'll suggest some delicious vegan recipes. What's in your pantry today?")

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
            st.markdown(f"**{st.user.name or ''}**")
            st.caption(user_id)
    else:
        st.markdown(f"**{st.user.name or ''}**")
        st.caption(user_id)

    st.divider()

    st.header("🥕 Your Pantry")
    pantry_result = supabase.table("pantry").select("ingredients").eq("user_id", user_id).execute()
    ingredients = pantry_result.data[0]["ingredients"] if pantry_result.data else []
    if ingredients:
        for item in ingredients:
            st.markdown(f"• {item.strip().capitalize()}")
    else:
        st.info("Pantry is empty.")

    st.divider()

    st.header("💛 Your Preferences")
    prefs_result = supabase.table("preferences").select("*").eq("user_id", user_id).execute()
    prefs = prefs_result.data[0] if prefs_result.data else {}
    if prefs.get("spice_level"):
        st.write(f"🌶️ Spice: **{prefs['spice_level']}**")
    if prefs.get("dislikes"):
        st.write(f"🚫 Dislikes: **{', '.join(prefs['dislikes'])}**")
    if prefs.get("favorite_cuisines"):
        st.write(f"❤️ Cuisines: **{', '.join(prefs['favorite_cuisines'])}**")
    if not any([prefs.get("spice_level"), prefs.get("dislikes"), prefs.get("favorite_cuisines")]):
        st.info("No preferences saved yet.")

    st.divider()

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    if st.button("🚪 Sign out"):
        st.logout()

    st.divider()
    st.caption("Built with LangGraph + Groq + Supabase")
    st.caption("🌱 Vegan recipes only")
