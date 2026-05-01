import streamlit as st
import uuid
import sys
import os

# ── Path fix for Streamlit Cloud ───────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pantry_genie.agent import build_agent, chat

# ── Config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="PantryGenie 🧞",
    page_icon="🧞",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS (mobile friendly) ───────────────────────────
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

# ── Header ─────────────────────────────────────────────────
st.title("🧞 PantryGenie")
st.markdown('<p class="subtitle">Tell me what\'s in your pantry — I\'ll grant your recipe wish 🌱</p>',
            unsafe_allow_html=True)
st.divider()

# ── Session state ──────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "agent" not in st.session_state:
    with st.spinner("🧞 Waking up PantryGenie..."):
        st.session_state.agent = build_agent()

# ── Set thread-local for user isolation ────────────────────
from pantry_genie import tools as t
t._thread_local.thread_id = st.session_state.thread_id

# ── Starter message ────────────────────────────────────────
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("Hey there! 👋 I'm PantryGenie. Tell me what ingredients you have and I'll suggest some delicious vegan recipes. What's in your pantry today?")

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
                thread_id=st.session_state.thread_id
            )
        st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.header("🥕 Your Pantry")
    try:
        from pantry_genie.tools import get_pantry_contents
        pantry = get_pantry_contents.invoke({})
        st.info(pantry)
    except:
        st.info("Start chatting to update your pantry!")

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        t._thread_local.thread_id = st.session_state.thread_id
        st.rerun()

    st.divider()
    st.caption("Built with LangGraph + Groq + Pinecone")
    st.caption("🌱 Vegan recipes only")