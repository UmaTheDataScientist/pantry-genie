import streamlit as st
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
</style>
""", unsafe_allow_html=True)

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
            st.session_state.user_info = _decode_id_token(id_token)
            st.session_state.token = result["token"]
            st.rerun()
    st.stop()

# ── Logged-in view ─────────────────────────────────────────
user_info = st.session_state.user_info
user_name = (user_info.get("name") or "there").split()[0]
user_id = user_info.get("email", "")

st.title("🧞 PantryGenie")
st.markdown(f'<p class="subtitle">Hey {user_name}! Welcome back 🌱</p>', unsafe_allow_html=True)
st.divider()
st.info("You are signed in. More features coming soon!")

with st.sidebar:
    st.markdown(f"**{user_info.get('name', '')}**")
    st.caption(user_id)
    st.divider()
    if st.button("🚪 Sign out"):
        st.session_state.pop("user_info", None)
        st.session_state.pop("token", None)
        st.rerun()
