import streamlit as st
import sys
import os
import json
import base64
import re

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

# ── Constants ──────────────────────────────────────────────────────────────
PANTRY_OPTIONS = [
    "Olive oil", "Garlic", "Onion", "Tomatoes", "Rice", "Pasta",
    "Chickpeas", "Lentils", "Tofu", "Spinach", "Potatoes", "Carrots",
    "Bell peppers", "Mushrooms", "Broccoli", "Eggs",
    "Coconut milk", "Soy sauce", "Black beans", "Sweet potato",
    "Tempeh", "Paneer", "Butter", "Yogurt", "Quinoa", "Oats",
    "Cauliflower", "Zucchini", "Eggplant", "Kidney beans",
    "Vegetable broth", "Ginger", "Lemon", "Cumin", "Turmeric",
    "Walnuts", "Almonds", "Corn", "Peas",
]

CUISINE_OPTIONS = [
    "Italian", "Indian", "Mexican", "Chinese", "Thai", "Japanese",
    "Mediterranean", "American", "Korean", "Vietnamese",
    "Greek", "Middle Eastern", "French", "Spanish", "Lebanese", "Ethiopian",
]

EQUIPMENT_OPTIONS = [
    "Stovetop", "Oven", "Instant Pot", "Air Fryer", "Microwave",
    "Blender", "Wok", "Slow Cooker", "Rice Cooker", "Grill / BBQ",
    "Food Processor", "Steamer", "Cast Iron Pan", "Stand Mixer",
]

WIZARD = [
    {
        "screen": "wizard_0",
        "emoji": "🥕",
        "title": "What's in your pantry?",
        "hint": "Tap to add — tap again to remove",
        "options": PANTRY_OPTIONS,
        "key": "ingredients",
        "placeholder": "Add custom ingredient...",
    },
    {
        "screen": "wizard_1",
        "emoji": "🍜",
        "title": "What cuisines do you love?",
        "hint": "Pick all that sound delicious",
        "options": CUISINE_OPTIONS,
        "key": "cuisines",
        "placeholder": "Add another cuisine...",
    },
    {
        "screen": "wizard_2",
        "emoji": "🍳",
        "title": "What equipment do you have?",
        "hint": "We'll tailor recipes to what you own",
        "options": EQUIPMENT_OPTIONS,
        "key": "equipment",
        "placeholder": "Add other equipment...",
    },
]

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PantryGenie 🧞",
    page_icon="🧞",
    layout="centered",
    initial_sidebar_state="auto",
)

# ── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Layout ── */
  .block-container {
    padding-top: 0.75rem !important;
    padding-left: 0.75rem !important;
    padding-right: 0.75rem !important;
    max-width: 640px !important;
  }

  /* ── Mobile ── */
  @media (max-width: 640px) {
    .block-container {
      padding-left: 0.4rem !important;
      padding-right: 0.4rem !important;
    }
    /* Prevent iOS zoom on input focus */
    input, textarea, select { font-size: 16px !important; }
    button[data-baseweb="tab"] { font-size: 0.75rem !important; padding: 5px 5px !important; }
  }

  /* ── All buttons: 44px minimum tap target ── */
  div[data-testid="stButton"] > button {
    min-height: 44px !important;
    border-radius: 10px !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
  }

  /* ── Inputs ── */
  div[data-testid="stTextInput"] input {
    min-height: 44px !important;
    font-size: 16px !important;
    border-radius: 10px !important;
  }

  /* ── Reduce column gap ── */
  div[data-testid="stHorizontalBlock"] { gap: 5px !important; }

  /* ── Wizard step header ── */
  .step-hdr {
    text-align: center;
    padding: 0.25rem 0 0.6rem;
  }
  .step-hdr .big-emoji { font-size: 2.6rem; display: block; margin-bottom: 4px; }
  .step-hdr h2 { font-size: 1.3rem !important; font-weight: 700; margin: 2px 0; }
  .step-hdr .hint { font-size: 0.82rem; color: #888; margin: 0; }

  /* ── Progress dots ── */
  .prog-dots { display: flex; justify-content: center; gap: 8px; margin-bottom: 12px; }
  .pdot { width: 9px; height: 9px; border-radius: 50%; background: #ddd; display: inline-block; }
  .pdot.done { background: #a5d6a7; }
  .pdot.active { background: #2e7d32; width: 22px; border-radius: 5px; }

  /* ── Count badge on chips ── */
  .sel-badge {
    display: inline-block;
    background: #e8f5e9;
    color: #2e7d32;
    border-radius: 20px;
    padding: 1px 10px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-left: 6px;
  }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] .block-container {
    padding: 0.5rem 0.6rem !important;
    max-width: none !important;
  }
  section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
    min-height: 32px !important;
    font-size: 0.78rem !important;
    padding: 0 8px !important;
    border-radius: 6px !important;
  }
  .sb-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 2px 0;
    font-size: 0.85rem;
  }

  /* ── Recipe screen ── */
  .recipe-welcome {
    text-align: center;
    padding: 0.5rem 0 1rem;
  }
  .empty-state { text-align: center; color: #bbb; padding: 28px 0; }
</style>
""", unsafe_allow_html=True)

# ── OAuth setup ──────────────────────────────────────────────────────────────
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

def _decode_id_token(t: str) -> dict:
    p = t.split(".")[1]
    p += "=" * (4 - len(p) % 4)
    return json.loads(base64.urlsafe_b64decode(p))

# ── Login gate ───────────────────────────────────────────────────────────────
if "user_info" not in st.session_state:
    st.markdown(
        '<div style="text-align:center;padding-top:3rem">'
        '<span style="font-size:3.5rem">🧞</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<h1 style="text-align:center;margin-top:0">PantryGenie</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align:center;color:#888;margin-bottom:2.5rem">'
        'Your personal vegetarian recipe assistant 🌱</p>',
        unsafe_allow_html=True,
    )
    _, col, _ = st.columns([1, 4, 1])
    with col:
        result = oauth2.authorize_button(
            name="Sign in with Google",
            redirect_uri=_secret("REDIRECT_URI") or "https://pantry-genie.streamlit.app",
            scope="openid email profile",
            key="google_login",
            use_container_width=True,
        )
    if result and "token" in result:
        tok = result["token"].get("id_token", "")
        if tok:
            st.session_state.user_info = _decode_id_token(tok)
            st.session_state.token = result["token"]
            st.rerun()
    st.stop()

# ── Identity ─────────────────────────────────────────────────────────────────
user_info = st.session_state.user_info
user_id = user_info.get("email", "")
user_name = (user_info.get("name") or "there").split()[0]
user_picture = user_info.get("picture", "") or ""
user_full_name = user_info.get("name") or ""

# ── Supabase ──────────────────────────────────────────────────────────────────
@st.cache_resource
def get_supabase():
    from supabase import create_client
    return create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

supabase = get_supabase()

# ── Load data once per session ────────────────────────────────────────────────
if "data_loaded" not in st.session_state:
    _p = supabase.table("pantry").select("ingredients").eq("user_id", user_id).execute()
    _pr = supabase.table("preferences").select("*").eq("user_id", user_id).execute()
    _s = supabase.table("shopping_list").select("items").eq("user_id", user_id).execute()
    _pd = _pr.data[0] if _pr.data else {}
    st.session_state.ingredients = list(_p.data[0]["ingredients"]) if _p.data else []
    st.session_state.cuisines = _pd.get("favorite_cuisines") or []
    st.session_state.equipment = _pd.get("equipment") or []
    st.session_state.spice_level = _pd.get("spice_level") or ""
    st.session_state.dislikes = _pd.get("dislikes") or []
    st.session_state.shop_items = list(_s.data[0]["items"]) if _s.data else []
    st.session_state.data_loaded = True

# ── Build agent ───────────────────────────────────────────────────────────────
if "agent" not in st.session_state:
    with st.spinner("🧞 Waking up..."):
        st.session_state.agent = build_agent()

# ── Initial screen ────────────────────────────────────────────────────────────
if "screen" not in st.session_state:
    has_setup = bool(
        st.session_state.ingredients
        or st.session_state.cuisines
        or st.session_state.equipment
    )
    st.session_state.screen = "recipes" if has_setup else "wizard_0"

# ── Supabase write helpers ─────────────────────────────────────────────────────
def _save_pantry():
    supabase.table("pantry").upsert(
        {"user_id": user_id, "ingredients": st.session_state.ingredients}
    ).execute()

def _save_prefs():
    payload = {
        "user_id": user_id,
        "favorite_cuisines": st.session_state.cuisines,
        "spice_level": st.session_state.spice_level,
        "dislikes": st.session_state.dislikes,
    }
    try:
        payload["equipment"] = st.session_state.equipment
        supabase.table("preferences").upsert(payload).execute()
    except Exception:
        payload.pop("equipment", None)
        supabase.table("preferences").upsert(payload).execute()

def _save_shop():
    supabase.table("shopping_list").upsert(
        {"user_id": user_id, "items": st.session_state.shop_items}
    ).execute()

def _toggle(state_key: str, value: str, save_fn):
    low = value.lower()
    current = st.session_state[state_key]
    lows = [x.lower() for x in current]
    if low in lows:
        st.session_state[state_key] = [x for x in current if x.lower() != low]
    else:
        st.session_state[state_key] = current + [value.lower()]
    save_fn()

# ── Chip grid helper ───────────────────────────────────────────────────────────
def render_chip_grid(opts, state_key, selected_low, save_fn, prefix, cols=3):
    columns = st.columns(cols)
    for i, opt in enumerate(opts):
        with columns[i % cols]:
            is_sel = opt.lower() in selected_low
            if st.button(
                opt,
                key=f"{prefix}_{i}",
                use_container_width=True,
                type="primary" if is_sel else "secondary",
            ):
                _toggle(state_key, opt, save_fn)
                st.rerun()

# ── Recipe card renderer ───────────────────────────────────────────────────────
def _render_recipe_cards(text: str):
    sections = re.split(r"\n?---\n?", text)
    rendered = 0
    for section in sections:
        section = section.strip()
        if not section or "##" not in section:
            continue
        name_m = re.search(r"##\s+(.+)", section)
        name = name_m.group(1).strip() if name_m else "Recipe"

        def _f(label):
            m = re.search(rf"\*\*{label}:\*\*\s*(.+?)(?=\n\*\*|\Z)", section, re.DOTALL)
            return m.group(1).strip() if m else ""

        with st.container(border=True):
            st.markdown(f"### {name}")
            if ing := _f("Ingredients"):
                st.markdown(f"**Ingredients:** {ing}")
            if dirs := _f("Directions"):
                st.markdown(f"**Directions:** {dirs}")
            if t := _f("Cook time"):
                st.markdown(f"**Cook time:** {t}")
            if w := _f("Watch"):
                st.markdown(f"**Watch:** {w}")
        rendered += 1

    if not rendered:
        st.markdown(text)

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Persistent selections panel
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    if user_picture:
        pc, nc = st.columns([1, 3])
        with pc:
            st.image(user_picture, width=38)
        with nc:
            st.markdown(f"**{user_name}**")
            st.caption(user_id[:28] + "…" if len(user_id) > 28 else user_id)
    else:
        st.markdown(f"**{user_full_name}**")
    st.divider()

    # ── Pantry ──
    ings = st.session_state.ingredients
    cnt = f" ({len(ings)})" if ings else ""
    st.markdown(f"**🥕 Pantry{cnt}**")
    if ings:
        for i, item in enumerate(ings):
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(f"<div class='sb-item'>• {item.capitalize()}</div>", unsafe_allow_html=True)
            with c2:
                if st.button("✕", key=f"sb_di_{i}"):
                    st.session_state.ingredients = [x for j, x in enumerate(ings) if j != i]
                    _save_pantry()
                    st.rerun()
    else:
        st.caption("Empty — use wizard to add")

    # ── Cuisines ──
    cuiss = st.session_state.cuisines
    cnt = f" ({len(cuiss)})" if cuiss else ""
    st.markdown(f"**🍜 Cuisines{cnt}**")
    if cuiss:
        for i, c in enumerate(cuiss):
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(f"<div class='sb-item'>• {c.capitalize()}</div>", unsafe_allow_html=True)
            with c2:
                if st.button("✕", key=f"sb_dc_{i}"):
                    st.session_state.cuisines = [x for j, x in enumerate(cuiss) if j != i]
                    _save_prefs()
                    st.rerun()
    else:
        st.caption("None set")

    # ── Equipment ──
    equip = st.session_state.equipment
    cnt = f" ({len(equip)})" if equip else ""
    st.markdown(f"**🍳 Equipment{cnt}**")
    if equip:
        for i, e in enumerate(equip):
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(f"<div class='sb-item'>• {e.capitalize()}</div>", unsafe_allow_html=True)
            with c2:
                if st.button("✕", key=f"sb_de_{i}"):
                    st.session_state.equipment = [x for j, x in enumerate(equip) if j != i]
                    _save_prefs()
                    st.rerun()
    else:
        st.caption("None set")

    st.divider()

    # ── Shopping list ──
    shop = st.session_state.shop_items
    cnt = f" ({len(shop)})" if shop else ""
    st.markdown(f"**🛒 Shopping{cnt}**")
    if shop:
        for i, si in enumerate(shop):
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(f"<div class='sb-item'>• {si.capitalize()}</div>", unsafe_allow_html=True)
            with c2:
                if st.button("✓", key=f"sb_buy_{i}", help="Mark bought → pantry"):
                    st.session_state.shop_items = [x for j, x in enumerate(shop) if j != i]
                    if si.lower() not in [x.lower() for x in st.session_state.ingredients]:
                        st.session_state.ingredients = st.session_state.ingredients + [si.lower()]
                    _save_shop()
                    _save_pantry()
                    st.rerun()

    with st.form("sb_shop_add", clear_on_submit=True):
        ns = st.text_input("", placeholder="Add item...", label_visibility="collapsed")
        if st.form_submit_button("➕ Add", use_container_width=True) and ns.strip():
            new_items = [x.strip().lower() for x in ns.split(",") if x.strip()]
            st.session_state.shop_items = list(dict.fromkeys(st.session_state.shop_items + new_items))
            _save_shop()
            st.rerun()

    st.divider()

    if st.session_state.screen == "recipes":
        if st.button("✏️ Update pantry & prefs", use_container_width=True):
            st.session_state.screen = "wizard_0"
            st.rerun()

    if st.button("🚪 Sign out", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    st.caption("Built with LangGraph · Groq · Supabase 🌱")


# ════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ════════════════════════════════════════════════════════════════════════════
screen = st.session_state.screen

# ── Wizard ───────────────────────────────────────────────────────────────────
if screen.startswith("wizard_"):
    step_idx = int(screen.split("_")[1])
    step = WIZARD[step_idx]
    state_key = step["key"]
    save_fn = _save_pantry if state_key == "ingredients" else _save_prefs
    selected_low = set(x.lower() for x in st.session_state.get(state_key, []))

    # Progress bar
    dots_html = '<div class="prog-dots">'
    for i in range(len(WIZARD)):
        if i < step_idx:
            dots_html += '<span class="pdot done"></span>'
        elif i == step_idx:
            dots_html += '<span class="pdot active"></span>'
        else:
            dots_html += '<span class="pdot"></span>'
    dots_html += '</div>'
    st.markdown(dots_html, unsafe_allow_html=True)

    # Header
    sel_count = len([x for x in st.session_state.get(state_key, []) if x])
    badge = f'<span class="sel-badge">{sel_count} selected</span>' if sel_count else ""
    st.markdown(
        f'<div class="step-hdr">'
        f'<span class="big-emoji">{step["emoji"]}</span>'
        f'<h2>{step["title"]}{badge}</h2>'
        f'<p class="hint">{step["hint"]}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Chip grid — first 15 fit on one screen, rest in expander
    primary_opts = step["options"][:15]
    extra_opts = step["options"][15:]

    render_chip_grid(primary_opts, state_key, selected_low, save_fn, f"main_{state_key}")

    if extra_opts:
        with st.expander(f"More options ({len(extra_opts)})"):
            render_chip_grid(extra_opts, state_key, selected_low, save_fn, f"extra_{state_key}")

    # Custom add
    st.markdown("")
    with st.form(f"custom_add_{state_key}", clear_on_submit=True):
        custom = st.text_input(
            "", placeholder=step["placeholder"], label_visibility="collapsed"
        )
        if st.form_submit_button("➕ Add custom", use_container_width=True) and custom.strip():
            items = [x.strip().lower() for x in custom.split(",") if x.strip()]
            existing_low = [x.lower() for x in st.session_state[state_key]]
            for item in items:
                if item not in existing_low:
                    st.session_state[state_key] = st.session_state[state_key] + [item]
            save_fn()
            st.rerun()

    # Navigation
    st.markdown("")
    if step_idx == 0:
        nav_cols = st.columns([3, 1])
        with nav_cols[0]:
            if st.button("Continue →", use_container_width=True, type="primary"):
                st.session_state.screen = "wizard_1"
                st.rerun()
        with nav_cols[1]:
            if st.button("Skip", use_container_width=True):
                st.session_state.screen = "wizard_1"
                st.rerun()
    else:
        bc, nc = st.columns([1, 2])
        with bc:
            if st.button("← Back", use_container_width=True):
                st.session_state.screen = f"wizard_{step_idx - 1}"
                st.rerun()
        with nc:
            is_last = step_idx == len(WIZARD) - 1
            label = "✨ Find Recipes!" if is_last else "Continue →"
            next_screen = "recipes" if is_last else f"wizard_{step_idx + 1}"
            if st.button(label, use_container_width=True, type="primary"):
                st.session_state.screen = next_screen
                st.rerun()

# ── Recipe screen ─────────────────────────────────────────────────────────────
elif screen == "recipes":
    ing_count = len(st.session_state.ingredients)
    cui_count = len(st.session_state.cuisines)

    st.markdown(
        f'<div class="recipe-welcome">'
        f'<span style="font-size:2.2rem">🧞</span>'
        f'<h2 style="margin:4px 0">Hey {user_name}!</h2>'
        f'<p style="color:#888;font-size:0.88rem;margin:0">'
        f'{ing_count} ingredient{"s" if ing_count != 1 else ""}'
        + (f" · {cui_count} cuisine{'s' if cui_count != 1 else ''}" if cui_count else "")
        + f'</p></div>',
        unsafe_allow_html=True,
    )

    if not ing_count:
        st.warning("Your pantry is empty — tap **✏️ Update pantry & prefs** in the sidebar to add ingredients.", icon="🥕")

    special = st.text_input(
        "Any special request?",
        placeholder="e.g. quick 30-min meal, high protein, pasta tonight...",
    )

    if st.button("✨ Suggest Recipes", use_container_width=True, type="primary"):
        prompt = "Suggest recipes from my pantry."
        if special.strip():
            prompt += f" Special request: {special.strip()}"
        with st.spinner("🧞 Cooking up some ideas..."):
            reply = chat(
                user_input=prompt,
                agent=st.session_state.agent,
                thread_id=user_id,
                user_id=user_id,
            )
        st.session_state.last_recipes = reply

    if st.session_state.get("last_recipes"):
        st.divider()
        _render_recipe_cards(st.session_state.last_recipes)
        if st.button("🔄 Get new suggestions", use_container_width=True):
            st.session_state.pop("last_recipes", None)
            st.rerun()
    else:
        st.markdown(
            '<div class="empty-state">'
            '<p style="font-size:2.5em;margin-bottom:6px">🥗</p>'
            '<p>Hit <b>Suggest Recipes</b> to get started!<br>'
            '<small>Update your pantry anytime from the sidebar.</small></p>'
            "</div>",
            unsafe_allow_html=True,
        )
