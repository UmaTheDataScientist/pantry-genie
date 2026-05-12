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

# ── Option lists (first 12 shown immediately; rest behind expander) ─────────
PANTRY_PRIMARY = [
    "Garlic", "Onion", "Tomatoes", "Rice", "Pasta", "Chickpeas",
    "Lentils", "Spinach", "Bell peppers", "Mushrooms", "Eggs", "Tofu",
]
PANTRY_MORE = [
    "Olive oil", "Broccoli", "Potatoes", "Carrots", "Coconut milk",
    "Soy sauce", "Black beans", "Sweet potato", "Tempeh", "Paneer",
    "Butter", "Yogurt", "Quinoa", "Oats", "Cauliflower", "Zucchini",
    "Eggplant", "Kidney beans", "Vegetable broth", "Ginger", "Lemon",
    "Cumin", "Turmeric", "Walnuts", "Almonds", "Corn", "Peas",
]

CUISINE_PRIMARY = [
    "Italian", "Indian", "Mexican", "Chinese", "Thai", "Japanese",
    "Mediterranean", "American", "Korean", "Vietnamese", "Greek", "French",
]
CUISINE_MORE = ["Middle Eastern", "Spanish", "Lebanese", "Ethiopian"]

EQUIP_PRIMARY = [
    "Stovetop", "Oven", "Instant Pot", "Air Fryer", "Microwave",
    "Blender", "Wok", "Slow Cooker", "Rice Cooker", "Grill / BBQ",
    "Food Processor", "Steamer",
]
EQUIP_MORE = ["Cast Iron Pan", "Stand Mixer"]

WIZARD = [
    {
        "screen": "wizard_0",
        "emoji": "🥕",
        "title": "What's in your pantry?",
        "hint": "Tap to add · tap again to remove",
        "primary": PANTRY_PRIMARY,
        "more": PANTRY_MORE,
        "key": "ingredients",
        "placeholder": "Type an ingredient, e.g. avocado...",
    },
    {
        "screen": "wizard_1",
        "emoji": "🍜",
        "title": "Which cuisines do you love?",
        "hint": "Pick all that apply",
        "primary": CUISINE_PRIMARY,
        "more": CUISINE_MORE,
        "key": "cuisines",
        "placeholder": "Another cuisine...",
    },
    {
        "screen": "wizard_2",
        "emoji": "🍳",
        "title": "What equipment do you have?",
        "hint": "We'll tailor recipes to what you own",
        "primary": EQUIP_PRIMARY,
        "more": EQUIP_MORE,
        "key": "equipment",
        "placeholder": "Other equipment...",
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
    padding-top: 0.5rem !important;
    padding-left: 0.75rem !important;
    padding-right: 0.75rem !important;
    max-width: 640px !important;
  }
  @media (max-width: 640px) {
    .block-container { padding-left: 0.4rem !important; padding-right: 0.4rem !important; }
    input, textarea, select { font-size: 16px !important; }
  }

  /* ── Main area buttons: 44px tap targets (Apple HIG) ── */
  div[data-testid="stButton"] > button {
    min-height: 44px !important;
    border-radius: 10px !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
  }

  /* ── Sidebar buttons: smaller so they don't crowd the text ── */
  section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
    min-height: 26px !important;
    min-width: 26px !important;
    max-height: 26px !important;
    padding: 0 5px !important;
    font-size: 0.75rem !important;
    border-radius: 5px !important;
    font-weight: 400 !important;
    line-height: 1 !important;
  }

  /* ── Inputs ── */
  div[data-testid="stTextInput"] input {
    min-height: 44px !important;
    font-size: 16px !important;
    border-radius: 10px !important;
  }

  /* ── Column gaps ── */
  div[data-testid="stHorizontalBlock"] { gap: 5px !important; }
  section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] { gap: 3px !important; }

  /* ── Wizard header ── */
  .wiz-hdr { text-align: center; padding: 0 0 0.5rem; }
  .wiz-emoji { font-size: 2.4rem; display: block; line-height: 1.1; margin-bottom: 3px; }
  .wiz-title { font-size: 1.2rem; font-weight: 700; margin: 0 0 2px; }
  .wiz-hint  { font-size: 0.80rem; color: #999; margin: 0 0 4px; }
  .wiz-count { font-size: 0.82rem; color: #2e7d32; font-weight: 600; margin: 0; }

  /* ── Progress dots ── */
  .prog { display: flex; justify-content: center; gap: 7px; margin-bottom: 10px; padding-top: 4px; }
  .pdot { width: 8px; height: 8px; border-radius: 50%; background: #ddd; display: inline-block; }
  .pdot.done { background: #a5d6a7; }
  .pdot.now  { background: #2e7d32; width: 22px; border-radius: 4px; }

  /* ── Sidebar item rows ── */
  .sb-row { font-size: 0.83rem; line-height: 1.6; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

  /* ── Recipe screen ── */
  .recipe-hdr { text-align: center; padding: 0.4rem 0 0.8rem; }
  .empty-hint { text-align: center; color: #ccc; padding: 2rem 0; }
</style>
""", unsafe_allow_html=True)

# ── OAuth ────────────────────────────────────────────────────────────────────
from streamlit_oauth import OAuth2Component

def _secret(key):
    try:
        return os.getenv(key) or st.secrets.get(key) or ""
    except Exception:
        return os.getenv(key) or ""

_client_id = _secret("GOOGLE_CLIENT_ID")
if not _client_id:
    st.error("Google OAuth not configured. Add GOOGLE_CLIENT_ID to Streamlit secrets.")
    st.stop()

oauth2 = OAuth2Component(
    client_id=_client_id,
    client_secret=_secret("GOOGLE_CLIENT_SECRET"),
    authorize_endpoint="https://accounts.google.com/o/oauth2/auth",
    token_endpoint="https://oauth2.googleapis.com/token",
    refresh_token_endpoint="https://oauth2.googleapis.com/token",
)

def _decode_token(t: str) -> dict:
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
    st.markdown('<h1 style="text-align:center;margin:0">PantryGenie</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align:center;color:#888;margin-bottom:2rem">'
        'Your vegetarian recipe assistant 🌱</p>',
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
            st.session_state.user_info = _decode_token(tok)
            st.session_state.token = result["token"]
            st.rerun()
    st.stop()

# ── Identity ─────────────────────────────────────────────────────────────────
user_info    = st.session_state.user_info
user_id      = user_info.get("email", "")
user_name    = (user_info.get("name") or "there").split()[0]
user_picture = user_info.get("picture", "") or ""

# ── Supabase ──────────────────────────────────────────────────────────────────
@st.cache_resource
def get_supabase():
    from supabase import create_client
    return create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

supabase = get_supabase()

# ── Load data once per session ────────────────────────────────────────────────
if "data_loaded" not in st.session_state:
    _p  = supabase.table("pantry").select("ingredients").eq("user_id", user_id).execute()
    _pr = supabase.table("preferences").select("*").eq("user_id", user_id).execute()
    _s  = supabase.table("shopping_list").select("items").eq("user_id", user_id).execute()
    _pd = _pr.data[0] if _pr.data else {}
    st.session_state.ingredients  = list(_p.data[0]["ingredients"]) if _p.data else []
    st.session_state.cuisines     = _pd.get("favorite_cuisines") or []
    st.session_state.equipment    = _pd.get("equipment") or []
    st.session_state.spice_level  = _pd.get("spice_level") or ""
    st.session_state.dislikes     = _pd.get("dislikes") or []
    st.session_state.shop_items   = list(_s.data[0]["items"]) if _s.data else []
    st.session_state.data_loaded  = True

# ── Agent ─────────────────────────────────────────────────────────────────────
if "agent" not in st.session_state:
    with st.spinner("🧞 Waking up..."):
        st.session_state.agent = build_agent()

# ── Screen: wizard always shows on startup so questions are always asked ──────
if "screen" not in st.session_state:
    st.session_state.screen = "wizard_0"

# ── Save helpers ──────────────────────────────────────────────────────────────
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
    cur = st.session_state[state_key]
    if low in [x.lower() for x in cur]:
        st.session_state[state_key] = [x for x in cur if x.lower() != low]
    else:
        st.session_state[state_key] = cur + [value.lower()]
    save_fn()

# ── Chip grid: 2 columns, togglable, primary = selected ──────────────────────
def chip_grid(options: list, state_key: str, selected_low: set, save_fn, prefix: str):
    cols = st.columns(2)
    for i, opt in enumerate(options):
        with cols[i % 2]:
            is_sel = opt.lower() in selected_low
            if st.button(
                opt,
                key=f"{prefix}_{i}",
                use_container_width=True,
                type="primary" if is_sel else "secondary",
            ):
                _toggle(state_key, opt, save_fn)
                st.rerun()

# ── Recipe parsing + modal ────────────────────────────────────────────────────
def _parse_recipes(text: str) -> list:
    recipes = []
    for sec in re.split(r"\n?---\n?", text):
        sec = sec.strip()
        if not sec or "##" not in sec:
            continue
        name_m = re.search(r"##\s+(?:🍲\s*)?(.+)", sec)
        if not name_m:
            continue

        def _field(label, _s=sec):
            m = re.search(rf"\*\*{label}:\*\*\s*(.+?)(?=\n\*\*|\Z)", _s, re.DOTALL)
            return m.group(1).strip() if m else ""

        watch = _field("Watch")
        yt_id = None
        if watch:
            yt_m = re.search(r"v=([A-Za-z0-9_-]{11})", watch)
            if yt_m:
                yt_id = yt_m.group(1)

        recipes.append({
            "name":        name_m.group(1).strip().lstrip("🍲").strip(),
            "ingredients": _field("Ingredients"),
            "directions":  _field("Directions"),
            "cook_time":   _field("Cook time"),
            "equipment":   _field("Equipment"),
            "watch":       watch,
            "yt_id":       yt_id,
        })
    return recipes


@st.dialog("📖 Recipe", width="large")
def _recipe_modal(recipe: dict):
    st.markdown(f"## {recipe['name']}")
    meta = []
    if recipe["cook_time"]: meta.append(f"⏱ {recipe['cook_time']}")
    if recipe["equipment"]: meta.append(f"🍳 {recipe['equipment']}")
    if meta:
        st.caption("  ·  ".join(meta))
    st.divider()

    if recipe["ingredients"]:
        st.markdown("### 🧺 Ingredients")
        for ing in [i.strip() for i in recipe["ingredients"].split(",") if i.strip()]:
            st.markdown(f"- {ing}")

    if recipe["directions"]:
        st.markdown("### 👩‍🍳 Directions")
        st.markdown(recipe["directions"])

    if recipe["yt_id"]:
        st.markdown("### ▶️ Watch it")
        # Responsive 16:9 YouTube embed
        st.markdown(
            f'<div style="position:relative;padding-bottom:56.25%;height:0;'
            f'overflow:hidden;border-radius:12px;margin-top:8px">'
            f'<iframe style="position:absolute;top:0;left:0;width:100%;height:100%;'
            f'border:none" src="https://www.youtube.com/embed/{recipe["yt_id"]}" '
            f'allowfullscreen></iframe></div>',
            unsafe_allow_html=True,
        )
    elif recipe["watch"]:
        st.markdown(f"▶️ {recipe['watch']}")


def render_recipe_cards(text: str):
    recipes = _parse_recipes(text)
    if not recipes:
        st.markdown(text)
        return
    for i, r in enumerate(recipes):
        with st.container(border=True):
            # Header row
            st.markdown(f"### 🍲 {r['name']}")
            meta = []
            if r["cook_time"]: meta.append(f"⏱ {r['cook_time']}")
            if r["equipment"]: meta.append(f"🍳 {r['equipment']}")
            if meta:
                st.caption("  ·  ".join(meta))
            # Teaser: first 3 ingredients
            if r["ingredients"]:
                tease = ", ".join(r["ingredients"].split(",")[:3]).strip()
                st.markdown(
                    f'<p style="color:#666;font-size:0.85rem;margin:4px 0 8px">'
                    f'{tease}…</p>',
                    unsafe_allow_html=True,
                )
            if st.button("Open Full Recipe →", key=f"open_{i}", use_container_width=True, type="primary"):
                _recipe_modal(r)

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR — persistent selections panel
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Profile
    if user_picture:
        pc, nc = st.columns([1, 3])
        with pc:  st.image(user_picture, width=36)
        with nc:
            st.markdown(f"**{user_name}**")
            uid_short = user_id[:24] + "…" if len(user_id) > 24 else user_id
            st.caption(uid_short)
    else:
        st.markdown(f"**{user_name}**")
    st.divider()

    # ── Pantry ──────────────────────────────────────────────────────────────
    ings = st.session_state.ingredients
    cnt  = f" ({len(ings)})" if ings else ""
    st.markdown(f"**🥕 Pantry{cnt}**")
    if ings:
        for i, item in enumerate(ings):
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(f"<p class='sb-row'>• {item.capitalize()}</p>", unsafe_allow_html=True)
            with c2:
                if st.button("✕", key=f"sbd_i{i}"):
                    st.session_state.ingredients = [x for j, x in enumerate(ings) if j != i]
                    _save_pantry(); st.rerun()
    else:
        st.caption("Empty — answer the first question to fill it!")

    # ── Cuisines ─────────────────────────────────────────────────────────────
    cuiss = st.session_state.cuisines
    cnt   = f" ({len(cuiss)})" if cuiss else ""
    st.markdown(f"**🍜 Cuisines{cnt}**")
    if cuiss:
        for i, c in enumerate(cuiss):
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(f"<p class='sb-row'>• {c.capitalize()}</p>", unsafe_allow_html=True)
            with c2:
                if st.button("✕", key=f"sbd_c{i}"):
                    st.session_state.cuisines = [x for j, x in enumerate(cuiss) if j != i]
                    _save_prefs(); st.rerun()
    else:
        st.caption("None yet")

    # ── Equipment ────────────────────────────────────────────────────────────
    equip = st.session_state.equipment
    cnt   = f" ({len(equip)})" if equip else ""
    st.markdown(f"**🍳 Equipment{cnt}**")
    if equip:
        for i, e in enumerate(equip):
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(f"<p class='sb-row'>• {e.capitalize()}</p>", unsafe_allow_html=True)
            with c2:
                if st.button("✕", key=f"sbd_e{i}"):
                    st.session_state.equipment = [x for j, x in enumerate(equip) if j != i]
                    _save_prefs(); st.rerun()
    else:
        st.caption("None yet")

    st.divider()

    # ── Shopping list ─────────────────────────────────────────────────────────
    shop = st.session_state.shop_items
    cnt  = f" ({len(shop)})" if shop else ""
    st.markdown(f"**🛒 Shopping{cnt}**")
    if shop:
        for i, si in enumerate(shop):
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(f"<p class='sb-row'>• {si.capitalize()}</p>", unsafe_allow_html=True)
            with c2:
                if st.button("✓", key=f"sbuy{i}", help="Mark bought → moves to pantry"):
                    st.session_state.shop_items = [x for j, x in enumerate(shop) if j != i]
                    if si.lower() not in [x.lower() for x in st.session_state.ingredients]:
                        st.session_state.ingredients = st.session_state.ingredients + [si.lower()]
                    _save_shop(); _save_pantry(); st.rerun()

    with st.form("sb_shop_add", clear_on_submit=True):
        ns = st.text_input("", placeholder="Add item...", label_visibility="collapsed")
        if st.form_submit_button("➕ Add", use_container_width=True) and ns.strip():
            new_items = [x.strip().lower() for x in ns.split(",") if x.strip()]
            st.session_state.shop_items = list(
                dict.fromkeys(st.session_state.shop_items + new_items)
            )
            _save_shop(); st.rerun()

    st.divider()

    if st.session_state.screen == "recipes":
        if st.button("✏️ Update pantry & prefs", use_container_width=True):
            st.session_state.screen = "wizard_0"; st.rerun()

    if st.button("🚪 Sign out", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    st.caption("LangGraph · Groq · Supabase 🌱")


# ════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ════════════════════════════════════════════════════════════════════════════
screen = st.session_state.screen

# ── WIZARD ────────────────────────────────────────────────────────────────────
if screen.startswith("wizard_"):
    step_idx = int(screen[-1])
    step     = WIZARD[step_idx]
    sk       = step["key"]
    save_fn  = _save_pantry if sk == "ingredients" else _save_prefs
    sel_low  = {x.lower() for x in st.session_state.get(sk, [])}

    # Progress dots
    dots = "".join(
        f'<span class="pdot {"done" if i < step_idx else ("now" if i == step_idx else "")}"></span>'
        for i in range(len(WIZARD))
    )
    st.markdown(f'<div class="prog">{dots}</div>', unsafe_allow_html=True)

    # Header
    sel_count  = len([x for x in st.session_state.get(sk, []) if x])
    count_html = f'<p class="wiz-count">✓ {sel_count} selected</p>' if sel_count else ""
    st.markdown(
        f'<div class="wiz-hdr">'
        f'<span class="wiz-emoji">{step["emoji"]}</span>'
        f'<p class="wiz-title">{step["title"]}</p>'
        f'<p class="wiz-hint">{step["hint"]}</p>'
        f'{count_html}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Primary chips (12 options, 2 columns = 6 rows, fits on any phone)
    chip_grid(step["primary"], sk, sel_low, save_fn, f"p_{sk}")

    # More options expander
    if step["more"]:
        with st.expander(f"More options ({len(step['more'])})"):
            chip_grid(step["more"], sk, sel_low, save_fn, f"m_{sk}")

    # Custom add
    with st.form(f"ca_{sk}", clear_on_submit=True):
        custom = st.text_input(
            "", placeholder=step["placeholder"], label_visibility="collapsed"
        )
        if st.form_submit_button("➕ Add custom", use_container_width=True) and custom.strip():
            items   = [x.strip().lower() for x in custom.split(",") if x.strip()]
            cur_low = [x.lower() for x in st.session_state[sk]]
            for item in items:
                if item not in cur_low:
                    st.session_state[sk] = st.session_state[sk] + [item]
            save_fn(); st.rerun()

    st.markdown("")

    # Navigation
    is_last = step_idx == len(WIZARD) - 1
    if step_idx == 0:
        c1, c2 = st.columns([3, 1])
        with c1:
            if st.button("Continue →", use_container_width=True, type="primary"):
                st.session_state.screen = "wizard_1"; st.rerun()
        with c2:
            if st.button("Skip all", use_container_width=True):
                st.session_state.screen = "recipes"
                st.session_state.auto_suggest = True
                st.rerun()
    else:
        c_back, c_next, c_skip = st.columns([1, 2, 1])
        with c_back:
            if st.button("← Back", use_container_width=True):
                st.session_state.screen = f"wizard_{step_idx - 1}"; st.rerun()
        with c_next:
            label    = "✨ Find Recipes!" if is_last else "Continue →"
            next_scr = "recipes" if is_last else f"wizard_{step_idx + 1}"
            if st.button(label, use_container_width=True, type="primary"):
                st.session_state.screen = next_scr
                if next_scr == "recipes":
                    st.session_state.auto_suggest = True  # fire immediately on arrival
                st.rerun()
        with c_skip:
            if not is_last:
                if st.button("Skip →", use_container_width=True):
                    st.session_state.screen = f"wizard_{step_idx + 1}"; st.rerun()

# ── RECIPE SCREEN ─────────────────────────────────────────────────────────────
elif screen == "recipes":
    ing_n = len(st.session_state.ingredients)
    cui_n = len(st.session_state.cuisines)

    # Auto-trigger if arriving from wizard "Find Recipes!" button
    if st.session_state.get("auto_suggest"):
        st.session_state.auto_suggest = False
        with st.spinner("🧞 Analyst → Chef → Nutritionist working on it..."):
            reply = chat(
                user_input="Suggest recipes from my pantry.",
                agent=st.session_state.agent,
                thread_id=user_id,
                user_id=user_id,
            )
        st.session_state.last_recipes = reply
        st.rerun()

    detail = f"🥕 {ing_n} item{'s' if ing_n != 1 else ''}"
    if cui_n:
        detail += f" · 🍜 {cui_n} cuisine{'s' if cui_n != 1 else ''}"

    st.markdown(
        f'<div class="recipe-hdr">'
        f'<span style="font-size:2.2rem">🧞</span>'
        f'<h2 style="margin:4px 0 2px">Hey {user_name}!</h2>'
        f'<p style="color:#888;font-size:0.85rem;margin:0">{detail}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if not ing_n:
        st.info("Pantry empty — open ☰ and tap **✏️ Update pantry & prefs**.", icon="🥕")

    if st.session_state.get("last_recipes"):
        # Split recipe cards from the wellness/nutrition section
        full = st.session_state.last_recipes
        w_start = re.search(r"\n(🥗|\*\*🥗|\*\*Nutrition)", full)
        recipe_part   = full[:w_start.start()].strip() if w_start else full
        wellness_part = full[w_start.start():].strip()  if w_start else ""

        render_recipe_cards(recipe_part)

        if wellness_part:
            with st.expander("🥗 Nutrition & Shopping Advice"):
                st.markdown(wellness_part)

        st.divider()
        special = st.text_input(
            "Refine or make a new request:",
            placeholder="e.g. something quicker, more protein, Italian…",
        )
        c1, c2 = st.columns([3, 1])
        with c1:
            if st.button("✨ New Suggestions", use_container_width=True, type="primary"):
                prompt = "Suggest recipes from my pantry."
                if special.strip():
                    prompt += f" Special request: {special.strip()}"
                with st.spinner("🧞 Working on it..."):
                    st.session_state.last_recipes = chat(
                        user_input=prompt,
                        agent=st.session_state.agent,
                        thread_id=user_id,
                        user_id=user_id,
                    )
                st.rerun()
        with c2:
            if st.button("🔄 Clear", use_container_width=True):
                del st.session_state["last_recipes"]
                st.rerun()
    else:
        special = st.text_input(
            "Any special request?",
            placeholder="e.g. quick 30-min meal, high protein, pasta…",
        )
        if st.button("✨ Suggest Recipes", use_container_width=True, type="primary"):
            prompt = "Suggest recipes from my pantry."
            if special.strip():
                prompt += f" Special request: {special.strip()}"
            with st.spinner("🧞 Analyst → Chef → Nutritionist working on it..."):
                st.session_state.last_recipes = chat(
                    user_input=prompt,
                    agent=st.session_state.agent,
                    thread_id=user_id,
                    user_id=user_id,
                )
            st.rerun()

        st.markdown(
            '<div class="empty-hint">'
            '<p style="font-size:2.2em;margin-bottom:6px">🥗</p>'
            '<p>Your 3-agent team is ready.<br>'
            '<small>Tap <b>Suggest Recipes</b> or finish the wizard.</small></p>'
            '</div>',
            unsafe_allow_html=True,
        )
