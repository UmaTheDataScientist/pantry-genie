"""
PantryGenie multi-agent system.

Architecture
────────────
Supervisor (Llama 3.3 70B, has memory)
  ├── Pantry Analyst    (Llama 3.1 8B)  — reads pantry + prefs, produces structured context
  ├── Recipe Architect  (Llama 3.3 70B) — creative recipe design + substitutions
  └── Wellness Advisor  (Llama 3.1 8B)  — nutrition check + smart shopping gaps

Sub-agents are exposed as tools to the supervisor so it can delegate selectively.
Only the supervisor has a memory checkpointer; sub-agents are stateless workers.
"""

import os
import re
import requests
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

from pantry_genie.tools import (
    ANALYST_TOOLS,
    RECIPE_TOOLS,
    WELLNESS_TOOLS,
    SUPERVISOR_TOOLS,
)

load_dotenv()

try:
    import streamlit as st
    for key, value in st.secrets.items():
        os.environ.setdefault(key, value)
except:
    pass

# ── Statsig (feature flags, optional) ────────────────────────────────────────
try:
    from statsig import statsig
    key = os.getenv("STATSIG_SERVER_KEY")
    if key:
        try:
            statsig.initialize_sync(key)
        except Exception:
            import asyncio
            asyncio.get_event_loop().run_until_complete(statsig.initialize(key))
except Exception:
    pass


# ── LLM builders ─────────────────────────────────────────────────────────────
def _llm(model: str = "llama-3.3-70b-versatile") -> ChatGroq:
    return ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model=model, temperature=0)

def _fast_llm() -> ChatGroq:
    """Smaller/faster model for agents that mostly call tools and format results."""
    return _llm("llama-3.1-8b-instant")


# ── System prompts ────────────────────────────────────────────────────────────
_ANALYST_PROMPT = """You are the Pantry Analyst for PantryGenie 🧞.
Your only job: read the user's pantry and preferences, then return a crisp structured report.

Steps (always follow this order):
1. Call get_pantry_contents
2. Call get_user_preferences
3. Return a structured report with these sections:
   • INGREDIENTS: full list grouped as Proteins / Grains & Carbs / Vegetables / Flavours & Oils
   • PREFERENCES: cuisines, spice level, dislikes, equipment
   • ANCHOR INGREDIENTS: the 3-5 most versatile items in the pantry right now
   • NUTRITIONAL SNAPSHOT: one sentence on where the pantry is strong and where it's light
   • WHAT'S NEARLY POSSIBLE: 1-2 dishes that need just one more ingredient

Be analytical. Your output feeds the Recipe Architect and Wellness Advisor — make it dense and useful."""


_RECIPE_PROMPT = """You are the Recipe Architect for PantryGenie 🧞 — a creative vegetarian chef.
You receive a Pantry Analysis and a user request. Design 2-3 outstanding recipes.

For EACH recipe:
1. Build it primarily from the provided pantry ingredients.
2. If a key ingredient is missing or would improve the dish, call find_vegetarian_substitution
   to find a pantry-friendly swap — then use THAT substitute in the recipe.
3. Format using EXACTLY this template (no variations, no extra fields):

---
## 🍲 [Recipe Name]
**Ingredients:** item1, item2, item3 *(note any not in pantry)*
**Directions:** Clear 2-3 sentence method with any key technique tips.
**Cook time:** X minutes
**Equipment:** [what's needed, e.g. stovetop, oven, instant pot]

Rules:
• No meat or fish — ever.
• Be creative. Flavour > simplicity.
• If equipment like Instant Pot is in the pantry analysis, prefer those recipes.
• Keep ingredient lists realistic — under 12 items.
• Do NOT add a Watch or YouTube field — that is handled separately."""


_WELLNESS_PROMPT = """You are the Wellness & Shopping Advisor for PantryGenie 🧞.
You receive a Pantry Analysis and the proposed recipes. Produce TWO sections:

SECTION 1 — NUTRITION CHECK:
Call estimate_meal_nutrition for each recipe (pass its ingredient list).
Summarise the nutritional picture across the meal suggestions.
Flag any gaps (e.g. low protein, low fibre) with a simple fix.

SECTION 2 — SMART SHOPPING:
Call analyze_shopping_gaps to find the best items to buy next.
Highlight the top 3 with a one-line reason each.

Format:
🥗 **Nutrition**
[your findings per recipe + overall flag]

🛒 **What to buy next**
[top 3 items with reason]

Be encouraging and practical. Keep it under 200 words total."""


_SUPERVISOR_PROMPT = """You are PantryGenie 🧞 — the friendly face of a multi-agent vegetarian recipe assistant.
Behind you are three specialists you coordinate:

• consult_pantry_analyst   — reads the user's pantry & preferences → returns structured context
• consult_recipe_architect — designs creative recipes (pass the analyst output in your call)
• consult_wellness_advisor — checks nutrition & shopping gaps (pass analyst + recipe output)

You also have direct tools to update the pantry or preferences when the user mentions changes.

== ROUTING RULES ==

"Suggest recipes" / "what can I cook" / any recipe request:
  1. consult_pantry_analyst("Read this user's full pantry and preferences")
  2. consult_recipe_architect("PANTRY ANALYSIS:\\n{analyst result}\\n\\nUSER REQUEST:\\n{request}")
  3. consult_wellness_advisor("PANTRY ANALYSIS:\\n{analyst result}\\n\\nRECIPES:\\n{recipe result}")
  4. Reply with the recipes (from step 2) followed by the wellness section (from step 3).

"What should I buy" / "shopping" / "what am I missing":
  1. consult_pantry_analyst(...)
  2. consult_wellness_advisor("PANTRY ANALYSIS:\\n{analyst result}\\n\\nFocus on: smart shopping gaps")
  3. Reply with a helpful shopping summary.

"Is my diet balanced" / "nutrition" / "am I eating well":
  1. consult_pantry_analyst(...)
  2. consult_wellness_advisor("PANTRY ANALYSIS:\\n{analyst result}\\n\\nFocus on: nutritional assessment")
  3. Reply with a friendly nutritional overview.

"I bought X" / "add X to my pantry":
  1. Call update_pantry with the FULL updated ingredient list (read pantry first if needed).
  2. Confirm warmly.

"I don't like X" / "I prefer Y spice level":
  1. Call update_user_preferences.
  2. Confirm warmly.

Substitution / cooking questions:
  1. consult_recipe_architect directly with the question — no analyst needed.

== RESPONSE STYLE ==
Warm, concise, personal. Use the user's name if you know it.
Present recipes exactly as the Recipe Architect formatted them (keep the --- separators and ## headers).
Append the wellness section below the recipes, separated by a divider.
Sign off as 🧞."""


# ── Sub-agent singletons (lazy-initialised) ───────────────────────────────────
_analyst_agent  = None
_recipe_agent   = None
_wellness_agent = None


def _get_analyst() -> object:
    global _analyst_agent
    if _analyst_agent is None:
        _analyst_agent = create_react_agent(
            _fast_llm(), ANALYST_TOOLS, prompt=_ANALYST_PROMPT
        )
    return _analyst_agent


def _get_recipe_agent() -> object:
    global _recipe_agent
    if _recipe_agent is None:
        _recipe_agent = create_react_agent(
            _llm(), RECIPE_TOOLS, prompt=_RECIPE_PROMPT
        )
    return _recipe_agent


def _get_wellness_agent() -> object:
    global _wellness_agent
    if _wellness_agent is None:
        _wellness_agent = create_react_agent(
            _fast_llm(), WELLNESS_TOOLS, prompt=_WELLNESS_PROMPT
        )
    return _wellness_agent


# ── Sub-agent wrapper tools (used by the supervisor) ──────────────────────────
@tool
def consult_pantry_analyst(question: str, config: RunnableConfig) -> str:
    """Consult the Pantry Analyst to read the user's ingredients and preferences.
    Always call this first before recipe suggestions or nutrition queries.
    question: what you want the analyst to investigate or report on.
    """
    result = _get_analyst().invoke(
        {"messages": [HumanMessage(content=question)]},
        config=config,
    )
    return result["messages"][-1].content


@tool
def consult_recipe_architect(context_and_request: str, config: RunnableConfig) -> str:
    """Ask the Recipe Architect to design vegetarian recipes.
    Always include the Pantry Analyst's output in context_and_request so the
    architect knows what ingredients are available.
    context_and_request: pantry analysis + the user's specific recipe request.
    """
    result = _get_recipe_agent().invoke(
        {"messages": [HumanMessage(content=context_and_request)]},
        config=config,
    )
    return result["messages"][-1].content


@tool
def consult_wellness_advisor(context: str, config: RunnableConfig) -> str:
    """Consult the Wellness & Shopping Advisor for nutrition insights and shopping strategy.
    Include the pantry analysis and any proposed recipes in context so the advisor
    can give specific, grounded advice.
    context: pantry analysis + recipes (if available) + specific focus question.
    """
    result = _get_wellness_agent().invoke(
        {"messages": [HumanMessage(content=context)]},
        config=config,
    )
    return result["messages"][-1].content


# ── Supervisor ────────────────────────────────────────────────────────────────
_memory = MemorySaver()

_SUPERVISOR_TOOLS = [
    consult_pantry_analyst,
    consult_recipe_architect,
    consult_wellness_advisor,
] + SUPERVISOR_TOOLS  # also has update_pantry, update_user_preferences


def build_agent():
    """Build the supervisor agent. Call once at startup."""
    return create_react_agent(
        _llm(),
        _SUPERVISOR_TOOLS,
        prompt=_SUPERVISOR_PROMPT,
        checkpointer=_memory,
    )


# ── YouTube injection (deterministic — never goes through the LLM) ────────────
def _yt_search(recipe_name: str) -> str:
    """Search YouTube for a recipe video. Returns a markdown link or empty string."""
    api_key = os.getenv("YOUTUBE_API_KEY", "")
    if not api_key:
        return ""
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
    return ""


def _inject_youtube_links(text: str) -> str:
    """Split output into recipe sections, search YouTube for each by name, inject link.
    This replaces any leaked <function=...> tags and fills missing Watch fields.
    """
    sections = re.split(r"(?=\n---\n|^---\n)", text, flags=re.MULTILINE)
    out = []
    for section in sections:
        name_m = re.search(r"##\s+(?:🍲\s*)?(.+)", section)
        if name_m:
            recipe_name = name_m.group(1).strip()
            link = _yt_search(recipe_name)
            if link:
                # Replace any existing Watch line (leaked tag, empty, placeholder)
                if "**Watch:**" in section:
                    section = re.sub(r"\*\*Watch:\*\*[^\n]*", f"**Watch:** {link}", section)
                else:
                    section = section.rstrip() + f"\n**Watch:** {link}"
        out.append(section)
    return "".join(out)


def _clean_output(text: str) -> str:
    # Ensure blank lines around --- dividers so Streamlit renders them correctly
    text = re.sub(r"\n---\n", "\n\n---\n\n", text)
    # Ensure blank lines before bold field labels
    for label in ("Ingredients", "Directions", "Cook time", "Equipment", "Watch"):
        text = re.sub(rf"(?<!\n)\n(\*\*{label}:\*\*)", r"\n\n\1", text)
    return text.strip()


# ── Public chat function ──────────────────────────────────────────────────────
def chat(user_input: str, agent, thread_id: str = "default", user_id: str = "default") -> str:
    """Send a message to the supervisor and return the formatted response."""
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    try:
        response = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )
        text = _clean_output(response["messages"][-1].content)
        text = _inject_youtube_links(text)  # deterministic, no LLM involved
        return text
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}"


# ── Terminal test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = build_agent()
    print("🧞 PantryGenie multi-agent system ready. Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue
        print(f"\nPantryGenie: {chat(user_input, agent)}\n")
