import os
import re
import requests
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from statsig import statsig, StatsigUser
from dotenv import load_dotenv
from pantry_genie.tools import TOOLS

load_dotenv()
# ── Load Streamlit secrets if running on cloud ─────────────
try:
    import streamlit as st
    for key, value in st.secrets.items():
        os.environ.setdefault(key, value)
except:
    pass
# ── Statsig Feature Flags ──────────────────────────────────
import asyncio

# ── Statsig Feature Flags ──────────────────────────────────
def init_statsig():
    try:
        key = os.getenv("STATSIG_SERVER_KEY")
        if key:
            if hasattr(statsig, "initialize_sync"):
                statsig.initialize_sync(key)
            else:
                import asyncio
                asyncio.get_event_loop().run_until_complete(statsig.initialize(key))
    except Exception:
        pass

init_statsig()

def get_model_name() -> str:
    try:
        import streamlit as st
        for key, value in st.secrets.items():
            os.environ.setdefault(key, value)
    except:
        pass
    return "llama-3.3-70b-versatile"
    
# ── LLM ───────────────────────────────────────────────────
def build_llm() -> ChatGroq:
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model=get_model_name(),
        temperature=0,
    )


# ── System Prompt ──────────────────────────────────────────
SYSTEM_PROMPT = """You are PantryGenie 🧞, a warm vegetarian recipe assistant (dairy and eggs OK, no meat or fish).

RULE: Before suggesting any recipes you MUST call get_pantry_contents and get_user_preferences. Do not skip these tool calls.

For recipe suggestions, follow this sequence:
1. Call get_pantry_contents
2. Call get_user_preferences
3. Suggest 2-3 recipes using pantry items, respecting preferences
4. For EACH recipe call search_youtube and include the returned link

Format each recipe exactly like this (use markdown):

---
## 🍲 Recipe Name
**Ingredients:** item1, item2, item3
**Directions:** 2-3 sentences.
**Cook time:** X minutes
**Watch:** [the link returned by search_youtube]

If the user mentions new ingredients: call update_pantry first, then suggest recipes.
If the user mentions a preference: call update_user_preferences, then acknowledge.

Be warm and concise."""


# ── YouTube fallback ───────────────────────────────────────
def _resolve_youtube_tags(text: str) -> str:
    """Replace leaked <function=search_youtube>{...}</function> tags with real links."""
    pattern = r'<function=search_youtube>\{"recipe_name":\s*"([^"]+)"\}</function>'

    def fetch(match):
        name = match.group(1)
        api_key = os.getenv("YOUTUBE_API_KEY", "")
        if not api_key:
            return "*(no YouTube key)*"
        try:
            resp = requests.get(
                "https://www.googleapis.com/youtube/v3/search",
                params={"part": "snippet", "q": f"{name} vegetarian recipe",
                        "type": "video", "maxResults": 1, "key": api_key},
                timeout=5,
            )
            items = resp.json().get("items", [])
            if items:
                vid = items[0]["id"]["videoId"]
                title = items[0]["snippet"]["title"]
                return f"[▶️ {title}](https://www.youtube.com/watch?v={vid})"
        except Exception:
            pass
        return "*(video unavailable)*"

    return re.sub(pattern, fetch, text)


def _format_recipes(text: str) -> str:
    """Convert plain-text recipe list into clean markdown."""
    text = _resolve_youtube_tags(text)

    # Remove filler intro line
    text = re.sub(r'^(here are [^\n]+\n+)', '', text, flags=re.IGNORECASE)

    # Turn "1. Recipe Name: ..." or "Recipe Name: ..." lines into ## headers
    def make_header(m):
        name = m.group(2).strip()
        rest = m.group(3).strip()
        # Split rest into labelled sections
        rest = re.sub(r'\bCook time:\s*', '\n**Cook time:** ', rest)
        rest = re.sub(r'\bYouTube link:\s*', '\n**Watch:** ', rest)
        rest = re.sub(r'\bIngredients:\s*', '\n**Ingredients:** ', rest)
        rest = re.sub(r'\bDirections:\s*', '\n**Directions:** ', rest)
        return f"\n---\n## 🍲 {name}\n{rest.strip()}"

    text = re.sub(
        r'(?m)^(\d+\.\s+)?([A-Z][^:\n]{3,60}):\s+(.+)',
        make_header,
        text,
    )
    return text.strip()


# ── Memory (LangGraph) ─────────────────────────────────────
memory = MemorySaver()  # In-memory checkpointer for session history


# ── Agent ──────────────────────────────────────────────────
def build_agent():
    """Build a LangGraph ReAct agent with memory and tools."""
    llm = build_llm()

    agent = create_react_agent(
        model=llm,
        tools=TOOLS,
        prompt=SYSTEM_PROMPT,
        checkpointer=memory,   # Enables session memory
    )
    return agent


# ── Chat function ──────────────────────────────────────────
def chat(user_input: str, agent, thread_id: str = "default", user_id: str = "default") -> str:
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    try:
        response = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )
        return _format_recipes(response["messages"][-1].content)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}"


# ── Quick terminal test ────────────────────────────────────
if __name__ == "__main__":
    agent = build_agent()
    print("🧞 PantryGenie is ready! Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            break
        if not user_input:
            continue
        response = chat(user_input, agent)
        print(f"\nPantryGenie: {response}\n")