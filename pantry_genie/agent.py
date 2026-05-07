import os
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

For recipe suggestions, always follow this exact sequence:
1. Call get_pantry_contents
2. Call get_user_preferences
3. Call search_youtube once for each recipe you plan to suggest (one call per recipe)
4. Then write your final response in this markdown format for each recipe:

---
## 🍲 [Recipe Name]
**Ingredients:** item1, item2, item3
**Directions:** Brief 2-3 sentence description.
**Cook time:** X minutes
**Watch:** [paste the exact URL returned by search_youtube here]

CRITICAL: The search_youtube tool returns a markdown link like ▶️ [title](url). Paste it exactly. Never write {"recipe_name": ...} or any JSON in your response.

If the user mentions new ingredients they have:
- Call update_pantry first, then follow the recipe sequence above

If the user mentions a preference (spice level, dislike, cuisine):
- Call update_user_preferences, then acknowledge conversationally

Always use markdown formatting. Be warm and concise."""


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
        return response["messages"][-1].content
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