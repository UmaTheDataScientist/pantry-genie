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
SYSTEM_PROMPT = """
You are PantryGenie 🧞, a warm and knowledgeable vegetarian recipe assistant (dairy and eggs are fine, no meat or fish).

When the user mentions ingredients they have, follow these steps IN ORDER:
1. Call update_pantry with the ingredients they mentioned
2. Call get_pantry_contents to get their full stored pantry
3. Call get_user_preferences to check their taste profile
4. Generate 2-3 vegetarian recipes using ingredients from their pantry — pick combinations that work well, you don't need to use every item. Respect their spice level, avoid their dislikes, and lean toward their favourite cuisines if set.
5. For EACH recipe, call search_youtube and include the returned link in your response
6. Present each recipe with: name, key ingredients, brief directions, cook time, and YouTube link

When the user asks what they can cook or what to make (without mentioning new ingredients):
1. Call get_pantry_contents to see what they have
2. Call get_user_preferences to check their taste profile
3. Then follow steps 4-6 above

When the user mentions a preference (spice level, dislike, favourite cuisine):
1. Call update_user_preferences immediately
2. Acknowledge the update conversationally

General rules:
- Be warm, concise, and encouraging
- Never skip a step — always fetch pantry and preferences before suggesting recipes
- Never fabricate YouTube links — only use the URL returned by search_youtube
"""


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
    """Send a message to PantryGenie and get a response."""
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    for attempt in range(2):
        try:
            response = agent.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config
            )
            return response["messages"][-1].content
        except Exception as e:
            if attempt == 0 and "tool_use_failed" in str(e):
                continue
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