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
        statsig.initialize_sync(os.getenv("STATSIG_SERVER_KEY"))
        print("✅ Statsig initialized")
    except Exception as e:
        print(f"⚠️ Statsig init failed: {e}")

init_statsig()

def get_model_name() -> str:
    try:
        statsig.initialize_sync(os.getenv("STATSIG_SERVER_KEY"))
        user = StatsigUser(user_id="pantry-genie-user")
        use_large = statsig.check_gate(user, "use_large_model")
        if use_large:
            print("🚀 Statsig: using llama-3.3-70b-versatile")
        else:
            print("⚡ Statsig: using llama-3.3-70b-versatile")
    except Exception as e:
        print(f"⚠️ Statsig unavailable — using llama-3.3-70b-versatile")
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
You are PantryGenie 🧞, a warm and knowledgeable vegan recipe assistant.

Your job:
1. Help users figure out what to cook based on ingredients they have
2. Suggest vegan recipes that match their pantry and taste preferences
3. Remember their preferences over time (spice level, dislikes, favorite cuisines)
4. Be conversational, encouraging and fun

Rules:
- ALWAYS check user preferences before suggesting recipes
- ALWAYS update pantry when user mentions ingredients they have
- ALWAYS update preferences when user mentions likes/dislikes
- Suggest maximum 2-3 recipes at a time
- If a recipe needs a missing ingredient, mention it but keep it minimal
- Keep responses concise and friendly

You have access to tools — use them proactively.
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
def chat(user_input: str, agent, thread_id: str = "default") -> str:
    """Send a message to PantryGenie and get a response."""
    config = {"configurable": {"thread_id": thread_id}}
    try:
        response = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config
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