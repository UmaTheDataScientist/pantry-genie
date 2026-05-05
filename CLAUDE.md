# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the app

```bash
streamlit run ui/app.py
```

To test the agent without the UI:
```bash
python -m pantry_genie.agent
```

To ingest recipes into Pinecone (one-time setup):
```bash
python -m pantry_genie.ingest
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Architecture

PantryGenie is a Streamlit + LangGraph chatbot that suggests vegan recipes from a user's pantry. It is deployed on [Streamlit Cloud](https://pantry-genie.streamlit.app).

**Entry point:** `ui/app.py` — handles Google OAuth login via `streamlit-oauth`, then renders the chat UI.

**Agent layer:** `pantry_genie/agent.py` — builds a LangGraph `create_react_agent` backed by Groq (Llama 3.3 70B). Uses `MemorySaver` for in-session conversation memory keyed by `thread_id`.

**Tools:** `pantry_genie/tools.py` — five LangChain `@tool` functions:
- `get_pantry_contents` / `update_pantry` — read/write a `pantry` table in Supabase
- `get_user_preferences` / `update_user_preferences` — read/write a `preferences` table in Supabase
- `search_youtube` — calls YouTube Data API v3 to find a recipe video

**Data ingestion:** `pantry_genie/ingest.py` — one-time script that loads `data/vegan_recipes.csv`, filters non-vegan rows, and upserts embeddings to Pinecone (namespace `recipes`). Pinecone is configured but not actively queried by the agent at runtime — recipes come from the LLM's own knowledge.

## Environment variables

All secrets are loaded from `.env` locally and from Streamlit Cloud Secrets in production. Required keys:

| Key | Purpose |
|-----|---------|
| `GOOGLE_CLIENT_ID` | OAuth client ID |
| `GOOGLE_CLIENT_SECRET` | OAuth client secret |
| `REDIRECT_URI` | OAuth redirect (e.g. `https://pantry-genie.streamlit.app`) |
| `GROQ_API_KEY` | Groq LLM inference |
| `SUPABASE_URL` / `SUPABASE_KEY` | Supabase DB |
| `YOUTUBE_API_KEY` | YouTube search |
| `PINECONE_API_KEY` / `PINECONE_INDEX` | Pinecone vector store |
| `STATSIG_SERVER_KEY` | Feature flags (optional) |

`.env` and `.streamlit/` are gitignored. For Streamlit Cloud, secrets must be added in the app dashboard under **Settings → Secrets**.

## Key patterns

- `user_id` flows from the OAuth `email` claim through `st.session_state.user_info` → passed as `config["configurable"]["user_id"]` to every agent invocation → extracted inside tools via `_get_user_id(config)`.
- Secrets are loaded into `os.environ` at startup in both `ui/app.py` and `pantry_genie/agent.py` / `tools.py` so all three modules can use `os.getenv()` uniformly.
- The `_secret()` helper in `ui/app.py` tries `os.getenv` first, then falls back to `st.secrets.get`.
