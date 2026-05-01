import os
import pandas as pd
from pinecone import Pinecone
from dotenv import load_dotenv
import uuid

load_dotenv()

# ── Init Pinecone ──────────────────────────────────────────
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

def load_recipes(path: str = "data/vegan_recipes.csv") -> pd.DataFrame:
    """Load and clean the recipes CSV."""
    df = pd.read_csv(path)
    
    # Drop rows missing the essentials
    df = df.dropna(subset=["recipe_name", "ingredients", "directions"])
    
    # Keep only relevant columns
    df = df[["recipe_name", "ingredients", "directions", "nutrition", "cuisine_path", "total_time"]]
    
    # Filter vegan-friendly (no meat/dairy keywords)
    exclude = ["chicken", "beef", "pork", "lamb", "fish", "shrimp", "bacon",
               "turkey", "salmon", "tuna", "cheese", "butter", "milk", "cream",
               "egg", "eggs", "gelatin", "honey", "lard", "mayo", "mayonnaise"]
    
    pattern = "|".join(exclude)
    df = df[~df["ingredients"].str.lower().str.contains(pattern, na=False)]
    
    print(f"✅ Loaded {len(df)} vegan-friendly recipes")
    return df.head(500)  # Pinecone free tier — keep it lean


def build_text(row: pd.Series) -> str:
    """Combine fields into a single rich text string for embedding."""
    return f"""
Recipe: {row['recipe_name']}
Cuisine: {row.get('cuisine_path', 'Unknown')}
Total Time: {row.get('total_time', 'Unknown')}
Ingredients: {row['ingredients']}
Directions: {row['directions']}
Nutrition: {row.get('nutrition', '')}
""".strip()

def ingest():
    """Embed and upsert all recipes into Pinecone."""
    df = load_recipes()
    
    vectors = []
    for _, row in df.iterrows():
        text = build_text(row)
        vector_id = str(uuid.uuid4())
        
        vectors.append({
            "id": vector_id,
            "text": text,
            "recipe_name": str(row["recipe_name"]),
            "ingredients": str(row["ingredients"]),
            "directions": str(row["directions"]),
            "cuisine": str(row.get("cuisine_path", "")),
            "total_time": str(row.get("total_time", "")),
            "nutrition": str(row.get("nutrition", "")),
        })
    
    # Upsert in batches of 50
    batch_size = 50
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert_records(
            namespace="recipes",
            records=batch
        )
        print(f"⬆️  Upserted batch {i // batch_size + 1} / {len(vectors) // batch_size + 1}")
    
    print("🎉 All recipes ingested into Pinecone!")

if __name__ == "__main__":
    ingest()