import os
from dotenv import load_dotenv
from crewai import LLM

load_dotenv()

def gemini_creative():
    """Creative Gemini model (higher temperature)."""
    return LLM(
        model="gemini/gemini-2.5-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.7,
        top_p=0.8
    )

def gemini():
    """Default Gemini model (deterministic)."""
    return LLM(
        model="gemini/gemini-2.5-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.3
    )

def azure_gpt_mini():
    """Example Azure GPT model (through LiteLLM backend)."""
    return LLM(
        model="azure/gpt-5-mini",
        api_key=os.getenv("AZURE_API_KEY")
    )

def gemini_embedder(texts):
    """Custom embedding function using Gemini Embedding model."""
    embedder = LLM(
        model="gemini/gemini-2.5-embed-text",
        api_key=os.getenv("GEMINI_API_KEY"),
        embedding_callable=True  # Indicate this LLM is used for embeddings
    )
    return embedder.embed(texts)

# Choose which one to expose as default for your agents
llm = gemini()
