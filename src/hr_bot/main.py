# Override system sqlite3 with pysqlite3 for Chroma compatibility
try:
    import sys
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3  # Replace sqlite3 module entirely
    import sqlite3
    print(f"✅ Overridden SQLite with pysqlite3 version: {sqlite3.sqlite_version}")
except ImportError:
    print("⚠️ pysqlite3 not installed; falling back to system SQLite (may fail with Chroma)")

import os
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from hr_bot.crew import HRCrew
from hr_bot.flows.hr_query_flow import HRQueryFlow

# Load environment variables
load_dotenv()

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# App and Crew initialization
# ---------------------------------------------------------------------
app = FastAPI(
    title="HR Bot",
    description="HR Policy Assistant powered by CrewAI",
    version="1.0"
)

# ---------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str

# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.get("/")
async def root():
    return {"message": "Welcome to the HR Bot API"}

@app.post("/ask")
async def ask_hr_bot(request: QueryRequest):
    """Endpoint to ask HR-related questions."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    logger.info(f"Received question: {question}")

    try:
        hr_flow = HRQueryFlow()
        answer = hr_flow.run_dynamic_task(question)
        return {"question": question, "answer": answer}
    except Exception as e:
        logger.error(f"❌ Error running HR query flow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("hr_bot.main:app", host="0.0.0.0", port=8000, reload=True)