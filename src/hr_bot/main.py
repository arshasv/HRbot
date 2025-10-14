import os
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

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

try:
    from hr_bot.crew import HRCrew
    hr_crew = HRCrew()
    logger.info("✅ HR Crew initialized successfully.")
except Exception as e:
    hr_crew = None
    logger.error(f"❌ Failed to initialize HR Crew: {e}")

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
    if not hr_crew:
        raise HTTPException(status_code=500, detail="HR Crew not initialized. Check server logs.")

    try:
        logger.info(f"📩 Received query =======> {request.question}")

        # Await the async handle_query method
        result = await hr_crew.handle_query_async(request.question)

        logger.info(f"✅ Response ========> {result}")
        return {"response": result}
    except Exception as e:
        logger.exception("Error while processing HR query")
        raise HTTPException(status_code=500, detail=str(e))



# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("hr_bot.main:app", host="0.0.0.0", port=8000, reload=True)
