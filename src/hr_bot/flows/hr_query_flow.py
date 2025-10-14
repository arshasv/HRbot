from crewai.flow.flow import Flow, start, listen
import asyncio
import logging

logger = logging.getLogger(__name__)

class HRQueryFlow(Flow):
    """
    Handles HR query resolution:
      1. Detects intent via conversation_manager
      2. Executes the relevant HR task
    """

    def __init__(self, agents, tasks):
        super().__init__()
        self.agents = agents
        self.tasks = tasks
        self.conversation_manager = agents["conversation_manager"]

    # ------------------------- STEP 1 -------------------------
    @start()
    async def detect_intent(self):
        """Detect which HR task the query belongs to."""
        query = self.state.get("query")
        if not query:
            raise ValueError("❌ Missing 'query' in flow state.")

        logger.info(f"🔍 Detecting intent for query: {query}")

        prompt = (
            "You are an HR assistant. Analyze the employee's question and determine which HR task it belongs to. "
            "Choose one of: leave_task, benefits_task, compliance_task, onboarding_task, "
            "performance_task, or employee_policy_task.\n\n"
            f"Employee question: {query}\n\n"
            "Return only the task name (e.g., 'leave_task')."
        )

        logger.info("🧠 Running intent detection LLM call...")
        # ✅ Run synchronous LLM call in a separate thread
        result = await asyncio.to_thread(self.conversation_manager.llm.call, prompt)
        logger.info(f"🧠 Intent detection result: {result}")
        cleaned = result.strip().lower()

        # Save to state for next step
        self.state["task_name"] = cleaned
        print(f"✅ Intent detected ==========================> {cleaned}")

        return cleaned

    # ------------------------- STEP 2 -------------------------
    @listen(detect_intent)
    async def execute_task(self):
        task_name = self.state.get("task_name")
        query = self.state.get("query")

        if not task_name or task_name not in self.tasks:
            return "Sorry, I couldn't map your question to a defined HR category."

        # Use HRCrew's run_task
        hr_crew = self.state.get("hr_crew")
        result = await hr_crew.run_task(task_name, query)
        return result

