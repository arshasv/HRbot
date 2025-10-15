import logging
from crewai import Crew, Process
from hr_bot.crew import HRCrew
from hr_bot.core.llm_config import llm

logger = logging.getLogger(__name__)

class HRQueryFlow:
    def __init__(self):
        self.hr_crew_instance = HRCrew()
        self.crew = self.hr_crew_instance.crew()
        self.hr_policy_pdf = self.hr_crew_instance.hr_policy_pdf
        self.embedder = self.hr_crew_instance.embedder

    def detect_intent(self, query: str) -> str:
        """Run intent detection using the conversation_manager agent."""
        try:
            cm = self.hr_crew_instance.conversation_manager()
            logger.info("🔍 Running intent detection...")
            result = cm.kickoff(
                f"Classify this HR question into: leave_task, benefits_task, "
                f"compliance_task, onboarding_task, performance_task, or employee_policy_task.\n\n"
                f"Question: {query}\n\nReturn only the task name."
            )
            return result.raw.strip().lower()
        except Exception as e:
            logger.error(f"❌ Intent detection failed: {e}")
            return "unclassified"

    def run_dynamic_task(self, query: str):
        """Dynamically run the correct HR policy task after intent detection."""
        intent = self.detect_intent(query)
        logger.info(f"Intent detected: {intent}")

        if intent == "unclassified":
            return "Sorry, I couldn’t classify your question into a known HR policy area."

        try:
            # Dynamically get the task from HRCrew (e.g. leave_task)
            task_method = getattr(self.hr_crew_instance, intent, None)
            if not task_method:
                return f"⚠️ No task found for intent: {intent}"

            # Instantiate the task and inject the query into its description
            task = task_method()
            if hasattr(task, "description") and "{query}" in task.description:
                task.description = task.description.replace("{query}", query)

            # Get the agent for this task
            agent_name_or_obj = getattr(task, "agent", None)

            if agent_name_or_obj is None:
                return f"⚠️ Task {intent} does not specify an agent."

            # Handle both string agent names and Agent objects
            if isinstance(agent_name_or_obj, str):
                agent_method = getattr(self.hr_crew_instance, agent_name_or_obj, None)
                if not agent_method:
                    return f"⚠️ No agent found for task: {agent_name_or_obj}"
                agent = agent_method()
            elif hasattr(agent_name_or_obj, "kickoff"):  # already an Agent instance
                agent = agent_name_or_obj
            else:
                return f"⚠️ Invalid agent configuration for task: {intent}"


            # Create a mini crew for this query and mention the knowledge source and to take only from it
            mini_crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True,
                memory=False,
                llm=llm,
                adapted_agent=True,
                knowledge_sources=[self.hr_crew_instance.hr_policy_pdf],
                collection_name="hr_policy_collection",
                embedder=self.embedder,
                )

            # Run the crew
            result = mini_crew.kickoff()
            return result.output if hasattr(result, "output") else result

        except Exception as e:
            logger.error(f"❌ Failed to run dynamic task: {e}", exc_info=True)
            return f"An error occurred while processing your request: {e}"