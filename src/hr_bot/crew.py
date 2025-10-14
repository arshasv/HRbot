import asyncio
import os
import yaml
import logging
from crewai import Agent, Task, Crew, Process
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource
from dotenv import load_dotenv
from crewai.utilities.constants import KNOWLEDGE_DIRECTORY
from hr_bot.llm_config import llm
from hr_bot.flows.hr_query_flow import HRQueryFlow

from hr_bot.llm_config import llm

# ---------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class HRCrew:
    def __init__(self):
        """
        Initialize the HR Crew, loading agents, tasks, and HR policy knowledge.
        """
        # -----------------------------------------------------------------
        # Load YAML configurations
        # -----------------------------------------------------------------
        base_path = os.path.dirname(__file__)
        with open(os.path.join(base_path, "config/agents.yaml"), "r") as f:
            agents_yaml = yaml.safe_load(f)
        with open(os.path.join(base_path, "config/tasks.yaml"), "r") as f:
            tasks_yaml = yaml.safe_load(f)

        # -----------------------------------------------------------------
        # Initialize agents and tasks
        # -----------------------------------------------------------------
        self.agents = self._init_agents(agents_yaml)
        self.tasks = self._init_tasks(tasks_yaml, self.agents)

        # -----------------------------------------------------------------
        # Define main Crew orchestrator
        # -----------------------------------------------------------------
        self.crew = Crew(
            agents=list(self.agents.values()),
            tasks=list(self.tasks.values()),
            process=Process.sequential,
            verbose=True,
            verbose_level=2,
        )

    # ---------------------------------------------------------------------
    # Agent Initialization
    # ---------------------------------------------------------------------
    def _init_agents(self, agents_yaml):
        """Initialize agents from YAML configuration."""
        agents = {}

        logger.info("✅ Setting up PDF knowledge source.")
        logger.info(f"KNOWLEDGE_DIRECTORY: {KNOWLEDGE_DIRECTORY}")

        # Build correct absolute PDF path
        knowledge_dir = KNOWLEDGE_DIRECTORY
        os.makedirs(knowledge_dir, exist_ok=True)  # Create directory if it doesn't exist

        # -----------------------------------------------------------------
        # Initialize PDF Knowledge Source
        # -----------------------------------------------------------------
        pdf_source = PDFKnowledgeSource(
            file_paths=[os.path.join(KNOWLEDGE_DIRECTORY, "HR_POLICY.pdf")],
            chunk_size=500,
            chunk_overlap=50,
        )

        # -----------------------------------------------------------------
        # Build Agents
        # -----------------------------------------------------------------
        for name, config in agents_yaml.items():
            tools = []
            knowledge_sources = []

            if name in ["policy_retriever", "hr_expert"]:
                knowledge_sources.append(pdf_source)

            agents[name] = Agent(
                role=config["role"],
                goal=config["goal"],
                backstory=config["backstory"],
                llm=llm,
                allow_delegation=config.get("allow_delegation", False),
                tools=tools,
                knowledge_sources=knowledge_sources,
                verbose=True,
            )

        logger.info(f"✅ Initialized {len(agents)} agents.")
        return agents

    # ---------------------------------------------------------------------
    # Task Initialization
    # ---------------------------------------------------------------------
    def _init_tasks(self, tasks_yaml, agents):
        """Initialize tasks from YAML configuration."""
        tasks = {}
        for name, config in tasks_yaml.items():
            agent_name = config["agent"]
            if agent_name not in agents:
                raise ValueError(f"Agent '{agent_name}' not found for task '{name}'")

            tasks[name] = Task(
                description=config["description"],
                expected_output=config["expected_output"],
                agent=agents[agent_name],
            )

        logger.info(f"✅ Initialized {len(tasks)} tasks.")
        return tasks

    # ---------------------------------------------------------------------
    # Query Handling
    # ---------------------------------------------------------------------
    async def handle_query_async(self, query: str):
        logger.info(f"Starting HR Flow for query: {query}")
        hr_flow = HRQueryFlow(self.agents, self.tasks)

        # Inject query and HRCrew instance into flow state
        hr_flow.state["query"] = query
        hr_flow.state["hr_crew"] = self  # ✅ pass self for run_task usage

        # kickoff with empty inputs
        result = await hr_flow.kickoff_async(inputs={})
        logger.info(f"✅ Flow Result: {result}")
        return result



    # ---------------------------------------------------------------------
    # Task Execution
    # ---------------------------------------------------------------------
    async def run_task(self, task_name: str, query: str):
        if task_name not in self.tasks:
            raise ValueError(f"Task '{task_name}' not found.")

        task = self.tasks[task_name]
        agent = task.agent
        prompt = task.description.format(query=query)
        print(f"Running Task {task_name} with agent {agent.role}")
        print(f"Knowledge sources: {agent.knowledge_sources}")

        # ✅ Run Gemini LLM synchronously in a thread
        result = await asyncio.to_thread(agent.llm.call, prompt)
        return result


    # ---------------------------------------------------------------------
    # Default Run
    # ---------------------------------------------------------------------
    def run(self):
        """Run a simple initialization test."""
        logger.info("HR Crew initialized. Ready for queries.")


# -------------------------------------------------------------------------
# Manual Test Entry
# -------------------------------------------------------------------------
if __name__ == "__main__":
    hr_crew = HRCrew()
    hr_crew.run()
