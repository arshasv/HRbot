from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from hr_bot.core.llm_config import llm
from hr_bot.core.knowledge_config import get_hr_policy_knowledge
from hr_bot.core.embedder_config import get_azure_openai_embedder

@CrewBase
class HRCrew:
    """HR Crew to handle employee policy queries."""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    hr_policy_pdf = get_hr_policy_knowledge()
    embedder = get_azure_openai_embedder()

    # ------------- Agents ------------- #
    @agent
    def policy_retriever(self) -> Agent:
        return Agent(
            config=self.agents_config["policy_retriever"],
            verbose=True,
            llm=llm,
            knowledge_sources=[self.hr_policy_pdf],
        )

    @agent
    def hr_expert(self) -> Agent:
        return Agent(
            config=self.agents_config["hr_expert"],
            verbose=True,
            llm=llm,
            knowledge_sources=[self.hr_policy_pdf],
        )

    @agent
    def conversation_manager(self) -> Agent:
        return Agent(
            config=self.agents_config["conversation_manager"],
            verbose=True,
            llm=llm,
        )

    # ------------- Tasks ------------- #
    @task
    def employee_policy_task(self) -> Task:
        return Task(config=self.tasks_config["employee_policy_task"], output_file="employee_policy_answer.md")

    @task
    def leave_task(self) -> Task:
        return Task(config=self.tasks_config["leave_task"], output_file="leave_policy_answer.md")

    @task
    def benefits_task(self) -> Task:
        return Task(config=self.tasks_config["benefits_task"], output_file="benefits_policy_answer.md")

    @task
    def compliance_task(self) -> Task:
        return Task(config=self.tasks_config["compliance_task"], output_file="compliance_answer.md")

    @task
    def onboarding_task(self) -> Task:
        return Task(config=self.tasks_config["onboarding_task"], output_file="onboarding_answer.md")

    @task
    def performance_task(self) -> Task:
        return Task(config=self.tasks_config["performance_task"], output_file="performance_answer.md")

    @task
    def intent_detection_task(self) -> Task:
        return Task(config=self.tasks_config["intent_detection_task"], output_file="intent_detection.md")

    # ------------- Crew ------------- #
    @crew
    def crew(self) -> Crew:
        """Creates the HR Crew."""
        return Crew(
            memory=False,
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            llm=llm,
            adapted_agent=True,
            knowledge_sources=[self.hr_policy_pdf],
            embedder=self.embedder,
            collection_name="hr_policy_collection",
        )
