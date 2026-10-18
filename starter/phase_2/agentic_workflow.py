# agentic_workflow.py
#
# General-purpose agentic workflow for technical project management.
# Pilot: turns the Email Router product spec into user stories, product
# features and engineering tasks.
#
# File layout:
#   1. Setup (imports, API key, product spec)
#   2. Agent instantiation (action planning, knowledge, evaluation, routing)
#   3. Support functions (one per routed role)
#   4. Main workflow (plan steps -> route each step -> consolidated plan)

# ---------------------------------------------------------------------------
# 1. SETUP
# ---------------------------------------------------------------------------
from workflow_agents.base_agents import (
    ActionPlanningAgent,
    KnowledgeAugmentedPromptAgent,
    EvaluationAgent,
    RoutingAgent,
)

import os
from dotenv import load_dotenv

# Folder of this script, so files load correctly from any working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the OpenAI key from the .env file next to this script
load_dotenv(os.path.join(SCRIPT_DIR, ".env"))
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY is not set. Add it to phase_2/.env")

# load the product spec
with open(os.path.join(SCRIPT_DIR, "Product-Spec-Email-Router.txt"), "r", encoding="utf-8") as spec_file:
    product_spec = spec_file.read()

# Maximum worker/evaluator rounds for each evaluation agent (keeps API use bounded)
MAX_INTERACTIONS = 5

# ---------------------------------------------------------------------------
# 2. AGENT INSTANTIATION
# ---------------------------------------------------------------------------

# Action Planning Agent
knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product "
    "described in the specification. \n"
    "Features are defined by grouping related user stories. \n"
    "Tasks are defined for each story and represent the engineering "
    "work required to develop the product. \n"
    "A development Plan for a product contains all these components"
)
action_planning_agent = ActionPlanningAgent(openai_api_key, knowledge_action_planning)

# Product Manager - Knowledge Augmented Prompt Agent
persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
knowledge_product_manager = (
    "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
    "The sentences always start with: As a "
    "Write several stories for the product spec below, where the personas are the different users of the product. "
    f"{product_spec}"
)
product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key, persona_product_manager, knowledge_product_manager
)

# Product Manager - Evaluation Agent
persona_product_manager_eval = "You are an evaluation agent that checks the answers of other worker agents"
evaluation_criteria_product_manager = (
    "The answer should be stories that follow the following structure: "
    "As a [type of user], I want [an action or feature] so that [benefit/value]."
)
product_manager_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona_product_manager_eval,
    evaluation_criteria_product_manager,
    product_manager_knowledge_agent,  # agent_to_evaluate (named worker_agent in EvaluationAgent)
    MAX_INTERACTIONS,
)

# Program Manager - Knowledge Augmented Prompt Agent
persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
knowledge_program_manager = "Features of a product are defined by organizing similar user stories into cohesive groups."
# Extra knowledge for the Program Manager: the expected output format, and the
# product spec so that its features are specific to the Email Router
knowledge_program_manager += (
    "\nWrite every feature with exactly these four labelled lines: Feature Name, Description, "
    "Key Functionality, User Benefit. List each feature only once."
    f"\nThe product specification is:\n{product_spec}"
)
program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key, persona_program_manager, knowledge_program_manager
)

# Program Manager - Evaluation Agent
persona_program_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."
evaluation_criteria_program_manager = (
    "The answer should be product features that follow the following structure: "
    "Feature Name: A clear, concise title that identifies the capability\n"
    "Description: A brief explanation of what the feature does and its purpose\n"
    "Key Functionality: The specific capabilities or actions the feature provides\n"
    "User Benefit: How this feature creates value for the user"
)
program_manager_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona_program_manager_eval,
    evaluation_criteria_program_manager,
    program_manager_knowledge_agent,  # agent_to_evaluate
    MAX_INTERACTIONS,
)

# Development Engineer - Knowledge Augmented Prompt Agent
persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
knowledge_dev_engineer = "Development tasks are defined by identifying what needs to be built to implement each user story."
# Extra knowledge for the Development Engineer: the expected output format, and the
# product spec so that its tasks are specific to the Email Router
knowledge_dev_engineer += (
    "\nDefine one task for each main component of the product specification. Write every task with "
    "exactly these seven labelled lines, starting each line with its label and a colon, and starting "
    "every task with a line like 'Task ID: TASK-001': Task ID, Task Title, Related User Story (written as "
    "'As a [type of user], I want [an action or feature] so that [benefit/value]'), Description, "
    "Acceptance Criteria, Estimated Effort, Dependencies."
    f"\nThe product specification is:\n{product_spec}"
)
development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key, persona_dev_engineer, knowledge_dev_engineer
)

# Development Engineer - Evaluation Agent
persona_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents."
evaluation_criteria_dev_engineer = (
    "The answer should be tasks following this exact structure: "
    "Task ID: A unique identifier for tracking purposes\n"
    "Task Title: Brief description of the specific development work\n"
    "Related User Story: Reference to the parent user story\n"
    "Description: Detailed explanation of the technical work required\n"
    "Acceptance Criteria: Specific requirements that must be met for completion\n"
    "Estimated Effort: Time or complexity estimation\n"
    "Dependencies: Any tasks that must be completed first"
)
development_engineer_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona_dev_engineer_eval,
    evaluation_criteria_dev_engineer,
    development_engineer_knowledge_agent,  # agent_to_evaluate
    MAX_INTERACTIONS,
)

# Routing Agent
# Each route's func is a lambda, so the support functions (defined below) are
# looked up only when a step is routed.
routing_agent = RoutingAgent(openai_api_key, [])
routing_agent.agents = [
    {
        "name": "Product Manager",
        "description": "Responsible for defining product personas and user stories only, written as "
                       "'As a [type of user], I want [an action or feature] so that [benefit/value]'. "
                       "Does not define features or tasks. Does not group stories",
        "func": lambda x: product_manager_support_function(x),
    },
    {
        "name": "Program Manager",
        "description": "Responsible for defining product features only, each with a Feature Name, Description, "
                       "Key Functionality and User Benefit. Does not write stories. Does not define engineering tasks",
        "func": lambda x: program_manager_support_function(x),
    },
    {
        "name": "Development Engineer",
        "description": "Responsible for defining engineering development tasks only, each with a Task ID, "
                       "Acceptance Criteria, Estimated Effort and Dependencies. Does not write stories or features",
        "func": lambda x: development_engineer_support_function(x),
    },
]

# ---------------------------------------------------------------------------
# 3. SUPPORT FUNCTIONS
# ---------------------------------------------------------------------------
# Final responses saved by role, used to print one consolidated project plan
project_plan = {
    "Product Manager": [],
    "Program Manager": [],
    "Development Engineer": [],
}


def product_manager_support_function(query):
    """Have the Product Manager evaluation agent produce validated user stories for the step."""
    result = product_manager_evaluation_agent.evaluate(query)
    project_plan["Product Manager"].append(result["final_response"])
    return result["final_response"]


def program_manager_support_function(query):
    """Have the Program Manager evaluation agent produce validated product features for the step."""
    result = program_manager_evaluation_agent.evaluate(query)
    project_plan["Program Manager"].append(result["final_response"])
    return result["final_response"]


def development_engineer_support_function(query):
    """Have the Development Engineer evaluation agent produce validated engineering tasks for the step."""
    result = development_engineer_evaluation_agent.evaluate(query)
    project_plan["Development Engineer"].append(result["final_response"])
    return result["final_response"]


# ---------------------------------------------------------------------------
# 4. MAIN WORKFLOW
# ---------------------------------------------------------------------------
print("\n*** Workflow execution started ***\n")
# Workflow Prompt
# ****
workflow_prompt = (
    "Create a complete development plan for the Email Router product. "
    "Return exactly three steps, one per line, with no headings and no extra sentences: "
    "a step to define the user stories for the Email Router, "
    "a step to define the product features for the Email Router by grouping related user stories, "
    "and a step to define the engineering development tasks for each Email Router user story."
)
# ****
print(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")

print("\nDefining workflow steps from the workflow prompt")
workflow_steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)
print("Workflow steps:")
for number, step in enumerate(workflow_steps, start=1):
    print(f"  {number}. {step}")

completed_steps = []
for number, step in enumerate(workflow_steps, start=1):
    print(f"\n{'=' * 80}\nExecuting step {number}/{len(workflow_steps)}: {step}\n{'=' * 80}")
    result = routing_agent.route(step)
    completed_steps.append(result)
    print(f"\nResult of step {number}:\n{result}")

# Print one consolidated project plan, grouped by section
sections = [
    ("USER STORIES", "Product Manager"),
    ("PRODUCT FEATURES", "Program Manager"),
    ("ENGINEERING TASKS", "Development Engineer"),
]
print("\n" + "#" * 80)
print("EMAIL ROUTER - CONSOLIDATED PROJECT PLAN")
print("#" * 80)
for title, role in sections:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")
    if project_plan[role]:
        print("\n\n".join(project_plan[role]))
    else:
        print(f"(No step was routed to the {role} in this run.)")

print("\n*** Workflow execution completed ***\n")
