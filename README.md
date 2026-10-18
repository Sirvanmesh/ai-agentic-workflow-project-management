# AI-Powered Agentic Workflow for Project Management

Udacity Agentic AI project. It has two parts:

- **Phase 1 – Agent library:** a reusable Python package (`workflow_agents/base_agents.py`) with seven agents, each with its own test script:
  `DirectPromptAgent`, `AugmentedPromptAgent`, `KnowledgeAugmentedPromptAgent`, `RAGKnowledgePromptAgent` (provided), `EvaluationAgent`, `RoutingAgent` and `ActionPlanningAgent`.
- **Phase 2 – Project-management workflow:** `agentic_workflow.py` takes the Email Router product spec and a high-level prompt from a technical project manager. The `ActionPlanningAgent` breaks the prompt into steps. The `RoutingAgent` sends each step to a Product Manager, Program Manager or Development Engineer agent. Each of these agents is a `KnowledgeAugmentedPromptAgent` checked by an `EvaluationAgent`. The script prints one consolidated project plan with **user stories**, **product features** and **engineering tasks**.

## Folder structure

```
.
├── requirements.txt
├── project_overview.md
└── starter/
    ├── phase_1/
    │   ├── workflow_agents/
    │   │   ├── __init__.py
    │   │   └── base_agents.py          # the 7 agent classes
    │   ├── direct_prompt_agent.py
    │   ├── augmented_prompt_agent.py
    │   ├── knowledge_augmented_prompt_agent.py
    │   ├── rag_knowledge_prompt_agent.py
    │   ├── evaluation_agent.py
    │   ├── routing_agent.py
    │   ├── action_planning_agent.py
    │   └── outputs/                    # terminal output of each test script
    └── phase_2/
        ├── workflow_agents/            # copy of the Phase 1 package
        ├── agentic_workflow.py
        ├── Product-Spec-Email-Router.txt
        └── outputs/
            └── agentic_workflow_output.txt
```

## Setup

1. Create and activate a Python 3.11 virtual environment in the root folder:
   ```bash
   py -3.11 -m venv .venv
   .venv\Scripts\activate          # Windows
   source .venv/bin/activate       # macOS / Linux
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in `starter/phase_1/` and another in `starter/phase_2/`, each with this one line:
   ```
   OPENAI_API_KEY=your_vocareum_openai_key
   ```
   The `.env` files are ignored by git, so never commit them.

All agents use the Vocareum OpenAI endpoint, `base_url="https://openai.vocareum.com/v1"`. The API key is passed to each agent when it is created and is never hardcoded.

## How to run

Phase 1 test scripts (run from `starter/phase_1`):

```bash
cd starter/phase_1
python direct_prompt_agent.py
python augmented_prompt_agent.py
python knowledge_augmented_prompt_agent.py
python rag_knowledge_prompt_agent.py
python evaluation_agent.py
python routing_agent.py
python action_planning_agent.py
```

Phase 2 workflow (run from `starter/phase_2`):

```bash
cd starter/phase_2
python agentic_workflow.py
```

To save the output to a file, for example:

```bash
python agentic_workflow.py 2>&1 | tee outputs/agentic_workflow_output.txt
```

On Windows, set `PYTHONUTF8=1` first so that emoji in the output print correctly when the output is piped.

## Outputs

- `starter/phase_1/outputs/` – one text file for each of the seven test scripts, showing the prompt and the agent's response.
- `starter/phase_2/outputs/agentic_workflow_output.txt` – the full workflow run. It ends with the consolidated Email Router project plan in three sections: USER STORIES, PRODUCT FEATURES and ENGINEERING TASKS.
