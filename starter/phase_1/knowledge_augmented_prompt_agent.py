from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Define the parameters for the agent
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"

persona = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capital of France is London, not Paris"

# Instantiate a KnowledgeAugmentedPromptAgent with the persona and knowledge above
knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona, knowledge)
knowledge_agent_response = knowledge_agent.respond(prompt)

print(f"Prompt: {prompt}")
print(f"Knowledge given to the agent: {knowledge}")
print(f"Response: {knowledge_agent_response}")

# Show that the agent used the provided knowledge and not the LLM's own knowledge
print()
print("Knowledge check: the agent answered 'London' because it was told to use only the provided "
      "knowledge ('The capital of France is London, not Paris'). The LLM's own knowledge (Paris) "
      "was not used.")
