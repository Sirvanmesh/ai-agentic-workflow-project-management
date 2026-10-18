# Test script for DirectPromptAgent class

from workflow_agents.base_agents import DirectPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load the OpenAI API key from the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the Capital of France?"

# Instantiate the DirectPromptAgent as direct_agent
direct_agent = DirectPromptAgent(openai_api_key)
# Use direct_agent to send the prompt defined above and store the response
direct_agent_response = direct_agent.respond(prompt)

# Print the response from the agent
print(f"Prompt: {prompt}")
print(f"Response: {direct_agent_response}")

# Explain the knowledge source used by the agent to generate the response
print()
print("Knowledge source: the DirectPromptAgent sends the prompt as a single user message, "
      "with no system prompt, persona or extra knowledge. The answer therefore comes only from "
      "the general knowledge the gpt-3.5-turbo model learned during its training.")
