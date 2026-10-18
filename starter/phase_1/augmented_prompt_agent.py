from workflow_agents.base_agents import AugmentedPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"
persona = "You are a college professor; your answers always start with: 'Dear students,'"

# Instantiate an object of AugmentedPromptAgent with the required parameters
augmented_agent = AugmentedPromptAgent(openai_api_key, persona)

# Send the 'prompt' to the agent and store the response in a variable named 'augmented_agent_response'
augmented_agent_response = augmented_agent.respond(prompt)

# Print the agent's response
print(f"Prompt: {prompt}")
print(f"Persona: {persona}")
print(f"Response: {augmented_agent_response}")

# Knowledge used: the agent gets no extra knowledge, only a persona. So the fact
# "Paris is the capital of France" comes from the general knowledge that the
# gpt-3.5-turbo model learned during training.
#
# Persona impact: the system prompt tells the model to act as a college professor
# and to forget previous context. This changes the style and tone of the answer
# (it starts with "Dear students," and sounds like a teacher), but not the facts.
print()
print("Knowledge source: no extra knowledge was given to the agent, so the answer comes from "
      "the general knowledge of the gpt-3.5-turbo model.")
print("Persona impact: the system prompt made the model answer as a college professor. "
      "The answer starts with 'Dear students,' and uses a teaching tone; the facts stay the same.")
