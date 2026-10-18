"""
Reusable agent library for agentic workflows.

Every agent receives its OpenAI API key through its constructor (the key is
never hardcoded here). All clients point to the Vocareum OpenAI proxy.
"""
from openai import OpenAI
import numpy as np
import pandas as pd
import re
import csv
import uuid
from datetime import datetime

# Base URL of the Vocareum OpenAI proxy used by every client in this module
VOCAREUM_BASE_URL = "https://openai.vocareum.com/v1"


# DirectPromptAgent class definition
class DirectPromptAgent:
    """Agent that sends the user prompt straight to the LLM, with no system prompt."""

    def __init__(self, openai_api_key):
        """Store the OpenAI API key used for every call made by this agent."""
        self.openai_api_key = openai_api_key

    def respond(self, prompt):
        """Send the prompt as a single user message and return only the response text."""
        client = OpenAI(base_url=VOCAREUM_BASE_URL, api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content


# AugmentedPromptAgent class definition
class AugmentedPromptAgent:
    """Agent that answers prompts while taking on a predefined persona."""

    def __init__(self, openai_api_key, persona):
        """Initialize the agent with given attributes."""
        self.persona = persona
        self.openai_api_key = openai_api_key

    def respond(self, input_text):
        """Generate a response in the agent's persona and return only the response text."""
        client = OpenAI(base_url=VOCAREUM_BASE_URL, api_key=self.openai_api_key)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"{self.persona}. Assume this persona when you answer. "
                               "Forget all previous conversational context."
                },
                {"role": "user", "content": input_text}
            ],
            temperature=0
        )

        return response.choices[0].message.content


# KnowledgeAugmentedPromptAgent class definition
class KnowledgeAugmentedPromptAgent:
    """Agent that answers only from the knowledge it is given, in a defined persona."""

    def __init__(self, openai_api_key, persona, knowledge):
        """Initialize the agent with provided attributes."""
        self.persona = persona
        self.knowledge = knowledge
        self.openai_api_key = openai_api_key

    def respond(self, input_text):
        """Generate a response based only on the provided knowledge and return the text."""
        client = OpenAI(base_url=VOCAREUM_BASE_URL, api_key=self.openai_api_key)
        system_message = (
            f"You are {self.persona} knowledge-based assistant. Forget all previous context.\n"
            f"Use only the following knowledge to answer, do not use your own knowledge: {self.knowledge}\n"
            "Answer the prompt based on this knowledge, not your own."
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": input_text}
            ],
            temperature=0
        )
        return response.choices[0].message.content


# RAGKnowledgePromptAgent class definition
class RAGKnowledgePromptAgent:
    """
    An agent that uses Retrieval-Augmented Generation (RAG) to find knowledge from a large corpus
    and leverages embeddings to respond to prompts based solely on retrieved information.
    """

    def __init__(self, openai_api_key, persona, chunk_size=2000, chunk_overlap=100):
        """
        Initializes the RAGKnowledgePromptAgent with API credentials and configuration settings.

        Parameters:
        openai_api_key (str): API key for accessing OpenAI.
        persona (str): Persona description for the agent.
        chunk_size (int): The size of text chunks for embedding. Defaults to 2000.
        chunk_overlap (int): Overlap between consecutive chunks. Defaults to 100.
        """
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key
        self.unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"

    def get_embedding(self, text):
        """
        Fetches the embedding vector for given text using OpenAI's embedding API.

        Parameters:
        text (str): Text to embed.

        Returns:
        list: The embedding vector.
        """
        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding

    def calculate_similarity(self, vector_one, vector_two):
        """
        Calculates cosine similarity between two vectors.

        Parameters:
        vector_one (list): First embedding vector.
        vector_two (list): Second embedding vector.

        Returns:
        float: Cosine similarity between vectors.
        """
        vec1, vec2 = np.array(vector_one), np.array(vector_two)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def chunk_text(self, text):
        """
        Splits text into manageable chunks, attempting natural breaks.

        Parameters:
        text (str): Text to split into chunks.

        Returns:
        list: List of dictionaries containing chunk metadata.
        """
        separator = "\n"
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) <= self.chunk_size:
            return [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]

        chunks, start, chunk_id = [], 0, 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if separator in text[start:end]:
                end = start + text[start:end].rindex(separator) + len(separator)

            chunks.append({
                "chunk_id": chunk_id,
                "text": text[start:end],
                "chunk_size": end - start,
                "start_char": start,
                "end_char": end
            })

            # Minimal fix to the provided code: stop once the last chunk reaches the
            # end of the text. Without this, "start = end - overlap" stays below
            # len(text) forever and the loop never ends.
            if end >= len(text):
                break

            start = end - self.chunk_overlap
            chunk_id += 1

        with open(f"chunks-{self.unique_filename}", 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
            writer.writeheader()
            for chunk in chunks:
                writer.writerow({k: chunk[k] for k in ["text", "chunk_size"]})

        return chunks

    def calculate_embeddings(self):
        """
        Calculates embeddings for each chunk and stores them in a CSV file.

        Returns:
        DataFrame: DataFrame containing text chunks and their embeddings.
        """
        df = pd.read_csv(f"chunks-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['text'].apply(self.get_embedding)
        df.to_csv(f"embeddings-{self.unique_filename}", encoding='utf-8', index=False)
        return df

    def find_prompt_in_knowledge(self, prompt):
        """
        Finds and responds to a prompt based on similarity with embedded knowledge.

        Parameters:
        prompt (str): User input prompt.

        Returns:
        str: Response derived from the most similar chunk in knowledge.
        """
        prompt_embedding = self.get_embedding(prompt)
        df = pd.read_csv(f"embeddings-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['embeddings'].apply(lambda x: np.array(eval(x)))
        df['similarity'] = df['embeddings'].apply(lambda emb: self.calculate_similarity(prompt_embedding, emb))

        best_chunk = df.loc[df['similarity'].idxmax(), 'text']

        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are {self.persona}, a knowledge-based assistant. Forget previous context."},
                {"role": "user", "content": f"Answer based only on this information: {best_chunk}. Prompt: {prompt}"}
            ],
            temperature=0
        )

        return response.choices[0].message.content


class EvaluationAgent:
    """
    Agent that checks a worker agent's answers against evaluation criteria and,
    when an answer fails, sends correction instructions back to the worker.
    """

    def __init__(self, openai_api_key, persona, evaluation_criteria, worker_agent, max_interactions):
        """Initialize the EvaluationAgent with given attributes."""
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.worker_agent = worker_agent  # the agent to evaluate
        self.max_interactions = max_interactions

    def evaluate(self, initial_prompt):
        """
        Run the worker/evaluator loop for up to max_interactions rounds.

        Each round: the worker answers, the evaluator judges the answer against the
        criteria (temperature=0), and if the verdict is not "yes" a second LLM call
        (temperature=0) writes correction instructions for the next round.

        Returns:
        dict: {"final_response": str, "evaluation": str, "iterations": int}
        """
        client = OpenAI(base_url=VOCAREUM_BASE_URL, api_key=self.openai_api_key)
        prompt_to_evaluate = initial_prompt
        response_from_worker = ""
        evaluation = ""
        iterations = 0

        for i in range(self.max_interactions):
            iterations = i + 1
            print(f"\n--- Interaction {i+1} ---")

            print(" Step 1: Worker agent generates a response to the prompt")
            print(f"Prompt:\n{prompt_to_evaluate}")
            response_from_worker = self.worker_agent.respond(prompt_to_evaluate)
            print(f"Worker Agent Response:\n{response_from_worker}")

            print(" Step 2: Evaluator agent judges the response")
            eval_prompt = (
                f"Does the following answer: {response_from_worker}\n"
                f"Meet this criteria: {self.evaluation_criteria}\n"
                f"Respond Yes or No, and the reason why it does or doesn't meet the criteria."
            )
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.persona},
                    {"role": "user", "content": eval_prompt}
                ],
                temperature=0
            )
            evaluation = response.choices[0].message.content.strip()
            print(f"Evaluator Agent Evaluation:\n{evaluation}")

            print(" Step 3: Check if evaluation is positive")
            # Ignore leading markdown symbols such as "**Yes**" before checking the verdict
            if evaluation.lower().lstrip("*# ").startswith("yes"):
                print("✅ Final solution accepted.")
                break
            else:
                print(" Step 4: Generate instructions to correct the response")
                instruction_prompt = (
                    f"Provide instructions to fix an answer based on these reasons why it is incorrect: {evaluation}"
                )
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": self.persona},
                        {"role": "user", "content": instruction_prompt}
                    ],
                    temperature=0
                )
                instructions = response.choices[0].message.content.strip()
                print(f"Instructions to fix:\n{instructions}")

                print(" Step 5: Send feedback to worker agent for refinement")
                prompt_to_evaluate = (
                    f"The original prompt was: {initial_prompt}\n"
                    f"The response to that prompt was: {response_from_worker}\n"
                    f"It has been evaluated as incorrect.\n"
                    f"Make only these corrections, do not alter content validity: {instructions}\n"
                    # Ask for the whole answer, so earlier correct parts are not dropped
                    f"Return the full corrected answer, not only the changed parts."
                )
        return {
            "final_response": response_from_worker,
            "evaluation": evaluation,
            "iterations": iterations
        }


class RoutingAgent():
    """
    Agent that sends a prompt to the best-matching agent, chosen by cosine
    similarity between the prompt embedding and each agent description embedding.
    """

    def __init__(self, openai_api_key, agents):
        """Initialize the agent with given attributes."""
        self.openai_api_key = openai_api_key
        # List of dicts, each with "name", "description" and "func" keys
        self.agents = agents

    def get_embedding(self, text):
        """Return the text-embedding-3-large embedding vector for the given text."""
        client = OpenAI(base_url=VOCAREUM_BASE_URL, api_key=self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        # Extract and return the embedding vector from the response
        embedding = response.data[0].embedding
        return embedding

    def route(self, user_input):
        """
        Pick the agent whose description is most similar to user_input and
        return the result of calling that agent's func with user_input.
        """
        input_emb = self.get_embedding(user_input)
        best_agent = None
        best_score = -1

        for agent in self.agents:
            agent_emb = self.get_embedding(agent["description"])
            if agent_emb is None:
                continue

            similarity = np.dot(input_emb, agent_emb) / (np.linalg.norm(input_emb) * np.linalg.norm(agent_emb))
            print(f"[Router] {agent['name']}: similarity={similarity:.3f}")

            # Keep the agent with the highest similarity score seen so far
            if similarity > best_score:
                best_score = similarity
                best_agent = agent

        if best_agent is None:
            return "Sorry, no suitable agent could be selected."

        print(f"[Router] Best agent: {best_agent['name']} (score={best_score:.3f})")
        return best_agent["func"](user_input)


class ActionPlanningAgent:
    """Agent that extracts, from a user prompt, the list of steps needed to complete an action."""

    def __init__(self, openai_api_key, knowledge):
        """Store the API key and the knowledge used to plan steps, and create the OpenAI client."""
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge
        self.client = OpenAI(base_url=VOCAREUM_BASE_URL, api_key=self.openai_api_key)

    def extract_steps_from_prompt(self, prompt):
        """Ask the LLM for the steps needed by the prompt and return them as a clean Python list."""
        system_prompt = (
            "You are an action planning agent. Using your knowledge, you extract from the user prompt "
            "the steps requested to complete the action the user is asking for. You return the steps as a list. "
            "Only return the steps in your knowledge. Forget any previous context. "
            f"This is your knowledge: {self.knowledge}"
        )
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        response_text = response.choices[0].message.content

        # Clean the steps: strip whitespace, list markers and numbering, and drop
        # empty lines, markdown headings and intro lines such as "Here are the steps:"
        steps = []
        for line in response_text.split("\n"):
            step = line.strip()
            step = re.sub(r"^(\d+[\.\)]|[-*•])\s*", "", step).strip()
            if not step or step.startswith("#"):
                continue
            if step.lower().startswith(("here are", "here is", "steps:", "the steps")) and step.endswith(":"):
                continue
            steps.append(step)

        return steps
