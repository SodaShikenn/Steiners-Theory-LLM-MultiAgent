"""
Utility functions for interacting with OpenAI LLM models.
Supports GPT-4o and GPT-4o-mini via LangChain.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

MODEL_4o_API_KEY = os.getenv("MODEL_4o_API_KEY")

MODEL_CONFIG = {
    "4o": {
        "api_key": MODEL_4o_API_KEY,
        "model_name": "gpt-4o"
    },
    "4o-mini": {
        "api_key": MODEL_4o_API_KEY,
        "model_name": "gpt-4o-mini"
    }
}


def get_llm(model_name):
    """Get a ChatOpenAI instance by model name."""
    config = MODEL_CONFIG.get(model_name)
    if not config:
        raise ValueError(f"Invalid model name: {model_name}")
    return ChatOpenAI(
        api_key=config["api_key"],
        model=config["model_name"],
    )


def chat_with_llm(system_prompt, chat_history, user_prompt, model_name="4o"):
    """Send a message to the LLM with system prompt, chat history, and user prompt."""
    llm = get_llm(model_name)
    messages = [
        ("system", system_prompt),
        ("human", str(chat_history)),
        ("human", user_prompt),
    ]
    response = llm.invoke(messages)
    return response.content


if __name__ == "__main__":
    system_prompt = "You are a helpful assistant."
    user_prompt = "What is the capital of France?"
    response = chat_with_llm(system_prompt, [], user_prompt, model_name="4o")
    print(f"GPT-4o Response:\n{response}")
