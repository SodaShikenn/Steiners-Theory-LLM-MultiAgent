"""
Batch Experiment Runner

Runs Cooperative or Competitive multi-agent frameworks on a dataset of questions
without the Streamlit UI. Used to generate the experiment results reported in the paper.

Usage:
    python batch_runner.py --dataset ../data/bbh.json --name bbh --framework cooperative
    python batch_runner.py --dataset ../data/gsm8k.json --name gsm8k --framework competitive
"""

import json
import os
import argparse
from typing import List, TypedDict, Annotated
import operator
from copy import deepcopy
from datetime import datetime
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from llm_utils import chat_with_llm

load_dotenv()
output_dir = os.getenv("OUTPUT_DIR", "results")


class State(TypedDict):
    input: str
    responses: Annotated[List[List[str]], operator.add]
    details: Annotated[List, operator.add]


# --- Agent Configurations ---

COOPERATIVE_AGENTS = [
    {
        "agent_name": "teammate_1",
        "system_prompt": (
            "You are an expert in analyzing fine details. "
            "Focus on breaking down the problem and providing a response that "
            "highlights specific elements and intricacies. "
            "Provide detailed insights while keeping the response concise."
        )
    },
    {
        "agent_name": "teammate_2",
        "system_prompt": (
            "You are an expert in synthesizing information and understanding "
            "overarching themes. Focus on providing a broad perspective and "
            "summarizing the problem as a whole. "
            "Deliver a high-level overview while keeping the response concise."
        )
    },
]

COOPERATIVE_FINAL_PROMPT = (
    "You are an expert at combining answers into a single final answer. "
    "Please make the answer comprehensive and complete."
)

COMPETITIVE_AGENTS = [
    {
        "agent_name": "debater_1",
        "system_prompt": "You are affirmative side. Please express your viewpoints."
    },
    {
        "agent_name": "debater_2",
        "system_prompt": (
            "You are negative side. You disagree with the affirmative side's points. "
            "Provide your reasons and answer."
        )
    },
]


def build_competitive_final_prompt(question: str) -> str:
    return (
        f"You are a moderator. "
        f"There will be two debaters involved in a debate competition. "
        f"They will present their answers and discuss their perspectives on the {question}. "
        f"At the end of each round, you will evaluate both sides' answers and decide which one is correct."
    )


def run_multi_agent(question: str, framework: str, num_rounds: int = 2, model_name: str = "4o"):
    """Run a multi-agent framework on a single question (no UI)."""
    chat_histories = {str(i): [] for i in range(num_rounds)}

    if framework == "cooperative":
        agents_config = COOPERATIVE_AGENTS
    else:
        agents_config = COMPETITIVE_AGENTS

    all_responses = []
    all_details = []

    # Run rounds
    for round_idx in range(num_rounds):
        for agent_config in agents_config:
            agent_name = agent_config["agent_name"]
            system_prompt = agent_config["system_prompt"]

            if framework == "competitive":
                system_prompt += f" Here are the previous conversations: {all_responses}"

            response = chat_with_llm(
                system_prompt, all_responses, question, model_name=model_name
            )

            details = {
                "agent_name": agent_name,
                "round": round_idx + 1,
                "input_message": question,
                "response": response,
            }

            all_responses.append(response)
            all_details.append(details)

            history_info = f"agent: {agent_name}\nquestion: {question}\ncandidate answer: {response}"
            chat_histories[str(round_idx)].append(history_info)

            print(f"  [{agent_name} R{round_idx + 1}] done")

    # Final agent
    final_chat_history = chat_histories[str(num_rounds - 1)]
    if framework == "cooperative":
        final_prompt = COOPERATIVE_FINAL_PROMPT
    else:
        final_prompt = build_competitive_final_prompt(question)

    final_response = chat_with_llm(
        final_prompt, final_chat_history,
        f"Question: {question}, Answers: {all_responses}",
        model_name=model_name
    )

    return {
        "details": all_details,
        "final_response": final_response,
        "all_responses": all_responses,
    }


def format_result(details: list, final_response: str) -> str:
    lines = []
    for detail in details:
        lines.append(f"Agent Name: {detail['agent_name']}")
        lines.append(f"Round: {detail['round']}")
        lines.append(f"Question: {detail['input_message']}")
        lines.append(f"Response: {detail['response']}")
        lines.append("=" * 40)
    lines.append("\nFinal Response:")
    lines.append(final_response)
    return "\n".join(lines)


def run_batch(dataset_path: str, dataset_name: str, framework: str,
              num_rounds: int = 2, model_name: str = "4o"):
    """Run multi-agent framework on a batch of questions.

    Expected JSON format:
    [
        {"question": "...", "answer": "..."},
        ...
    ]
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_dir = os.path.join(output_dir, f"{dataset_name}_{framework}")
    os.makedirs(out_dir, exist_ok=True)

    for i, item in enumerate(data):
        question = item["question"]
        print(f"[{i + 1}/{len(data)}] {framework}: {question[:80]}...")

        result = run_multi_agent(question, framework, num_rounds, model_name)

        formatted = format_result(result["details"], result["final_response"])
        filename = f"result_horizontal_{framework}_{dataset_name}_q{i + 1}.txt"
        filepath = os.path.join(out_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(formatted)

    print(f"All results saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Multi-Agent Experiment Runner")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset JSON file")
    parser.add_argument("--name", type=str, required=True, help="Dataset name (bbh or gsm8k)")
    parser.add_argument("--framework", type=str, required=True,
                        choices=["cooperative", "competitive"],
                        help="MA framework to use")
    parser.add_argument("--rounds", type=int, default=2, help="Number of discussion rounds")
    parser.add_argument("--model", type=str, default="4o", help="Model name (4o or 4o-mini)")
    args = parser.parse_args()

    run_batch(args.dataset, args.name, args.framework, args.rounds, args.model)
