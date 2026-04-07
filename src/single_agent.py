"""
Single-Agent (SA) Baseline

Runs a single GPT-4o agent on a question without multi-agent interaction.
Used as the baseline to compare against Cooperative and Competitive MA frameworks.
"""

import json
import os
import argparse
from datetime import datetime
from dotenv import load_dotenv
from llm_utils import chat_with_llm

load_dotenv()
output_dir = os.getenv("OUTPUT_DIR", "results")


def run_single_agent(question: str, model_name: str = "4o") -> str:
    """Run a single agent on one question."""
    system_prompt = (
        "You are a helpful assistant. Please answer the following question. "
        "Put your final answer at the end of your response."
    )
    response = chat_with_llm(system_prompt, [], question, model_name=model_name)
    return response


def run_batch(dataset_path: str, dataset_name: str, model_name: str = "4o"):
    """Run single-agent baseline on a batch of questions from a JSON file.

    Expected JSON format:
    [
        {"question": "...", "answer": "..."},
        ...
    ]
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_dir = os.path.join(output_dir, f"{dataset_name}_single_agent")
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for i, item in enumerate(data):
        question = item["question"]
        answer = item.get("answer", "")

        print(f"[{i + 1}/{len(data)}] Processing: {question[:80]}...")
        response = run_single_agent(question, model_name=model_name)

        result = {
            "question_id": i + 1,
            "question": question,
            "ground_truth": answer,
            "single_agent_response": response,
        }
        results.append(result)

        # Save individual result
        filename = f"sa_{dataset_name}_q{i + 1}.txt"
        with open(os.path.join(out_dir, filename), "w", encoding="utf-8") as f:
            f.write(f"Question: {question}\n")
            f.write(f"Ground Truth: {answer}\n")
            f.write(f"{'=' * 40}\n")
            f.write(f"Single Agent Response:\n{response}\n")

    # Save all results as JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(out_dir, f"sa_{dataset_name}_all_{timestamp}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Results saved to {out_dir}/")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-Agent Baseline")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset JSON file")
    parser.add_argument("--name", type=str, required=True, help="Dataset name (bbh or gsm8k)")
    parser.add_argument("--model", type=str, default="4o", help="Model name (4o or 4o-mini)")
    args = parser.parse_args()

    run_batch(args.dataset, args.name, args.model)
