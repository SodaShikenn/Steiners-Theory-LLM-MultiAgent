"""
Evaluation Metrics based on Steiner's Theory

Uses GPT-4o to automatically evaluate multi-agent conversation logs against
Steiner's Productivity Losses and Productivity Gains metrics.

Metrics (from Appendix B of the paper):
  Productivity Losses:
    - choosing_wrong_ideas (Wrong Selection): Agent selects incorrect idea from others
    - generating_wrong_ideas (Wrong Generation): Number of incorrect messages
    - same_idea (Same Opinion): Repeated ideas across agents

  Productivity Gains:
    - correction: Derived as (total_opinions - generating_wrong_ideas) if errors exist
    - novel_idea: New ideas not present in single-agent response

Usage:
    python evaluate_metrics.py --ma_dir ../results/bbh_cooperative \
                               --sa_dir ../results/bbh_single_agent \
                               --dataset ../data/bbh.json \
                               --output ../results/evaluation/bbh_cooperative_metrics.json
"""

import json
import os
import re
import argparse
from dotenv import load_dotenv
from llm_utils import chat_with_llm

load_dotenv()

# Total number of agent opinions per question (2 agents x 2 rounds)
TOTAL_OPINIONS = 4

EVALUATION_PROMPT = """From the multi-agent conversation in the attachment, answer the following.
The conversation is as follows:
answer from agent_1 -> answer from agent_2 -> answer from agent_1 (2nd time) -> answer from agent_2 (2nd time) -> final answer by judge

- is_correct: If the multi-agent's final answer is correct, please give me a 1; if it is wrong, please give me a 0.
- choosing_wrong_ideas: If you had the correct answer during the conversation, but the final result chose a different idea than the correct one, give 1. If no one was able to give the correct answer, give 0. Always give 0 if the answer is correct.
- generating_wrong_ideas: Please tell us the number of messages (excluding the last judge) that were incorrect in comparison to the correct answer.
- same_idea: Look at every agent's conclusion. Please count the number of times where the agent reached the same idea as the previous agent (compare only with the last agent). Count 1 if they are the same, otherwise 0. (excluding the last judge) If all the agents reached the same conclusion, the count is 3. If every agent reached a different conclusion, the count is 0.
- single_agent_correct: If the single-agent is correct, set the number to 1; otherwise, set the number to 0. Please include the number of times the correct answer was obtained after the wrong answer was given.

Please respond ONLY in the following JSON format:
{{"is_correct": <0 or 1>, "choosing_wrong_ideas": <0 or 1>, "generating_wrong_ideas": <0-4>, "same_idea": <0-3>, "single_agent_correct": <0 or 1>}}

###Question###
{query}

###Conversation History of multi-agent###
{multi_agent_answer}

###Right answer###
{right_answer}

###Answer by a single-agent###
{single_agent_answer}"""


def parse_llm_metrics(response_text: str) -> dict:
    """Parse the LLM's JSON response into a metrics dictionary."""
    # Try to extract JSON from the response
    json_match = re.search(r'\{[^}]+\}', response_text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: try parsing the whole response
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        print(f"WARNING: Could not parse LLM response: {response_text[:200]}")
        return None


def compute_derived_metrics(raw_metrics: dict) -> dict:
    """Compute correction and novel_idea from raw LLM metrics.

    Based on Section 4.3 of the paper:
    - correction = (TOTAL_OPINIONS - generating_wrong_ideas) if generating_wrong_ideas > 0, else 0
    - novel_idea = generating_wrong_ideas if is_correct else (TOTAL_OPINIONS - generating_wrong_ideas)
    """
    metrics = dict(raw_metrics)
    gen_wrong = metrics.get("generating_wrong_ideas", 0)

    # Correction: how many correct opinions appeared when errors existed
    if gen_wrong > 0:
        metrics["correction"] = TOTAL_OPINIONS - gen_wrong
    else:
        metrics["correction"] = 0

    # Novel idea: new ideas not in single-agent response
    if metrics.get("is_correct", 0) == 1:
        metrics["novel_idea"] = gen_wrong  # wrong ideas that got overridden by novel correct ones
    else:
        metrics["novel_idea"] = TOTAL_OPINIONS - gen_wrong

    return metrics


def evaluate_single_question(
    question: str,
    right_answer: str,
    ma_conversation: str,
    sa_answer: str,
    model_name: str = "4o"
) -> dict:
    """Evaluate a single question's multi-agent conversation."""
    prompt = EVALUATION_PROMPT.format(
        query=question,
        multi_agent_answer=ma_conversation,
        right_answer=right_answer,
        single_agent_answer=sa_answer,
    )

    response = chat_with_llm(
        "You are an evaluation assistant. Analyze the conversation and return metrics in JSON format.",
        [],
        prompt,
        model_name=model_name,
    )

    raw_metrics = parse_llm_metrics(response)
    if raw_metrics is None:
        return None

    return compute_derived_metrics(raw_metrics)


def evaluate_batch(
    ma_dir: str,
    sa_dir: str,
    dataset_path: str,
    output_path: str,
    model_name: str = "4o"
):
    """Evaluate all questions in a dataset."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_metrics = []

    for i, item in enumerate(data):
        question = item["question"]
        right_answer = item.get("answer", "")
        q_id = i + 1

        # Read multi-agent conversation log
        ma_files = [f for f in os.listdir(ma_dir) if f"q{q_id}.txt" in f or f"q{q_id}_" in f]
        if not ma_files:
            print(f"  WARNING: No MA result found for q{q_id}, skipping")
            continue
        with open(os.path.join(ma_dir, ma_files[0]), "r", encoding="utf-8") as f:
            ma_conversation = f.read()

        # Read single-agent response
        sa_files = [f for f in os.listdir(sa_dir) if f"q{q_id}.txt" in f or f"q{q_id}_" in f]
        if sa_files:
            with open(os.path.join(sa_dir, sa_files[0]), "r", encoding="utf-8") as f:
                sa_answer = f.read()
        else:
            sa_answer = "(single-agent result not available)"

        print(f"[{q_id}/{len(data)}] Evaluating: {question[:60]}...")
        metrics = evaluate_single_question(
            question, right_answer, ma_conversation, sa_answer, model_name
        )

        if metrics:
            metrics["question_id"] = q_id
            metrics["question"] = question
            all_metrics.append(metrics)
            print(f"  -> is_correct={metrics['is_correct']}, "
                  f"wrong_sel={metrics['choosing_wrong_ideas']}, "
                  f"wrong_gen={metrics['generating_wrong_ideas']}, "
                  f"same={metrics['same_idea']}, "
                  f"correction={metrics['correction']}, "
                  f"novel={metrics['novel_idea']}")

    # Save results
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    print(f"\nEvaluation complete. {len(all_metrics)} questions evaluated.")
    print(f"Results saved to {output_path}")
    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MA metrics based on Steiner's theory")
    parser.add_argument("--ma_dir", type=str, required=True,
                        help="Directory containing multi-agent result files")
    parser.add_argument("--sa_dir", type=str, required=True,
                        help="Directory containing single-agent result files")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to dataset JSON file")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for evaluation metrics JSON")
    parser.add_argument("--model", type=str, default="4o",
                        help="Model for evaluation (default: 4o)")
    args = parser.parse_args()

    evaluate_batch(args.ma_dir, args.sa_dir, args.dataset, args.output, args.model)
