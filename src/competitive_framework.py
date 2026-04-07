"""
Competitive Multi-Agent Framework (Debate)

Based on Liang et al. (2024) - Encouraging divergent thinking in large
language models through multi-agent debate.

Two agents debate from opposing viewpoints across multiple rounds.
A judge agent evaluates both sides and produces the final answer.

Agent roles:
  - Debater 1 (Affirmative): Argues in favor
  - Debater 2 (Negative): Argues against
  - Judge: Neutral arbitrator who decides the final answer
"""

from typing import List, TypedDict, Annotated, Dict
import operator
from langgraph.graph import StateGraph, END
from llm_utils import chat_with_llm
import streamlit as st
from copy import deepcopy
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()
over_all_response = None
output_dir = os.getenv("OUTPUT_DIR", "results/competitive")


class State(TypedDict):
    input: str
    responses: Annotated[List[List[str]], operator.add]
    details: Annotated[List, operator.add]


def expert_agent_func(state: State, agent_config, round_index: int):
    """Process a single debater's turn in the competitive discussion."""
    agent_name = agent_config.get("agent_name", "default_agent")
    system_prompt = agent_config.get("system_prompt", "You are a helpful assistant.")
    system_prompt += f" Here are the previous conversations: {state.get('responses')}"
    input_message = state.get("input")

    if not input_message:
        raise ValueError("Question is required")

    response = chat_with_llm(system_prompt, state.get('responses'), input_message)

    details = {
        "agent_name": agent_name,
        "history": state.get('responses'),
        "input_message": input_message,
        "response": response,
    }

    print(f"agent: {agent_name}\nquestion: {input_message}\ncandidate answer: {response}")
    history_info = deepcopy(f"agent: {agent_name}\nquestion: {input_message}\ncandidate answer: {response}")

    if str(round_index) in chat_histories:
        chat_histories[str(round_index)].append(history_info)

    return {
        "responses": [response],
        "details": [details]
    }


def final_agent_func(state: State):
    """Judge agent evaluates both sides and produces the final answer."""
    global over_all_response
    input_message = state.get("input")
    responses = state.get("responses", [])
    flat_responses = [sublist for sublist in responses]

    final_system_prompt = f"""
You are a moderator.
There will be two debaters involved in a debate competition.
They will present their answers and discuss their perspectives on the {input_message}.
At the end of each round, you will evaluate both sides' answers and decide which one is correct.
    """

    final_chat_history = chat_histories[str(num_rounds - 1)]
    final_response = chat_with_llm(
        final_system_prompt, final_chat_history,
        f"Question: {input_message}, Answers: {flat_responses}"
    )
    over_all_response = final_response
    return {"output": final_response, "responses": responses, "chat_histories": chat_histories}


def create_graph(num_rounds=2):
    """Build the competitive (debate) agent interaction graph."""
    workflow = StateGraph(State)
    expert_agents_config = [
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

    def create_agent_node(agent_config, round_index):
        def _wrapper(state):
            return expert_agent_func(state, agent_config, round_index)
        return _wrapper

    entry_points = []
    for i in range(num_rounds):
        for agent_config in expert_agents_config:
            node_name = f"{agent_config['agent_name']}_{i + 1}"
            workflow.add_node(node_name, create_agent_node(agent_config, i))
            if i == 0:
                entry_points.append(node_name)

    workflow.add_node("final_agent", final_agent_func)

    for i in range(num_rounds - 1):
        workflow.add_edge(f"debater_1_{i + 1}", f"debater_2_{i + 2}")
        workflow.add_edge(f"debater_2_{i + 1}", f"debater_1_{i + 2}")

    workflow.add_edge(f"debater_1_{num_rounds}", "final_agent")
    workflow.add_edge(f"debater_2_{num_rounds}", "final_agent")
    workflow.add_edge("final_agent", END)

    for entry_point in entry_points:
        workflow.set_entry_point(entry_point)

    return workflow.compile()


def save_to_file(content: str, directory: str, filename: str) -> None:
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, filename)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)


def format_result(details: List[Dict], final_response: str) -> str:
    lines = []
    for detail in details:
        lines.append(f"Agent Name: {detail['agent_name']}")
        lines.append(f"Question: {detail['input_message']}")
        lines.append(f"Response: {detail['response']}")
        lines.append(f"History: {detail['history']}")
        lines.append("=" * 40)
    lines.append("\nFinal Response:")
    lines.append(final_response)
    return "\n".join(lines)


if __name__ == "__main__":
    st.title("Competitive Multi-Agent Framework (Debate)")

    num_rounds = st.slider("Select number of rounds", 1, 5, 2)
    app = create_graph(num_rounds)
    chat_histories = {}
    for i in range(num_rounds):
        chat_histories[str(i)] = []

    input_message = st.text_input("Enter your question:")

    if st.button("Run Workflow"):
        inputs = {"input": input_message}
        with st.spinner("Processing..."):
            result = app.invoke(inputs)

        all_details = result["details"]
        for detail in all_details:
            agent_name = detail["agent_name"]
            with st.expander(f"{agent_name.capitalize()} Details"):
                st.write(f"**Question:** {detail['input_message']}")
                st.write(f"**Response:** {detail['response']}")
                st.write(f"**History:** {detail['history']}")

        st.markdown("## Final Result")
        st.write(over_all_response)

        formatted_result = format_result(all_details, over_all_response)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"result_competitive_{timestamp}.txt"
        save_to_file(formatted_result, output_dir, filename)
        st.success(f"Results saved to {filename}")
