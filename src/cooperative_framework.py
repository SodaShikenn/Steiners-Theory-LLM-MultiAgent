"""
Cooperative Multi-Agent Framework

Based on Du et al. (2023) - Improving factuality and reasoning in language
models through multiagent debate.

Two agents collaborate across multiple rounds. Each agent's response is
incorporated into the next round's input for all agents. A final agent
combines the answers into a single response.

Agent roles:
  - Teammate 1: Expert in analyzing fine details
  - Teammate 2: Expert in synthesizing information and summarizing
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
output_dir = os.getenv("OUTPUT_DIR", "results/cooperative")


class State(TypedDict):
    input: str
    responses: Annotated[List[List[str]], operator.add]
    details: Annotated[List, operator.add]


def expert_agent_func(state: State, agent_config, round_index: int):
    """Process a single agent's turn in the cooperative discussion."""
    agent_name = agent_config.get("agent_name", "default_agent")
    system_prompt = agent_config.get("system_prompt", "You are a helpful assistant.")
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
    """Combine all agent responses into a single final answer."""
    global over_all_response
    input_message = state.get("input")
    responses = state.get("responses", [])
    flat_responses = [sublist for sublist in responses]

    final_system_prompt = (
        "You are an expert at combining answers into a single final answer. "
        "Please make the answer comprehensive and complete."
    )

    final_chat_history = chat_histories[str(num_rounds - 1)]
    final_response = chat_with_llm(
        final_system_prompt, final_chat_history,
        f"Question: {input_message}, Answers: {flat_responses}"
    )
    over_all_response = final_response
    return {"output": final_response, "responses": responses, "chat_histories": chat_histories}


def create_graph(num_rounds=2):
    """Build the cooperative agent interaction graph."""
    workflow = StateGraph(State)
    expert_agents_config = [
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
        workflow.add_edge(f"teammate_1_{i + 1}", f"teammate_2_{i + 2}")
        workflow.add_edge(f"teammate_2_{i + 1}", f"teammate_1_{i + 2}")

    workflow.add_edge(f"teammate_1_{num_rounds}", "final_agent")
    workflow.add_edge(f"teammate_2_{num_rounds}", "final_agent")
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
    st.title("Cooperative Multi-Agent Framework")

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
        filename = f"result_cooperative_{timestamp}.txt"
        save_to_file(formatted_result, output_dir, filename)
        st.success(f"Results saved to {filename}")
