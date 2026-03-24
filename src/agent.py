import json

from dotenv import load_dotenv

import operator
from typing import Literal
from typing_extensions import TypedDict, Annotated

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langchain.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from tools.ticker.metrics_tools import ToolResult, get_metrics_for_tickers, run_l1_screener

try:
    load_dotenv()
    model = init_chat_model(model="claude-haiku-4-5", model_provider="anthropic", temperature=0)
    tools = [get_metrics_for_tickers, run_l1_screener]
    tools_by_name = {tool.name: tool for tool in tools}
    model_with_tools = model.bind_tools(tools)
    print("Tools bound")
except Exception as initE:
    print("Init failed: ", initE)

# Model node - method to call LLM.
def llm_call(state: dict):
    return {
        'messages': [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content = 'You are a personal investor that want to invest on stocks to make money with low risks.'
                    )
                ]
                + state['messages']
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    last_tool_calls_msg_index: int
    tool_fatal_errors: dict


# Tool node - method to call tools
def tool_node(state: dict):
    result = []
    tracking = []
    tool_calls_msg_index = len(state['messages']) - 1
    for tool_call in state['messages'][-1].tool_calls:
        tool = tools_by_name[tool_call['name']]
        print(f'[tool_node] Calling tool {tool.name} with args {tool_call['args']}')
        observation = tool.invoke(tool_call)
        if isinstance(observation, ToolMessage):
            result.append(observation)
        else:
            result.append(ToolMessage(content=str(observation), tool_call_id=tool_call['id']))
        tracking.append({
            'tool_name': tool.name,
            'tool_args': tool_call['args']
        })

    return {
        'messages': result,
        'last_tool_calls_msg_index': tool_calls_msg_index
    }

# Interrupt
def human_clarification(state: MessagesState):
    errors = state["tool_fatal_errors"]
    response = interrupt(
        f"Can't proceed with your query for the following reasons: {errors} "
        "Are these correct? Some exchanges need a suffix (e.g. 'CBA.AX' for ASX). "
        "Please confirm or provide corrected tickers."
    )
    return {
        "messages": [HumanMessage(content=response)],
        "tool_fatal_errors": []
    }


def combine_tool_fatal_errors(state: MessagesState):
    try:
        last_tool_calls = state['messages'][state['last_tool_calls_msg_index']].tool_calls
    except Exception as e:
        print(f'Couldn\'t get tool calls from last AI message with index {state['last_tool_calls_msg_index']}')
        print('Skip tool results validation')
        return { 'tool_fatal_errors': [] }
    
    tool_fatal_errors = []
    for i in range(len(last_tool_calls)):
        tool_message: ToolMessage = state['messages'][-1 - i]
        tool_result: ToolResult = tool_message.artifact
        if len(tool_result['errors_fatal']) > 0:
            tool_fatal_errors = tool_fatal_errors + tool_result['errors_fatal']

    return {
        'tool_fatal_errors': tool_fatal_errors
    }


# Edge logic
def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    messages = state['messages']
    last_message = messages[-1]

    if last_message.tool_calls:
        return 'tool_node'

    return END

def validate_tool_fatal_errors(state: MessagesState) -> Literal['human_clarification', 'llm_call']:
    if len(state['tool_fatal_errors']) > 0:
        return 'human_clarification'
    
    return 'llm_call'

agent_builder = StateGraph(MessagesState)

agent_builder.add_node('llm_call', llm_call)
agent_builder.add_node('tool_node', tool_node)
agent_builder.add_node('human_clarification', human_clarification)
agent_builder.add_node('tool_error_validation', combine_tool_fatal_errors)

agent_builder.add_edge(START, 'llm_call')
agent_builder.add_conditional_edges(
    'llm_call',
    should_continue,
    ['tool_node', END]
)
agent_builder.add_edge('tool_node', 'tool_error_validation')

# agent_builder.add_edge('tool_node', 'llm_call')
agent_builder.add_conditional_edges(
    'tool_error_validation',
    validate_tool_fatal_errors,
    ['human_clarification', 'llm_call']
)

agent_builder.add_edge('human_clarification', 'llm_call')

agent = agent_builder.compile()
