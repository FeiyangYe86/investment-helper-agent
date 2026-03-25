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
from tools.ticker.macro_tools import get_macro_context
from tools.ticker.announcements_tools import get_asx_announcements
from tools.ticker.short_interest_tools import get_short_interest
from tools.ticker.news_sentiment_tools import get_news_sentiment

try:
    load_dotenv()
    model = init_chat_model(model="claude-haiku-4-5", model_provider="anthropic", temperature=0)
    tools = [
        get_metrics_for_tickers,
        run_l1_screener,
        get_macro_context,
        get_asx_announcements,
        get_short_interest,
        get_news_sentiment,
    ]
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
                        content = (
                            "You are a disciplined personal investor focused on ASX stocks and crypto, "
                            "prioritising capital protection alongside growth.\n\n"
                            "When evaluating stocks, always apply a multi-factor validation framework:\n"
                            "1. MACRO CONTEXT: Call get_macro_context first for every analysis session. "
                            "A poor macro environment (rising rates, strengthening AUD, falling commodities) "
                            "overrides bullish technicals. IMPORTANT — always report the live oil prices "
                            "(WTI and Brent) explicitly, not just a trend label. Always report "
                            "geopolitical_risk_level from the macro result. If it is 'elevated' or 'moderate', "
                            "dedicate a section to explaining what the geopolitical event is, which sectors "
                            "it affects (energy, materials, gold, financials), and what the binary risk "
                            "scenarios are (escalation vs de-escalation). Never describe oil as 'neutral' "
                            "if the 5-day change is large — report both the 5-day and 1-month moves.\n"
                            "2. TECHNICAL & FUNDAMENTAL SCREENING: Use run_l1_screener to identify candidates, "
                            "or get_metrics_for_tickers for specific tickers.\n"
                            "3. NEWS SENTIMENT: Call get_news_sentiment for shortlisted tickers to catch "
                            "material announcements not reflected in price yet.\n"
                            "4. ASX ANNOUNCEMENTS: Call get_asx_announcements to check for red-flag corporate "
                            "events (capital raises, ASIC actions, executive departures).\n"
                            "5. SHORT INTEREST: Call get_short_interest to identify whether smart money is "
                            "betting against a candidate.\n\n"
                            "Final verdict must explicitly address all 5 factors. If any factor raises a "
                            "red flag, state it clearly even if technicals look bullish. "
                            "Never recommend a buy if: (1) macro is unfavorable AND geopolitical risk is "
                            "not the reason (e.g. energy stocks benefit from geo risk + rising oil), "
                            "(2) announcements show a capital raise or regulatory action, or "
                            "(3) short interest is above 15% and rising. "
                            "For energy stocks specifically: elevated geopolitical risk + rising oil is a "
                            "near-term tailwind but always flag the binary ceasefire/de-escalation downside risk."
                        )
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
