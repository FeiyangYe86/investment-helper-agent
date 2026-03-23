from dotenv import load_dotenv

import operator
from typing import Literal
from typing_extensions import TypedDict, Annotated

from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from tools.ticker.metrics_tools import get_metrics_for_tickers, run_l1_screener

try:
    load_dotenv()
    model = init_chat_model(model="claude-haiku-4-5", model_provider="anthropic", temperature=0)
    print('=====')
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


# Tool node - method to call tools
def tool_node(state: dict):
    result = []
    for tool_call in state['messages'][-1].tool_calls:
        tool = tools_by_name[tool_call['name']]
        print('====> Calling tool ' + tool.name)
        observation = tool.invoke(tool_call['args'])
        result.append(ToolMessage(content=str(observation), tool_call_id=tool_call['id']))

    return {'messages': result}


# Edge logic
def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    messages = state['messages']
    last_message = messages[-1]

    if last_message.tool_calls:
        return 'tool_node'

    return END

agent_builder = StateGraph(MessagesState)

agent_builder.add_node('llm_call', llm_call)
agent_builder.add_node('tool_node', tool_node)
agent_builder.add_edge(START, 'llm_call')
agent_builder.add_conditional_edges(
    'llm_call',
    should_continue,
    ['tool_node', END]
)

agent_builder.add_edge('tool_node', 'llm_call')

agent = agent_builder.compile()

try:
    messages = [HumanMessage(content="Analyze the performance of SDR.AX in ASX and suggest if I should buy it.")]
    print("Invoking agent...", flush=True)
    messages = agent.invoke({"messages": messages})
    print("Agent done, messages:", len(messages["messages"]), flush=True)
    for m in messages["messages"]:
        m.pretty_print()
except Exception as e:
    print("Error:", e, flush=True)
    import traceback; traceback.print_exc()
