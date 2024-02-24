from langchain import hub
from langchain.agents import AgentExecutor, tool
from langchain.agents.output_parsers import XMLAgentOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
from fastapi import FastAPI
from langchain.pydantic_v1 import BaseModel
from langserve import add_routes
from typing import Any

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, streaming=True)

prompt = hub.pull("hwchase17/xml-agent-convo")

@tool
def search(query: str) -> str:
	"""Search about things with DuckDuckGo Search Engine."""
	_search = DuckDuckGoSearchResults()
	return _search.run(query)


tool_list = [search]

def convert_intermediate_steps(intermediate_steps):
    log = ""
    for action, observation in intermediate_steps:
        log += (
            f"<tool>{action.tool}</tool><tool_input>{action.tool_input}"
            f"</tool_input><observation>{observation}</observation>"
        )
    return log


def convert_tools(tools):
    return "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: convert_intermediate_steps(
            x["intermediate_steps"]
        ),
    }
    | prompt.partial(tools=convert_tools(tool_list))
    | model.bind(stop=["</tool_input>", "</final_answer>"])
    | XMLAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tool_list, verbose=True)


class Input(BaseModel):
    input: str

class Output(BaseModel):
    output: Any

app = FastAPI(
     title="DuckDuckGo Agent",
     version="1.0",
     description="API for serving a LangChain Agent that can access the web."
)

add_routes(
    app,
    agent_executor.with_types(input_type=Input, output_type=Output).with_config(
        {"run_name": "agent"}
    ),
    path="/agent"
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)