# 🦜👑 LangChain Zero-to-Hero / 🤖 _Agents_
 Get started with LangChain Agents, part of the zero-to-hero series.

## Before you start

* This tutorial uses the terminal to install dependencies and run Python scripts.
* When you see the 🆕 emoji before a set of terminal commands, open a new terminal process.
* When you see the ♻️ emoji before a set of terminal commands, you can re-use the same terminal you used last time.

## Prerequisites

1. Download and install [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer).

2. Setup a Poetry environment:

🆕
```sh
poetry init --no-interaction --python="^3.11" --dependency=langchain --dependency=langchain-openai --dependency=langchainhub --dependency="langserve[all]" --dependency=duckduckgo-search
poetry install
```

3. Get an OpenAI API key and save it as an environment variable (e.g.: with [DirEnv](https://direnv.net/)):

```sh
export OPENAI_API_KEY=...
```

## Getting Started

1. Let's build a simple agent script.

* Create a file `langchain_zero_to_hero_agents/src/main.py` and create a simple agent (code modified from the [LangChain Agent Cookbook](https://python.langchain.com/docs/expression_language/cookbook/agent))

```python
from langchain import hub
from langchain.agents import AgentExecutor, tool
from langchain.agents.output_parsers import XMLAgentOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults

#######################
# LangChain Agent Code
#######################

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, streaming=True)


@tool
def search(query: str) -> str:
    """Search things about current events."""
    search = DuckDuckGoSearchResults()
    return search.run(query)


tool_list = [search]

prompt = hub.pull("hwchase17/xml-agent-convo")


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

if __name__ == "__main__":
    print(agent_executor.invoke({"input": "whats the weather in New york?"}))
```

2. Try running the script in the terminal to test it works:

♻️
```sh
poetry run python langchain_zero_to_hero_agents/src/main.py
```

3. To make this agent useful, we can setup a simple API with LangServe:

```python
from fastapi import FastAPI
from langchain.pydantic_v1 import BaseModel
from langserve import add_routes
from typing import Any

#######################
# LangChain Agent Code
#######################

# ...

######################
# LangServe API Code
######################

class Input(BaseModel):
    input: str

class Output(BaseModel):
    output: Any

app = FastAPI(
    title="DuckDuckGo Agent",
    version="1.0",
    description="API for accessing a simple LangChain agent that can query the web with DuckDuckGo.",
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
```

4. Startup your LangServe API server:

♻️
```sh
poetry run python langchain_zero_to_hero_agents/src/main.py
```

5. Visit http://localhost:8000/agent/playground/ to access a simple UI for interacting with your agent.

6. Create a test script (`langchain_zero_to_hero_agents/tests/test.py`) to experiment with accessing your API via Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/agent/invoke",
    json={'input': {"input": "what is the weather in new york"}}
)

print(response.json())
```

7. Run your test script and observe the structured JSON output:

🆕
```sh
poetry run python langchain_zero_to_hero_agents/tests/test.py
```

## License
MIT
