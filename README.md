# COVALE

COVALE is a Python project for neuroimaging workflows using NiBabel and the
consumer OpenAI API.

## Prerequisites

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- An OpenAI API key from the [OpenAI Platform](https://platform.openai.com/api-keys)

## Install

Clone the repository and enter its directory:

```bash
git clone <repository-url> covale
cd covale
```

Create the virtual environment outside the repository, tell `uv` where to find
it, and install the locked dependencies:

```bash
uv venv "$HOME/venvs/covale" --python 3.11
export UV_PROJECT_ENVIRONMENT="$HOME/venvs/covale"
uv sync --locked
source "$HOME/venvs/covale/bin/activate"
```

For each new shell, select and activate the environment again:

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/venvs/covale"
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
```

To run the starter command:

```bash
uv run covale
```

## Configure OpenAI

Add your consumer OpenAI API key to the local `.env` file:

```dotenv
OPENAI_API_KEY=your-api-key
```

The `.env` file is excluded from Git. Load it before creating the OpenAI client:

```python
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()
response = client.responses.create(
    model="gpt-5-mini",
    input="Summarize the purpose of a NIfTI header.",
)
print(response.output_text)
```

## Create a deep agent

`deepagents` can use the OpenAI model configured by `OPENAI_API_KEY`. Replace
the example tool with tools appropriate for the workflow:

```python
from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain_core.tools import tool


@tool
def my_custom_tool(query: str) -> str:
    """Search a project-specific research source."""
    return f"Research results for: {query}"


load_dotenv()
agent = create_deep_agent(
    model="openai:gpt-6-astra",
    tools=[my_custom_tool],
    system_prompt="You are a research assistant.",
)
result = agent.invoke(
    {"messages": "Research LangGraph and write a summary"}
)
```

## Load a neuroimaging file

```python
import nibabel as nib

image = nib.load("scan.nii.gz")
print(image.shape)
```

## Add dependencies

Use `uv` to keep `pyproject.toml` and `uv.lock` synchronized:

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/venvs/covale"
uv add <package>
```