# Adding new tools

### Creating a tool
```python
# Finagent/tools/<tool_name>.py
from ..schema.schema import BaseTool
from dotenv import load_dotenv
load_dotenv() # if API key is required

class ToolName(BaseTool):
    def __init__(self, **kwargs):
        super().__init__(
            # Note: name, description and args here are given to the LLM for tool calling
            name=kwargs.get("name", "tool_name"),
            description=kwargs.get(
                "description",
                "Tool description",
            ),
            version="<version number> Example: 1.0",
            args={
                "arg1_name": "arg1 description (<data type 1>)",
                "arg2_name": "arg2 description (<data type 2>)"
            },
        )
    def run(self, arg1: <datatype 1>, arg2: <datatype 2>):
        """
        <Tool purpose/description>

        Args:
            <arg1> (<data type 1>): arg1 description
            <arg2> (<data type 2>): arg2 description

        Returns:
            <return datatype>: description of response
        """
        return tool_response
```

### Adding a tool to the pipeline
```python
# Finagent/config/states.py
from Finagent.tools.<tool_name> import ToolName

tool_name = ToolName()

# The same method can be followed to add tools in stateless_agent_states, auto_states, meta_states
multi_states_agent_states = {"BaseState": BaseState(...),
                             "FactBasedFinance": BaseState(
                                    name = "state name",
                                    goal = "state goal"
                                    instructions = "state instructions",
                                    model = GeminiModel(),
                                    tools = [..., tool_name])
                                }
```