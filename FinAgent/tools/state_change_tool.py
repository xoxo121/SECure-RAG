from ..schema.schema import BaseState, BaseTool


class StateChangeTool(BaseTool):
    """
    A tool that updates the system prompt based on the next state received from the LLM.
    """

    def __init__(self, states: dict[str, BaseState]):
        """
        Initialize the tool with the agent instance.

        Args:
            agent (Agent): The agent instance that contains the current states.
            states (dict[str, BaseState]): A dictionary of states in the agent.
                ! Initialising this tool adds itself to every state

        """
        self.states = states
        for state in states:
            self.states[state].tools.append(self)
        super().__init__(
            name="state_change_tool",
            description=f"""
            The tool allows you to change between states.{[state.basic_info() for key, state in states.items()]}
            """,
            version="1.0",
            args={
                "next_state": "The name of the next state to transition to.",
            },
        )  # The agent instance

    def run(self, next_state: str) -> str:
        """
        Executes the tool to update the system message based on the next state.

        Args:
            args (dict): A dictionary containing:
                - 'next_state' (str): The name of the next state to transition to.

        Returns:
            str: The next state name.
        """

        return next_state
