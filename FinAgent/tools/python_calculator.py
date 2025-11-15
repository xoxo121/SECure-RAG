from ..schema.schema import BaseTool
from dotenv import load_dotenv
import re

load_dotenv()


class Python_Calculator_Tool(BaseTool):
    def __init__(self, **kwargs):
        super().__init__(
            name=kwargs.get("name", "Python_Calculator"),
            description=kwargs.get(
                "description",
                "A tool to do the calculations (Using python interpreter) for a given mathematical expression as a string.",
            ),
            version="1.0",
            args={
                "query": """A string containing the mathematical expression in the format of a python expression to be evaluated 
(should be strictly a mathematical expression ; no wordings / description allowed) 
No libraries of any kind may be used.""",
            },
        )

    async def run(self, query: str):
        """
        Asynchronously runs a calculator.
        Args:
            query (str): query string of a mathematical expression.
        Returns:
            float : value of the mathematical expression.
        """
        try:
            query = re.sub(r"[a-zA-Z]", "", query)  # safety
            return eval(query)
        except Exception as e:
            raise SyntaxError(
                f"given expression : {query} is not a valid mathematical expression"
            ) from e
