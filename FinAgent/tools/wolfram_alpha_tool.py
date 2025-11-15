from ..schema.schema import BaseTool
from dotenv import load_dotenv
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
import os
from concurrent.futures import ThreadPoolExecutor
import asyncio

load_dotenv()

class WolframAlphaTool(BaseTool):
    def __init__(self, **kwargs):
        super().__init__(
            name=kwargs.get("name", "wolfram_alpha_query"),
            description=kwargs.get(
                "description",
                "A tool to perform all computational and mathematical queries using Wolfram Alpha",
            ),
            version="1.0",
            args={
                "query": "The computational or mathematical query to solve (str)",
            },
        )
        # Get API key from environment or kwargs
        api_key = kwargs.get("api_key", os.environ["WOLFRAM_ALPHA_APPID"])

        # Initialize the Wolfram Alpha client
        self.client = WolframAlphaAPIWrapper(wolfram_alpha_appid=api_key)

    async def run(self, query: str):
        """
        Asynchronously runs a Wolfram Alpha query.

        Args:
            query (str): The computational or mathematical query.

        Returns:
            str: A string output.
        """
        if not isinstance(query, str):
            raise TypeError(f"Expected query to be of type str, got {type(query)}")
        try:
            # Use a thread pool to run the blocking query in a non-blocking way
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor() as pool:
                response = await loop.run_in_executor(pool, self.client.run, query)

            return response
        except Exception as e:
            raise RuntimeError("An error occurred while calling Wolfram Alpha") from e

if __name__ == "__main__":
    async def main():
        # Example usage
        tool = WolframAlphaTool()

        # Perform a query
        print("Wolfram Alpha Query Result:")
        query_results = await tool.run("What is 10x + 9 = -3x + 7?")

        print(query_results)

    asyncio.run(main())
