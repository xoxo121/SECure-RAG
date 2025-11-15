from ..schema.schema import BaseTool
from dotenv import load_dotenv
from langchain_community.tools.asknews import AskNewsSearch
import asyncio
load_dotenv()

class NewsSearchTool(BaseTool):
    def __init__(self, **kwargs):
        super().__init__(
            name=kwargs.get("name", "ask_news_tool"),
            description=kwargs.get(
                "description",
                "A tool to search for news articles given a query string. It can also return financial news articles.",
            ),
            version="1.0",
            args={
                "query": "The news search query string (str)",
            },
        )
        # Initialize the AskNewsSearch client
        self.client = AskNewsSearch(max_results=kwargs.get("max_results", 2))

    async def run(self, query: str) -> str:
        """
        Asynchronously runs a news search query.

        Args:
            query (str): The search query string.

        Returns:
            str: A string containing the news
        """
        try:
            # Use invoke method with the query
            response = self.client.invoke({"query": query})
            return response
        except Exception as e:
            e.add_note("An error occurred while running the news search tool.")
            raise e