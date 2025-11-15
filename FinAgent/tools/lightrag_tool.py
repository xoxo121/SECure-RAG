from ..schema.schema import BaseTool
from dotenv import load_dotenv
import os
import aiohttp
import aiohttp.http_exceptions

load_dotenv()


class LightRAGTool(BaseTool):
    """
    LightRAGTool: A tool to retrieve context from a knowledge graph using a RAG approach.
    """

    def __init__(self, **kwargs):
        self.url = os.environ["LIGHT_RAG_URL"]
        self.mode = kwargs.get("mode", "hybrid")
        super().__init__(
            name=kwargs.get("name", "financial_rag_tool"),
            description=kwargs.get(
                "description",
                "A tool to retrieve context from a knowledge graph",
            ),
            version="1.0",
            args={
                "query": "The query to search for",
                "top_k": "The number of chunks to retrieve (default: 3)",
            },
        )

    async def run(self, query: str, top_k: int = 1) -> dict:
        """
        Executes an asynchronous request to the RAG API.

        Args:
            query (str): The search query.
            top_k (int): Number of results to retrieve.

        Returns:
            dict: The JSON response from the RAG API or error message.
        """
        payload = {
            "query": query,
            "mode": self.mode,
            "only_need_context": True,
            "top_k": top_k,
        }
        headers = {"Content-Type": "application/json"}

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    self.url, json=payload, headers=headers
                ) as response:
                    response.raise_for_status()  # Raise exception for HTTP errors
                    return await response.json()
            except (
                aiohttp.ClientError,
                aiohttp.http_exceptions.HttpProcessingError,
            ) as e:
                raise ConnectionError("Error in post request") from e
