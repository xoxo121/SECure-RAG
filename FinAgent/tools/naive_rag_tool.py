from typing import Optional, TypedDict, NotRequired
from ..schema.schema import BaseTool
from dotenv import load_dotenv
from ..utils.subclassed_client import PathwayVectorClient_with_timeout as PathwayVectorClient

load_dotenv()

class PathwayRAGToolResponse(TypedDict):
    content: str
    source: str
    score: NotRequired[float]

class NaiveRAGTool(BaseTool):

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None, url: Optional[str] = None, **kwargs):
        super().__init__(
            name=kwargs.get("name", "naive_rag_tool"),
            description=kwargs.get(
                "description",
                "A tool to retrieve the top k chunks from a vector store containing real-time data using semantic search. The vector store contains SEC filings of several companies",
            ),
            version="1.0",
            args={
                "query": "The query to search for",
                "top_k": "The number of chunks to retrieve",
            },
        )
        if not ((host and port) or url):
            raise ValueError("Either ('host' and 'port') or 'url' must be provided")
        self.client = PathwayVectorClient(
            host=host,
            port=port,
            url=url
        )

    async def run(self, query: str, top_k: int) -> list[PathwayRAGToolResponse]:
        """
        Asynchronously runs a similarity search query.
        Args:
            query (str): The search query string.
            top_k (int): The number of top results to return.
        Returns:
            list: A list of search results.
        """
        docs = await self.client.asimilarity_search(query, top_k)
        answer: list[PathwayRAGToolResponse] = []
        for doc in docs:
            relevant_info: PathwayRAGToolResponse = {
                "content": doc.page_content,
                "source": doc.metadata["path"],
            }
            answer.append(relevant_info)
        return answer