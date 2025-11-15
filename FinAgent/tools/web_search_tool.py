from ..schema.schema import BaseTool
from dotenv import load_dotenv
from duckduckgo_search import DDGS

load_dotenv()


class WebSearchTool(BaseTool):
    def __init__(self, **kwargs):
        super().__init__(
            name=kwargs.get("name", "ddgs_web_search"),
            description=kwargs.get(
                "description",
                "A tool to search the web for information given a query",
            ),
            version="1.0",
            args={
                "query": "The query to search for(str)",
                "top_k": "The number of search results to retrieve(by default use 5)(int)",
            },
        )
        self.client = DDGS()

    async def run(self, query: str, top_k: int):
        """
        Asynchronously runs a search query.
        Args:
            query (str): The search query string.
            top_k (int): The number of top results to return.
        Returns:
            list: A list of search results.
        """
        response_from_duckduckgo = self.client.chat(query, model="mixtral-8x7b")

        return response_from_duckduckgo

    def clean_output(self, results):
        """
        Cleans the output from the vector store.
        Args:
            chunks (list): A list of search results.
        Returns:
            list: A list of cleaned search results.
        """
        chunks = []
        for result in results:
            chunks.append(
                {
                    "title": result["title"],
                    "description": result["body"],
                }
            )
        return chunks


if __name__ == "__main__":
    tool = WebSearchTool()
    print(tool.run("python programming", 5))
