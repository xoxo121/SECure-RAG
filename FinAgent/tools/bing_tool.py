from ..schema.schema import BaseTool
from dotenv import load_dotenv
from typing import Optional, TypedDict
import requests
import os
import asyncio
load_dotenv()

class Chunk(TypedDict):
    id:int
    title: str
    description: str
    url: str
class BingWebSearchTool(BaseTool):
    def __init__(self, **kwargs):
        super().__init__(
            name=kwargs.get("name", "bing_web_search"),
            description=kwargs.get(
                "description",
                "A tool to search the web for information (including financial information) using Bing given a query",
            ),
            version="1.0",
            args={
                "query": "The query to search for(str)",
                "top_k": "The number of search results to retrieve(by default use 5)(int)",
            },
        )
        self.endpoint = "https://api.bing.microsoft.com/v7.0/search"
        self.subscription_key = os.environ['BING_SEARCH_API_KEY']

    async def run(self, query: str, top_k: int) -> list[Chunk]:
        """
        Asynchronously runs a search query using Bing Web Search API.
        Args:
            query (str): The search query string.
            top_k (int): The number of top results to return.
        Returns:
            list: A list of search results.
        """
        headers = {
            "Ocp-Apim-Subscription-Key": self.subscription_key
        }
        params = {
            "q": query,
            "count": top_k,
            "textDecorations": True,
            "textFormat": "HTML"
        }

        response = requests.get(self.endpoint, headers=headers, params=params)
        response.raise_for_status()
        search_results = response.json()

        return self.clean_output(search_results)

    def clean_output(self, results):
        """
        Cleans the output from the Bing Web Search API.
        Args:
            results (list): A list of raw search results.
        Returns:
            list: A list of cleaned search results.
        """
        chunks: list[Chunk] = []
        i=1
        for result in results.get("webPages", {}).get("value", []):
            chunks.append(
                {
                    "id":i,
                    "title": result["name"].replace("\u2018", "'").replace("\u2019", "'"),
                    "description": result["snippet"].replace("<b>","").replace("</b>","").replace("\u2018", "'").replace("\u2019", "'"),
                    "url": result["url"]
                }
            )
            i+=1
        return chunks
