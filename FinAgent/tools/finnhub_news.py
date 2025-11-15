from ..schema.schema import BaseTool
from dotenv import load_dotenv
from typing import Optional, List, Dict
import asyncio
load_dotenv()
import os
import finnhub

class FinnhubToolMarketNews(BaseTool):
    def __init__(self, **kwargs):
        super().__init__(
            name=kwargs.get("name", "finmarket_search"),
            description=kwargs.get(
                "description",
                "A tool to search realtime data on latest market news on general, forex, crypto, merger topics.",
            ),
            version="1.0",
            args={
                "category": "The category of the realtime data to search [] (str)",
            },
        )
        # Initialize the AskNewsSearch client
        self.client = finnhub.Client(api_key=os.getenv("FINNHUB_API_KEY"))


    async def run(self, category: str) -> str:
        """
        Asynchronously runs a news search query.

        Args:
            category (str): The category of the realtime data to search [general, forex, crypto, merger] (str).

        Returns:
            str: A string containing the news
        """
        # Use invoke method with the query
        response = self.client.general_news(category)
        return response
