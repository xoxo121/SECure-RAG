from ..schema.schema import BaseTool
from dotenv import load_dotenv
from langchain_community.utilities.alpha_vantage import AlphaVantageAPIWrapper
import asyncio
import os
load_dotenv()

class ExchangeRateTool(BaseTool):
    def __init__(self, **kwargs):
        super().__init__(
            name=kwargs.get("name", "exchange_rate_tool"),
            description=kwargs.get(
                "description",
                "A tool to retrieve currency exchange rates.",
            ),
            version="1.0",
            args={
                "base_currency": "The base currency for exchange rate (str)",
                "target_currency": "The target currency for exchange rate (str)",
            },
        )
        # Set the API key from environment variable or passed in kwargs
        api_key = kwargs.get("api_key", os.getenv("ALPHAVANTAGE_API_KEY"))
        if not api_key:
            raise ValueError("Alpha Vantage API key is required")

        # Set the API key in the environment
        os.environ["ALPHAVANTAGE_API_KEY"] = api_key

        # Initialize the Alpha Vantage API wrapper
        self.client = AlphaVantageAPIWrapper()

    async def run(self, base_currency: str, target_currency: str):
        """
        Asynchronously runs a financial market query.

        Args:
            base_currency (str): The base currency
            target_currency (str): The target currency for exchange rates

        Returns:
            dict: A dictionary containing the query results.
        """
        response = self.client._get_exchange_rate(base_currency, target_currency)
        return response