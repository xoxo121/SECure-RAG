from ..schema.schema import BaseTool
from dotenv import load_dotenv
import yfinance as yf  # Ensure yfinance is listed in requirements.txt: pip install yfinance

load_dotenv()


class StockDataTool(BaseTool):
    """
    A tool to fetch stock-related data using the yfinance API.
    -----------------------------------------------------------
    Features:
    - Retrieve historical stock data for a given ticker symbol.
    - Specify the time period (e.g., '1d', '5d', '1mo').
    - Specify the interval between data points (e.g., '1m', '1h', '1d').

    Prerequisites:
    - Install yfinance: pip install yfinance.
    - Set up a .env file for environment variables (if needed for extensions).

    Usage:
    - Initialize the tool:
      tool = StockDataTool()
    - Fetch stock data using the run method:
      data = await tool.run("AAPL", "5d", "1h")
    - Optionally clean the output:
      cleaned_data = tool.clean_output(data)
    """

    def __init__(self, **kwargs):
        """
        Initialize the StockDataTool with default settings or custom arguments.
        Args:
            kwargs: Optional keyword arguments to customize the tool.
        """
        super().__init__(
            name=kwargs.get("name", "yahoo_finance_tool"),  # Name of the tool
            description=kwargs.get(
                "description",
                "A tool to fetch stock-related data using the yfinance API.",
            ),
            version="1.0",  # Version of the tool
            args={
                "ticker": 'The ticker symbol of the stock (e.g., "AAPL" for Apple).',
                "period": "The time period of the data: one of (1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max).",
                "interval": "The interval between data points: one of (1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo)",
            },
        )

    async def run(
        self, ticker: str, period: str = "1mo", interval: str = "1d"
    ):
        """
        Asynchronously fetches stock data.
        Args:
            ticker (str): The ticker symbol of the stock (e.g., "AAPL" for Apple).
            period (str): The time period of the data: one of (1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max).
            interval (str): The interval between data points: one of (1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo).

        Returns:
            dict: A dictionary containing stock data, converted to a list of records.
        Example:
            tool = StockDataTool()
            data = await tool.run("AAPL", "5d", "1h")
        """
        try:
            # Fetch the stock data using yfinance
            stock = yf.Ticker(ticker)  # Initialize the Ticker object
            data = stock.history(
                period=period, interval=interval
            )  # Fetch historical data

            # Convert the DataFrame to a dictionary for easier handling
            formatted_data = data.reset_index().to_dict(orient="records")
            max_count = 5
            n = len(formatted_data)
            return (
                formatted_data[:: (n // max_count)] + [formatted_data[-1]]
                if n > max_count
                else formatted_data
            )
        except Exception as e:
            raise RuntimeError(
                "An error occured while calling Yahoo Finance"
            ) from e
