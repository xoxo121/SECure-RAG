from ..schema.schema import BaseTool
import os
import requests
import certifi
import logging

api_key = os.environ["FINPREP_API_KEY"]
logger = logging.getLogger(__name__)


async def request(url):
    response = requests.get(url, verify=certifi.where())
    json_response = response.json()
    if isinstance(response, str):
        logger.error(f"Error occurred while fetching data from {url}")
    return json_response


class GetCompanyIncomeStatement(BaseTool):
    def __init__(self, **kwargs):
        super().__init__(
            name=kwargs.get("name", "company_income_statement_tool"),
            description=kwargs.get(
                "description",
                "A tool to retrieve the income statement of a company for a financial year",
            ),
            version="1.0",
            args={
                "ticker": "Ticker of the company",
                "years": "The years for which the income statement will be retrieved (only years 2020 or later). Example: ['2024', '2023']",
            },
        )

    async def run(self, ticker: str, years: list):
        """
        Asynchronously retrieves income statement.
        Args:
            ticker(str): Ticker of the company
            years(list): The years for which the income statement will be retrieved (only years 2020 or later). Example: ['2024', '2023']
        Returns:
            list: Income statements in json format.
        """
        url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?period=annual&apikey={api_key}"
        response = await request(url)
        print(response)
        results = []
        if response:
            for i in response:
                if (
                    int(i["date"].split("-")[0]) in years
                    or i["date"].split("-")[0] in years
                ):
                    results.append(i)
        return results


class GetCompanyRatios(BaseTool):
    def __init__(self, **kwargs):
        super().__init__(
            name=kwargs.get("name", "company_ratios_tool"),
            description=kwargs.get(
                "description",
                "A tool to retrieve the key ratios of a company for the last 12 months",
            ),
            version="1.0",
            args={"ticker": "Ticker of the company"},
        )

    async def run(self, ticker: str):
        """
        Asynchronously retrieves important ratios of a company.
        Args:
            ticker(str): Ticker of the company
        Returns:
            list: Ratios in json format.
        """
        url = f"https://financialmodelingprep.com/api/v3/ratios-ttm/{ticker}?apikey={api_key}"
        return await request(url)


class GetCompanyKeyMetrics(BaseTool):
    def __init__(self, **kwargs):
        super().__init__(
            name=kwargs.get("name", "company_key_metrics_tool"),
            description=kwargs.get(
                "description",
                "A tool to retrieve key metrics of a company for a financial year",
            ),
            version="1.0",
            args={
                "ticker": "Ticker of the company",
                "years": "The years for which the key metrics will be retrieved (only years 2020 or later). Example: ['2024', '2023']",
            },
        )

    async def run(self, ticker: str, years: list[int]):
        """
        Asynchronously retrieves key metrics of a company.
        Args:
            ticker(str): Ticker of the company
            years(list[int]): The years for which key mtrics will be retrieved
        Returns:
            list: Key metrics in json format.
        """
        url = f"https://financialmodelingprep.com/api/v3/key-metrics/{ticker}?period=annual&apikey={api_key}"
        response = await request(url)
        results = []
        if response:
            for i in response:
                if (
                    int(i["date"].split("-")[0]) in years
                    or i["date"].split("-")[0] in years
                ):
                    results.append(i)
        return results

if __name__ == '__main__':
    # python -m FinAgent.tools.finprep
    import asyncio
    asyncio.run(GetCompanyIncomeStatement().run("AAPL", ["2021", "2020"]))