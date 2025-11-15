from typing import cast
from ..schema.schema import BaseTool
from dotenv import load_dotenv
from edgar import set_identity, Company, CompanyFilings
import asyncio


load_dotenv()


def get_financials_edgar(edgar_obj):
    """
    Hack for getting financials property from tenk[i].obj()

    Args:
        edgar_obj: tenk[i].obj()

    Returns:
        tenk[i].obj().financials
    """
    return edgar_obj.financials


class EdgarTool(BaseTool):
    def __init__(self, email: str = "johndoe@mail.com", **kwargs):
        super().__init__(
            name="edgar_tool",
            description=kwargs.get(
                "description",
                "A tool to retrieve the financial data and reports from EDGAR for any given company which takes all arguments compulsorily only as {company name, form type, date range ['YYYY-MM-DD to YYYY-MM-DD'], and financial specifics ['income', 'balance', 'cashflow', 'equity', 'comprehensive_income']}. Also use the retrieved data to extract and process if any user-specified variable which are never line-specified in the retrieved data and please please do not pass any extracted information in any argument.",
            ),
            version="1.0",
            args={
                "company_name_in_ticker_form": "The name of the company. For example, 'AAPL' for Apple Inc.",
                "form": "The type of form to filter (enum['10-K', '10-Q'])",
                "date_range": "The range of dates to search in, formatted as 'YYYY-MM-DD to YYYY-MM-DD' [when not given in this format, change it to the format]. Convert the year to the date range format as 'YYYY-01-01 to YYYY-12-31' if only one year is provided.",
                "financial_specifics": "A list of financials to retrieve as a list only from (enum['income', 'balance', 'cashflow', 'equity', 'comprehensive_income'])",
            },
        )
        self.mail = email

    async def run(
        self,
        company_name_in_ticker_form: str,
        form: str,
        date_range: str,
        financial_specifics: list[str],
    ):
        """
        Fetches financial data from EDGAR for a given company.
        Args:
            company_name_in_ticker_form (str): The name of the company to search. for example for
            form (str): The form type to search for (e.g., 10-K, 10-Q).
            date_range (str): The range of dates to search in, formatted as ('YYYY-MM-DD to YYYY-MM-DD').
            financial_specifics (List[str]): A list of financials to retrieve (enum['income', 'balance', 'cashflow', 'equity', 'comprehensive_income'])
        Returns:
            list: A dictionary containing concatenated financial data for the requested types.
        """
        set_identity(self.mail)
        start_date, end_date = date_range.split(" to ")
        start_date, end_date = (start_date.strip(), end_date.strip())
        year_start = int(start_date[:4]) + 1
        year_end = int(end_date[:4]) + 1
        date_range = f"{year_start}{start_date[4:]}:{year_end}{end_date[4:]}"

        tenk = cast(
            CompanyFilings,
            Company(company_name_in_ticker_form).get_filings(form=form),
        ).filter(date=date_range)

        context = []

        for i in range(len(tenk)):
            edgar_obj = await asyncio.to_thread(tenk[i].obj)

            financials = await asyncio.to_thread(
                get_financials_edgar, edgar_obj
            )
            for stat in financial_specifics:
                if stat == "income":
                    table = (
                        financials.get_income_statement()
                        .get_dataframe()
                        .iloc[:, [0, -1]]
                    )  # .iloc[:,0::3]

                elif stat == "balance":
                    table = (
                        financials.get_balance_sheet()
                        .get_dataframe()
                        .iloc[:, [0, -1]]
                    )
                elif stat == "cashflow":
                    table = (
                        financials.get_cash_flow_statement()
                        .get_dataframe()
                        .iloc[:, [0, -1]]
                    )

                elif stat == "equity":
                    table = (
                        financials.get_statement_of_changes_in_equity()
                        .get_dataframe()
                        .iloc[:, [0, -1]]
                    )
                elif stat == "comprehensive_income":
                    table = (
                        financials.get_statement_of_comprehensive_income()
                        .get_dataframe()
                        .iloc[:, [0, -1]]
                    )
                else:
                    raise ValueError(
                        f"Invalid financial statement type: {stat}"
                    )
                context.append(table)

        return [context[i].to_html() for i in range(len(context))]


if __name__ == "__main__":
    tool = EdgarTool()
    print(
        asyncio.run(
            tool.run(
                "AAPL",
                "10-K",
                "2020-01-01 to 2021-01-01",
                [
                    "income",
                    "balance",
                    "cashflow",
                    "equity",
                    "comprehensive_income",
                ],
            )
        )
    )
