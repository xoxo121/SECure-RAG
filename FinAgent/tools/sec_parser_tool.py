from ..schema.schema import BaseTool
from dotenv import load_dotenv
from edgar import Company, set_identity

load_dotenv()


class SECParserTool(BaseTool):
    def __init__(self, email: str = "johndoe@mail.com", **kwargs):
        super().__init__(
            name=kwargs.get("name", "sec_parser_tool"),
            description=kwargs.get(
                "description",
                "A tool to retrieve SEC report from EDGAR for a given company by accepting the company name [change it to its relevant ticker name], form and the years as arguments in markdown",
            ),
            version="1.0",
            args={
                "company_name": "The name of the company ticker name [when only company name given, change it to its ticker name].",
                "form": "The type of form to filter (e.g., 10-K, 10-Q)",
                "years": 'The range of years to search for in one of the formats ("YYYY-MM-DD","YYYY-MM-DD:YYYY-MM-DD","YYYY-MM-DD:",":YYYY-MM-DD")',
            },
        )
        self.mail = email

    def run(self, company_name: str, form: str, years: str):
        """
        Fetches SEC report from EDGAR for a given company in markdown.
        Args:
            company_name (str): The name of the company ticker name [when only company name given, change it to its ticker name].
            form (str): The form type to search for (e.g., 10-K, 10-Q).
            years (str): The range of years to search for in one of the formats ("YYYY-MM-DD:YYYY-MM-DD","YYYY-MM-DD:",":YYYY-MM-DD")
        Returns:
            str: A dictionary containing concatenated financial data for the requested types.
        """
        set_identity(self.mail)
        result = ""
        if not isinstance(years, str):
            raise TypeError("Years should be of type `str`")

        tenk = Company(company_name).get_filings(date=years, form=form)
        if not tenk:
            raise ValueError(f"""Cannot extract a date or date range from string {years}
    Provide either 
        1. A date in the format "YYYY-MM-DD" e.g. "2022-10-27"
        2. A date range in the format "YYYY-MM-DD:YYYY-MM-DD" e.g. "2022-10-01:2022-10-27"
        3. A partial date range "YYYY-MM-DD:" to specify dates after the value e.g.  "2022-10-01:"
        4. A partial date range ":YYYY-MM-DD" to specify dates before the value  e.g. ":2022-10-27""")
        for i in range(len(tenk)):
            result += tenk[i].markdown()
            result += "\n\n"

        return result


if __name__ == "__main__":
    tool = SECParserTool()
    print(doc := tool.run("AAPL", "10-K", "2020-01-01:2022-01-01"))
    print(len(doc))
