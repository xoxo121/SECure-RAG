from ..schema.schema import BaseTool
from GoogleNews import GoogleNews
import asyncio


class GoogleNewsTool(BaseTool):
    def __init__(self, **kwargs):
        super().__init__(
            name=kwargs.get("name", "google_news_tool"),
            description=kwargs.get(
                "description",
                "A tool to fetch the latest news headlines and summaries from Google News.",
            ),
            version="1.0",
            args={
                "query": "The topic or keyword to search for (str) Please put only important keywords not sentences",
                "language": "The language for the news articles (default is 'en')(str)",
                "region": "The region for the news articles (default is 'US')(str)",
                "num_results": "The number of news articles to retrieve (default is 10)(int)",
            },
        )
        self.client = GoogleNews()

    async def run(
        self,
        query: str,
        language: str = "en",
        region: str = "US",
        num_results: int = 10,
    ):
        """
        Asynchronously fetches news articles from Google News.
        Args:
            query (str): The search query for news articles.
            language (str): The language of the news articles.
            region (str): The region of the news articles.
            num_results (int): The number of news articles to fetch.
        Returns:
            list: A list of news articles with headlines and summaries.
        """
        # Configure the client
        self.client.set_lang(language)
        self.client.search(query)

        # Fetch results
        news_items = self.client.result()[:num_results]
        return self.clean_output(news_items)

    def clean_output(self, results):
        """
        Cleans the output from the Google News search results.
        Args:
            results (list): The list of news articles from Google News.
        Returns:
            list: A cleaned list of news articles with key details.
        """
        cleaned_results = []
        for result in results:
            cleaned_results.append(
                {
                    "title": result.get("title", "No Title"),
                    "link": result.get("link", "No Link"),
                    "media": result.get("media", "No Media Source"),
                    "date": result.get("date", "No Date"),
                    "summary": result.get("desc", "No Description"),
                }
            )
        return cleaned_results
