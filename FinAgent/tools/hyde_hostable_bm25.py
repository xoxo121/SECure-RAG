from typing import Optional
from ..schema.schema import BaseTool
import os
from dotenv import load_dotenv
from ..utils.subclassed_client import PathwayVectorClient_with_timeout as PathwayVectorClient
from . import hyde_tools_bm25 as hyde_tool

load_dotenv()


class Hyde_Multi_Reranker_Tool(BaseTool):

    def __init__(self, host: Optional[str] = None, 
                 port: Optional[int] = None, 
                 url: Optional[str] = None,
                 url_bm25: Optional[str] = None,
                 **kwargs):

        super().__init__(
            name="financial_rag_tool",
            description=kwargs.get(
                "description",
                "A tool to search and retrieve the documents related to the given query, from a vector_store containing a corpus of documents",
            ),
            version="1.0",
            args={
                "query": "The query to search for(str)",
                "top_k1": "The number of search results to retrieve(by default use 20)(int)",
                "top_k2": "The number of reranked results to retrieve(by default use 10)(int)",
                "n_similar": "N queries similar to that will be created and documents will be searched based on all , so complex queries can be handled effectively(by default use 2)(int)",
            },
        )

        if not ((host and port) or url):
            raise ValueError("Either ('host' and 'port') or 'url' must be provided")

        self.client = PathwayVectorClient(
            host=host,
            port=port,
            url=url,
        )
        self.bm_25_retriever=PathwayVectorClient(
            host=host,
            port=port,
            url=url_bm25,
        )
        groq_api_key = kwargs.get("GROQ_API_KEY", None)
        model_name = kwargs.get("model_name", "groq/llama3-70b-8192")
        reranker_model_name = kwargs.get(
            "reranker_model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )


        self.hyde_doc_creator = hyde_tool.Hyde_Multi_document_creator(
            api_key=groq_api_key, model=model_name
        )
        if os.getenv("REPLICATE_API_KEY"):
            self.reranker = hyde_tool.BGEReranker_replicate()
        else:
            self.reranker = hyde_tool.CrossEncoderReranker(model_name=reranker_model_name)


    async def run(self, query: str, top_k1: int, top_k2: int, n_similar : int = 2):
        """
        Asynchronously runs a similarity search query.
        Args:
            "query" (str): The search query string.
            "top_k1" (int) : "Top k documents to retreive",
            "top_k2" (int) : "Top k reranked documents"
            "n_similar" (int) : "N queries similar to that will be created and documents will be searched based on all , so complex queries can be handled effectively"
        Returns:
            list: A list of search results.
        """

        self.hyde_retriver_reranker = hyde_tool.full_multi_hyde_without_mean_reranker_pipeline_bm25(
            HyDE_generator=self.hyde_doc_creator,
            retriever=self.client,
            reranker=self.reranker,
            top_k = top_k1,
            top_k_rerank=min(top_k2,20),
            bm_25 = self.bm_25_retriever,
            top_k_table = 5,
        )

        docs = await self.hyde_retriver_reranker.get_documents( queries=[query], n_similar = min(n_similar,5) )

        return docs[0]
