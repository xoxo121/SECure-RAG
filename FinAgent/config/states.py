import os

import dotenv

from FinAgent.models import GeminiModel, GPTModel
from FinAgent.models.models import MockModel
from FinAgent.schema.schema import BaseState
from FinAgent.tools.alphavantage_tool import ExchangeRateTool
from FinAgent.tools.asknews_tool import NewsSearchTool
from FinAgent.tools.bing_tool import BingWebSearchTool
from FinAgent.tools.edgar_tool import EdgarTool
from FinAgent.tools.finnhub_news import FinnhubToolMarketNews
from FinAgent.tools.finprep import (
    GetCompanyIncomeStatement,
    GetCompanyKeyMetrics,
    GetCompanyRatios,
)
from FinAgent.tools.google_news_tool import GoogleNewsTool
from FinAgent.tools.hyde_hostable_bm25 import Hyde_Multi_Reranker_Tool

# from FinAgent.tools.hyde_tools_bm25 import BM25_retriever
from FinAgent.tools.naive_rag_tool import NaiveRAGTool
from FinAgent.tools.python_calculator import Python_Calculator_Tool
from FinAgent.tools.sec_parser_tool import SECParserTool
from FinAgent.tools.web_search_tool import WebSearchTool
from FinAgent.tools.wolfram_alpha_tool import WolframAlphaTool
from FinAgent.tools.yfin_tool import StockDataTool
from FinAgent.tools.lightrag_tool import LightRAGTool  # noqa: F401

dotenv.load_dotenv()


url = os.environ["PATHWAY_VECTOR_STORE_URL"]

hyde_multi_reranker_tool = Hyde_Multi_Reranker_Tool(
    url=url, url_bm25=os.environ["HYDE_BM25_URL"]
)
naive_rag_tool = NaiveRAGTool(url=url)
bing_web_search_tool = BingWebSearchTool()
yahoo_finance_stock_data_tool = StockDataTool()
sec_parser_tool = SECParserTool()
wolfram_alpha_tool = WolframAlphaTool()
google_news_tool = GoogleNewsTool()
web_search_tool = WebSearchTool()
finn_hub_tool = FinnhubToolMarketNews()
company_income_statement_tool = GetCompanyIncomeStatement()
company_ratios_tool = GetCompanyRatios()
company_key_metrics_tool = GetCompanyKeyMetrics()
edgar_tool = EdgarTool()
python_calculator_tool = Python_Calculator_Tool()
news_search_tool = NewsSearchTool()
exchange_rate_tool = ExchangeRateTool()
ask_news_tool = NewsSearchTool()
# light_rag_tool = LightRAGTool()


multi_state_agent_states = {
    "BaseState": BaseState(
        name="BaseState",
        goal="To Answer the user's query by routing it to one of the states below. Do not hallucinate any instructions",
        instructions=""" In this state, your only job is to transition to be best state to answer the query. You should only use the TOOL: StateChangeTool. Please do not call any other tool in this state. My life depends on this.
                    You are given a query, classify the query into one of the following categories and use the TOOL: StateChangeTool to transition to the appropriate state. Once transitioned follow the instructions given in the state without fail:
                    - If the query is related to finance and is asking about a number then change the state to the STATE:FactBasedFinance Use this also to retrieve all necessary information since it has access to TOOL: naive_rag_tool, TOOL: financial_rag_tool, TOOL: bing_web_search, TOOL: stock_data_tool, TOOL: company_income_statement_tool, TOOL: company_ratios_tool, TOOL: company_key_metrics_tool, TOOL: exchange_rate_tool and TOOL: sec_parser_tool..  
                    - If the query asks about a numerical analysis or if you need to use calculators or wolfram alpha change the state to the STATE:StatisticalAnalysis. 
                    - If the query asks about a decision or event change, say like access news, use TOOL: StateChangeTool to change the state to the STATE:DecisionAndEventBased.
                    - If you are not sure which state to transition to, use TOOL: StateChangeTool to change the state to the STATE: FactBasedFinance.
                    - You should never call any other tool  other that TOOL: StateChangeTool in this state.
                    - Do not hallucinate any information or additional tool arguments.
                    """,
        model=GeminiModel(),
        tools=[],
    ),
    "FactBasedFinance": BaseState(
        name="FactBasedFinance",
        goal="To provide accurate and grounded answers to fact-based descriptive financial user queries which require minimal calculations by following the instructions below. Has access to TOOL: naive_rag_tool, TOOL: financial_rag_tool, TOOL: bing_web_search, TOOL: stock_data_tool, TOOL: company_income_statement_tool, TOOL: company_ratios_tool, TOOL: company_key_metrics_tool, TOOL: exchange_rate_tool and TOOL: sec_parser_tool.",
        instructions=""" You can rewrite the query to make it more comprehensive for better retrieval and downstream tasks. Send the improved query to the user.
                        - Always use TOOL: financial_rag_tool to retrieve information from the vector store first. Assume that information for the query is present in the vector store. Assume information in the vector store is always up-to-date and accurate.
                        - If unsatisfactory, rewrite the query to make it more comprehensive, try financial_rag_tool again, then only use the TOOL: bing_web_search to fetch more information.
                        - Ensure the response is grounded on the retrieved data from the above tools.
                        - Avoid assuming or hallucinating information or using outdated context.
                        - If the query cannot be answered well in this state, use the TOOL: StateChangeTool to change the state to the STATE:BaseState
                        - If the generated answer is satisfactory and still requires more information to completely answer the query, use the tools provided to expand your horizons immediately.
                        - If the generated answer is satisfactory, then use TOOL: state_change_tool to change to the STATE: BaseState and then follow the instructions given in the state without fail.
                        - If the generated answer is still not satisfactory, ask the user for more information or clarify the query and repeat the process once.
                        - If the query is still not answered then use TOOL: state_change_tool to change the state to the STATE:BaseState and then follow the instructions given in the state without fail.
                        - Very Important: When you are changing state, do not output anything. Just change the state and follow the instructions given in the state without fail.""",
        model=GeminiModel(),
        tools=[
            hyde_multi_reranker_tool,
            naive_rag_tool,
            bing_web_search_tool,
            company_income_statement_tool,
            company_ratios_tool,
            company_key_metrics_tool,
            yahoo_finance_stock_data_tool,
            sec_parser_tool,
            exchange_rate_tool,
        ],
    ),
    "StatisticalAnalysis": BaseState(
        name="StatisticalAnalysis",
        goal="To assist the user with statistical or numerical analysis of financial data. Has access to several mathematical tools like a TOOL: Python_Calculator and TOOL: wolfram_alpha_query alongside, TOOL: financial_rag_tool TOOL: bing_web_search, TOOL: company_income_statement_tool, TOOL: company_ratios_tool, TOOL: stock_data_tool, TOOL: exchange_rate_tool and TOOL: sec_parser_tool.",
        instructions="""- If you think the query needs decomposition, then do it
                        - Use the relevant given tools only to retrieve data for numerical or statistical processing.
                        - Make sure to stick to the format of parameters required by the tools.
                        - If required, break down the problem into smaller steps (e.g., fetching data, performing analysis).
                        - Hyde_Multi_Reranker_Tool will help retrieve relevant information from the vector store, use this first.
                        - Them you may use any other tool as you find fit.
                        - Provide detailed explanations of methods or calculations when presenting results.
                        - If the analysis requires user input, request it explicitly.
                        - Moreover generate answer with the whatever information you retrieve [context]
                        - If information is missing, reattempt retrieval or ask for clarification.
                        - If the generated answer is not satisfactory, breakdown the query and ask again once.
                        - If the generated answer is still not satisfactory, ask the user for more information or clarify the query.
                        - Once you have answered the query, use the TOOL: state_change_tool to change the state to the STATE:BaseState.
                        - If you don't have a tool to answer the query satisfactorily, then use TOOL: state_change_tool to change the state to the STATE:BaseState.
                        - Very Important: When you are changing state, do not output anything. Just change the state and follow the instructions given in the state without fail.""",
        model=GeminiModel(),
        tools=[
            hyde_multi_reranker_tool,
            bing_web_search_tool,
            python_calculator_tool,
            wolfram_alpha_tool,
            company_income_statement_tool,
            company_ratios_tool,
            company_key_metrics_tool,
            yahoo_finance_stock_data_tool,
            sec_parser_tool,
            exchange_rate_tool,
        ],
    ),
    "DecisionAndEventBased": BaseState(
        name="DecisionAndEventBased",
        goal="To analyze financial decisions and their impact, and provide insights into event-driven scenarios. Has access to TOOL: financial_rag_tool, TOOL: google_news_tool, TOOL: news_search_tool, TOOL: stock_data_tool, TOOL: bing_web_search and TOOL: wolfram_alpha_query.",
        instructions="""- If you think the query needs decomposition, then do it
                        - Use the given tools only to retrieve relevant event-related or decision-related data.
                        - Hyde_Multi_Reranker_Tool will help retrieve relevant information from the vector store, use this first.
                        - Them you may use any other tool as you find fit.
                        - Provide a well-reasoned analysis grounded in retrieved data about financial decisions or events.
                        - Moreover generate answer with the whatever information you retrieve [context]
                        - Explain the implications of the event or decision clearly.
                        - Ensure responses are grounded in the retrieved data and avoid speculative statements.
                        - If information is missing, reattempt retrieval or ask for clarification.
                        - If the generated answer is not satisfactory, breakdown the query and ask again once.
                        - If the generated answer is still not satisfactory, ask the user for more information or clarify the query.
                        - Once you have answered the query satisfactorily, use the TOOL: state_change_tool to change the state to the STATE:BaseState.
                        - If you don't have a tool to answer the query satisfactorily, then use TOOL: state_change_tool to change the state to the STATE:BaseState.""",
        model=GeminiModel(),
        tools=[
            hyde_multi_reranker_tool,
            google_news_tool,
            ask_news_tool,
            wolfram_alpha_tool,
            bing_web_search_tool,
            yahoo_finance_stock_data_tool,
        ],
    ),
}
alt_states = {
    "BaseState": BaseState(
        name="BaseState",
        goal="To Answer the user's query by routing it to one of the states below [You are the central router]. Do not hallucinate any instructions or create any tools",
        instructions="""
                        You are given a query, classify the query into one of the following categories, and use the TOOL:state_change_tool to switch to the appropriate state. Once transitioned, follow the instructions given in the state without fail:
                        You are given a query, classify the query into one of the following categories, and use the TOOL:state_change_tool to switch to the appropriate state. Once transitioned, follow the instructions given in the state without fail:
                        - If the query concerns finance and asks about a number, change the state to the STATE: FactBasedFinance.
                        - If the query asks about statistical analysis, change the state to the STATE: StatisticalAnalysis.
                        - If the query asks about a decision or event, change the state to the STATE: TrendAnalysisAndEventBased.
                        - If the query asks about a creativity task, change the state to the STATE: CreativeQuery.
                        - Otherwise, change the state to the STATE: NonFinance.
                        When you are not able to classify the query into any of the above categories, clarify the query with the user and repeat the process once.
                        """,
        model=GeminiModel(),
        tools=[],
    ),
    "FactBasedFinance": BaseState(
        name="FactBasedFinance",
        goal="To provide accurate and grounded answers to fact-based financial user queries by following the instructions below.",
        instructions="""- Always use TOOL: financial_rag_tool to retrieve information from the vector store first. Assume that information for the query is present in the vector store. Assume information in the vector store is always up-to-date and accurate.
                        - Move to any other tool only after calling financial_rag_tool.
                        - If unsatisfactory, rewrite the query to make it more comprehensive, try financial_rag_tool again.
                        - If that still doesn't work, break down the query into the sub questions needed to answer and original query.
                        - Only then use the TOOL: bing_web_search to fetch more information.
                        - Ensure the response is grounded on the retrieved data from the above tools.
                        - If you don't have enough information, use other tools available to you or retry with ones you have already tried by decomposing the query
                        - Avoid assuming or hallucinating information or using outdated context.
                        - If the generated answer is satisfactory, then use TOOL: state_change_tool to change to the STATE: BaseState and then follow the instructions given in the state without fail.
                        - If the query is still not answered, report whatever you have and then use TOOL: StateChangeTool to change the state to the STATE:BaseState and then follow the instructions given in the state without fail.
                        - Once you have answered the query satisfactorily, use the TOOL: StateChangeTool to change the state to the STATE:BaseState.""",
        model=GeminiModel(),
        tools=[
            hyde_multi_reranker_tool,
            naive_rag_tool,
            finn_hub_tool,
            edgar_tool,
            company_income_statement_tool,
            company_ratios_tool,
            company_key_metrics_tool,
            bing_web_search_tool,
            sec_parser_tool,
            exchange_rate_tool,
            yahoo_finance_stock_data_tool,
        ],
    ),
    "StatisticalAnalysis": BaseState(
        name="StatisticalAnalysis",
        goal="To provide accurate and grounded answers to Statistical Analysis based user queries by following the instructions below. Provide as much statistical analysis as possible to support your statement.",
        instructions="""- Use the TOOL:naive_rag_tool to retrieve relevant information from the vector store first.
                        - If the retrieved context is not sufficient or unsatisfactory, use the TOOL:financial_rag_tool to retrieve additional information from the vector store.
                        - If still unsatisfactory, then only use the other tools provided to your state to fetch more information.
                        - Ensure the response is grounded on the retrieved data from the above tools.
                        - Avoid assuming or hallucinating information or using outdated context.
                        - After retrieving sufficient context from the above tools, use the TOOL:wolfram_alpha_query or TOOL: Python_Calculator for any mathematical or statistical analysis and calculations.
                        - If the generated answer is satisfactory, then use TOOL: state_change_tool to change to the STATE: BaseState and then follow the instructions given in the state without fail.
                        - If the generated answer is still not satisfactory,ask the user for more information or clarify the query and repeat the process once.
                        - If the query is still not answered then use TOOL: state_change_tool to change the state to the STATE:BaseState and then follow the instructions given in the state without fail.
                        - Very Important is that when you are changing state, do not output anything. Just change the state and follow the instructions given in the state without fail.""",
        model=GeminiModel(),
        tools=[
            hyde_multi_reranker_tool,
            naive_rag_tool,
            wolfram_alpha_tool,
            bing_web_search_tool,
            python_calculator_tool,
            company_income_statement_tool,
            company_ratios_tool,
            company_key_metrics_tool,
            exchange_rate_tool,
            yahoo_finance_stock_data_tool,
        ],
    ),
    "TrendAnalysisAndEventBased": BaseState(
        name="TrendAnalysisAndEventBased",
        goal="To provide accurate and grounded answers to event based user queries and Trend Analysis based user queries by following the instructions below. Give as much numbers and facts as possible to support your statement.",
        instructions="""- The TOOL:financial_rag_tool which has more weightage than other tools to retrieve additional information from the vector store.
                        - Then use all the other relevant information retrieval tools provided to your state to expand your horizons.
                        - Ensure the response is grounded on the retrieved data from the above tools.
                        - After retrieving sufficient context from the above tools, use the TOOL:wolfram_alpha_query or TOOL:Python_Calculator for analyzing mathematical data if any.
                        - Avoid assuming or hallucinating information or using outdated context.
                        - You are only satisfied when you are completely sure that you do not need any more further processing to answer the query. So feel free to use other relevant tools provided to your state.
                        - If the generated answer is satisfactory, then use TOOL: state_change_tool to change to the STATE: BaseState and then follow the instructions given in the state without fail.
                        - If the generated answer is not satisfactory, use TOOL: bing_web_search or TOOL: news_search to expand your horizons .
                        - If the generated answer is still not satisfactory, ask the user for more information or clarify the query and repeat the process once.
                        - If the query is still not answered then use TOOL: state_change_tool to change the state to the STATE:BaseState and then follow the instructions given in the state without fail.
                        - Very Important is that when you are changing state, do not output anything. Just change the state and follow the instructions given in the state without fail.""",
        model=GeminiModel(),
        tools=[
            hyde_multi_reranker_tool,
            naive_rag_tool,
            finn_hub_tool,
            ask_news_tool,
            bing_web_search_tool,
            company_income_statement_tool,
            company_ratios_tool,
            company_key_metrics_tool,
            yahoo_finance_stock_data_tool,
            wolfram_alpha_tool,
            python_calculator_tool,
        ],
    ),
    "CreativeQuery": BaseState(
        name="CreativeQuery",
        goal="To provide accurate and grounded answers to out-of-the-box or unconventional user queries by following the instructions below. Provide as much Creative content as possible to support your statement.",
        instructions="""- Use yourself as a tool to generate creative content. Act as if your temperature parameter is set to 1 [Maximum Creativity] .
                        - If the generated answer is satisfactory, then use TOOL: state_change_tool to change to the STATE:BaseState and then follow the instructions given in the state without fail.
                        - If the generated answer is not satisfactory, ask the user for more information or clarify the query and repeat the process once.
                        - If the query is still not answered then use TOOL: state_change_tool to change the state to the STATE:BaseState and then follow the instructions given in the state without fail.
                        - Very Important is that when you are changing state, do not output anything. Just change the state and follow the instructions given in the state without fail.""",
        model=GeminiModel(),
        tools=[],
    ),
    "NonFinance": BaseState(
        name="NonFinanceFactBased",
        goal="To provide accurate and grounded answers to Non-Financial Fact based  user queries and non-creative user queries by following the instructions below. Cite your information sources as much as possible to support your statement.",
        instructions="""- Use the TOOL:naive_rag_tool to retrieve relevant information from the vector store first.
                        - If the retrieved context is not sufficient or unsatisfactory, use the TOOL:financial_rag_tool to retrieve additional information.
                        - If still unsatisfactory, then only use the other tools provided to your state to fetch more information.
                        - Ensure the response is grounded on the retrieved data from the above tools.
                        - Avoid assuming or hallucinating information or using outdated context.
                        - If the query is still not answered then use TOOL: state_change_tool to change the state to the STATE:BaseState and then follow the instructions given in the state without fail.
                        - If the generated answer is satisfactory, then use TOOL: state_change_tool to change to the STATE:BaseState and then follow the instructions given in the state without fail.
                        - If the generated answer is still not satisfactory, ask the user for more information or clarify the query and repeat the process once.
                        - Very Important is that when you are changing state, do not output anything. Just change the state and follow the instructions given in the state without fail.""",
        model=GeminiModel(),
        tools=[
            hyde_multi_reranker_tool,
            naive_rag_tool,
            bing_web_search_tool,
            news_search_tool,
            python_calculator_tool,
            wolfram_alpha_tool,
        ],
    ),
}
stateless_agent_states = {
    "MasterState": BaseState(
        name="MasterState",
        model=GeminiModel(),
        goal="To try as much as possible to answer the users with complex queries by decomposing/analyzing them and calling the appropriate tool to answer the query.",
        instructions="""
    - you are already given the query with retrieved context from the vector store using the TOOL:financial_rag_tool. If the context is unsatisfactory, proceed with the steps below. Else answer the query using the context provided.
    - Decompose the query into smaller parts such that each sub-query retrieves only one piece of information using only one tool.
    - Add the decomposed queries to the queries list with answers as empty strings.
    - Make a plan on how to answer each sub-query using one tool per sub-query one by one. Do not combine the sub-queries into one tool call. Each sub-query must be answered by separate tool calls.
    - Use the TOOL:financial_rag_tool to retrieve information about SEC filing before using any other tool. While using any tool, ensure that query is a question whose answer is a single fact.
    - Based on the plan, call the tools to answer the sub-queries one by one.
    - If any of the sub-queries are not answered satisfactorily, use other alternative tools to retrieve the information. You can call any number of alternative tools to answer the sub-queries.
    - Once the sub-queries are answered satisfactorily, use the TOOL:wolfram_alpha_query to do any mathematical calculations if required. Do not do any calculations manually by yourself.
    - You can make multiple calls to the same tool if required. Do each tool call step-by-step and update the answer for sub query key after each tool call.
    - Once all the sub-queries are answered satisfactorily, combine the answers of all sub-queries to form the final answer.
    - If the final answer is not satisfactory, ask the user for more information or clarify the query and repeat the process once.
    """,
        tools=[
            hyde_multi_reranker_tool,
            bing_web_search_tool,
            python_calculator_tool,
            wolfram_alpha_tool,
            yahoo_finance_stock_data_tool,
            google_news_tool,
            edgar_tool,
            news_search_tool,
            company_income_statement_tool,
            company_ratios_tool,
            company_key_metrics_tool,
            exchange_rate_tool,
            finn_hub_tool,
        ],
    ),
}
explainability_states = {
    "ExplainabilityState": BaseState(
        name="ExplainabilityState",
        goal="Your task is to explain the user's answer based on the context provided above.",
        instructions="""- The user input is answer given by agent by usage of tools and the respective tool results are also provided
                        - You have to explain from which parts of the tool results the answer was derived from
                        - Quote exact statements/facts which support the answer provided as user input
                        - If available quote the source of the statements like tool name, document name or website name
                        - Do not provide any new information which is not present in the tool results""",
        model=GeminiModel(),
    ),
}
auto_states = {
    "AutoState": BaseState(
        name="AutoState",
        goal="To provide an automated response to the user query based on an optimised but fixed pipeline.",
        instructions="""- Provide an automated response to the user query based on a fixed pipeline.
                        - Use the fixed pipeline provided in the completion function.
                        - Do not use any tools to generate the response.
                        - Do not provide any new information which is not present in the fixed pipeline.""",
        model=GeminiModel(),
        tools=[
            hyde_multi_reranker_tool,  # NOTE - this tool should be placed 1st in the list, as it should be called automatically
            bing_web_search_tool,
            python_calculator_tool,
            yahoo_finance_stock_data_tool,
        ],
    ),
}
mock_states = {
    "MockState": BaseState(
        name="MockState",
        goal="To provide a mock response to the user query.",
        instructions="""- Provide a mock response to the user query.
                        - Use the mock response provided in the completion function.
                        - Do not use any tools to generate the response.
                        - Do not provide any new information which is not present in the mock response.""",
        model=MockModel(max_sleep_time=3),
        tools=[python_calculator_tool],
    )
}


meta_states = multi_state_agent_states | stateless_agent_states
meta_states["BaseState"] = BaseState(
    name="BaseState",
    goal="To Answer the user's query with either the available tools or by switching states.",
    instructions="""- You will be given results from a high quality rag tool financial_rag_tool. Assume these results to be correct. 
                    - You have a set of useful tools. If these may be enough to answer the query, try them first.
                    - If needed, switch to a different state to answer the query. Use the TOOL: StateChangeTool to transition to the appropriate state. Once transitioned follow the instructions given in the state without fail:
                    - If the query is related to finance and is asking about a number then change the state to the STATE:FactBasedFinance Use this also to retrieve all necessary information since it has access to TOOL: naive_rag_tool, TOOL: financial_rag_tool, TOOL: bing_web_search, TOOL: yahoo_finance_tool, and TOOL: sec_parser_tool.  
                    - If the query asks about a numerical analysis or if you need to use calculators or wolfram alpha change the state to the STATE:StatisticalAnalysis. 
                    - If the query asks about a decision or event change, say like access news, use TOOL: StateChangeTool to change the state to the STATE:DecisionAndEventBased.
                    - If the query asks about a creative query change the state to the STATE:CreativeQuery.
                    - If you are not sure which state to transition to, use TOOL: StateChangeTool to change the state to the STATE: FactBasedFinance.
                    - If the query is complex, switch to the STATE:MasterState to decompose the query, plan and answer it with all available tools.
                    - Do not hallucinate any information or additional tool arguments.
    
    """,
    model=GeminiModel(),
    tools=[
        hyde_multi_reranker_tool,
        bing_web_search_tool,
        python_calculator_tool,
        yahoo_finance_stock_data_tool,
    ],
)

hyfer_states = stateless_agent_states
