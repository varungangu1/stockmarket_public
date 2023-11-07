# Import necessary libraries and modules
from langchain.callbacks import get_openai_callback
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import Field
from langchain.schema import BaseOutputParser
from langchain.cache import InMemoryCache
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
import os
import dotenv
import langchain
import yfinance as yt
import requests


# Create a template to extract the company full name from a query
def get_company_name(llm, query):
    # Template for extracting the company name
    template_company = """You are a stock market expert. Your task is to return the company name from the given query.\n\n{instructions}\n\nQUERY:{query}\n\nCompany name:"""

    class CompanyName(BaseOutputParser):
        company_name: str = Field(description="Company full name")

    parser_company = PydanticOutputParser(pydantic_object=CompanyName)

    prompt_company = PromptTemplate(
        input_variables=["instructions", "query"], template=template_company
    )

    chain_company = LLMChain(llm=llm, prompt=prompt_company)

    result = chain_company.predict(
        instructions=parser_company.get_format_instructions(), query=query
    )

    return result


# Create a function to get the company symbol
def get_symbol(llm, vs, result_company_name):
    doc = vs.similarity_search(query=result_company_name, k=1)

    # Template for extracting the symbol
    template_symbol = (
        """Return the symbol from the given document\n\n{instruction}\n\n{doc}"""
    )

    prompt_symbol = PromptTemplate(
        input_variables=["doc", "instruction"], template=template_symbol
    )

    class GetSymbol(BaseOutputParser):
        symbol: str = Field(description="Company symbol")

        def parse(self, symbol):
            return {"symbol": symbol}

    parser_symbol = PydanticOutputParser(pydantic_object=GetSymbol)

    chain_symbol = LLMChain(llm=llm, prompt=prompt_symbol)

    result = chain_symbol.predict(
        doc=doc, instruction=parser_symbol.get_format_instructions()
    )
    try:
        result = parser_symbol.parse(result)
        return result.symbol
    except:
        return result


# Create a function to get the financial summary
def get_financial_summary(llm, result_symbol):
    result_symbol = result_symbol.replace('"', "")

    company = yt.Ticker(ticker=result_symbol + ".NS")

    balance_sheet = company.balance_sheet
    balance_sheet = balance_sheet.to_string()

    # Template for getting a financial summary
    template_balance_sheet = """You are an expert in the stock market. Your task is to provide a brief summary using the following financial table. Return a list of 10 key points, comprising a balanced mix of positive and negative aspects.\n\n{balance_sheet}"""

    prompt_balance_sheet = PromptTemplate(
        input_variables=["balance_sheet"], template=template_balance_sheet
    )

    chain_balance_sheet = LLMChain(llm=llm, prompt=prompt_balance_sheet)

    result = chain_balance_sheet.predict(balance_sheet=balance_sheet)

    return result


# Create a function to get the technical summary
def get_tec_summary(llm, result_symbol):
    url = f"https://scanner.tradingview.com/symbol?symbol=NSE:{result_symbol}&fields=Recommend.Other,Recommend.All,Recommend.MA,RSI,RSI[1],Stoch.K,Stoch.D,Stoch.K[1],Stoch.D[1],CCI20,CCI20[1],ADX,ADX+DI,ADX-DI,ADX+DI[1],ADX-DI[1],AO,AO[1],AO[2],Mom,Mom[1],MACD.macd,MACD.signal,Rec.Stoch.RSI,Stoch.RSI.K,Rec.WR,W.R,Rec.BBPower,BBPower,Rec.UO,UO,EMA10,close,SMA10,EMA20,SMA20,EMA30,SMA30,EMA50,SMA50,EMA100,SMA100,EMA200,SMA200,Rec.Ichimoku,Ichimoku.BLine,Rec.VWMA,VWMA,Rec.HullMA9,HullMA9,Pivot.M.Classic.S3,Pivot.M.Classic.S2,Pivot.M.Classic.S1,Pivot.M.Classic.Middle,Pivot.M.Classic.R1,Pivot.M.Classic.R2,Pivot.M.Classic.R3,Pivot.M.Fibonacci.S3,Pivot.M.Fibonacci.S2,Pivot.M.Fibonacci.S1,Pivot.M.Fibonacci.Middle,Pivot.M.Fibonacci.R1,Pivot.M.Fibonacci.R2,Pivot.M.Fibonacci.R3,Pivot.M.Camarilla.S3,Pivot.M.Camarilla.S2,Pivot.M.Camarilla.S1,Pivot.M.Camarilla.Middle,Pivot.M.Camarilla.R1,Pivot.M.Camarilla.R2,Pivot.M.Camarilla.R3,Pivot.M.Woodie.S3,Pivot.M.Woodie.S2,Pivot.M.Woodie.S1,Pivot.M.Woodie.Middle,Pivot.M.Woodie.R1,Pivot.M.Woodie.R2,Pivot.M.Woodie.R3,Pivot.M.Demark.S1,Pivot.M.Demark.Middle,Pivot.M.Demark.R1&no_404=true"

    tec = requests.get(url=url)

    # Template for getting a technical summary
    template_tec = """You are an expert in the stock market. Your task is to provide a brief summary using the following technical data. Return a list of 10 key points, comprising a balanced mix of positive and negative aspects.\n\n{tec}"""

    prompt_tec = PromptTemplate(input_variables=["tec"], template=template_tec)

    chain_tec = LLMChain(llm=llm, prompt=prompt_tec)

    result_tec = chain_tec.predict(tec=tec.json())

    return result_tec


# Create a function to get the final result
def get_final_result(llm, result_company_name, result_finance, result_tec):
    template_final = """You are an expert in the stock market. Your task is to return a one-line recommendation on whether to buy the stock or not using the following details.\n\nCompany name: {result_company_name}\n\nDetails:\n\n{result_finance}\n\n{result_tec}\n\nYour helpful answer:"""

    prompt_final = PromptTemplate(
        input_variables=["result_company_name", "result_finance", "result_tec"],
        template=template_final,
    )

    chain_final = LLMChain(llm=llm, prompt=prompt_final)
    result_final = chain_final.predict(
        result_company_name=result_company_name,
        result_finance=result_finance,
        result_tec=result_tec,
    )
    return result_final


# Create a function to get the overall answer
def get_answer(
    get_company_name,
    get_symbol,
    get_financial_summary,
    get_tec_summary,
    get_final_result,
    llm,
    vs,
    query,
):
    # Get the company name
    result_company_name = get_company_name(llm, query)
    print(f"Company Name: {result_company_name}")

    # Get the symbol name
    result_symbol = get_symbol(llm, vs, result_company_name)
    print(f"Company symbol: {result_symbol}")

    # Get financial insight
    result_finance = get_financial_summary(llm, result_symbol)
    print(f"Company financial status:\n\n{result_finance}")

    # Get technical summary
    result_tec = get_tec_summary(llm, result_symbol)
    print(f"Company technical status:\n\n{result_tec}")

    # Get the final result
    result_final = get_final_result(
        llm, result_company_name, result_finance, result_tec
    )
    return result_final


if __name__ == "__main__":
    # Load environment variables
    dotenv.load_dotenv()

    # Create a local cache
    langchain.llm_cache = InMemoryCache()

    # Initialize the LLM class
    llm = ChatOpenAI(temperature=0, openai_api_key=os.getenv("openai_api_key"))

    # Create an embedding class
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("openai_api_key"))

    # Load embeddings for the ticker retriever
    vs = FAISS.load_local(folder_path="vectorstore", embeddings=embeddings)

    # Query to start with
    query = "Is it a good time to invest in Tata Steel Ltd"

    with get_openai_callback() as cb:
        result = get_answer(
            get_company_name,
            get_symbol,
            get_financial_summary,
            get_tec_summary,
            get_final_result,
            llm,
            vs,
            query,
        )
        # print(cb)

    print(f"Final recommendation: {result}")
