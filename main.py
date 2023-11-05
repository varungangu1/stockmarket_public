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
from langchain.callbacks import get_openai_callback


# creating template to extract company full name from query
def get_company_name(llm, query):
    template_company = """You are stock market expert. Your task is to return the company name from given query.\n\n{instructions}\n\nQUERY:{query}\n\nCompany name:"""

    class Company_name(BaseOutputParser):
        company_name: str = Field(description="Company full name")

    parser_company = PydanticOutputParser(pydantic_object=Company_name)

    prompt_company = PromptTemplate(
        input_variables=["instructions", "query"], template=template_company
    )

    chain_company = LLMChain(llm=llm, prompt=prompt_company)

    result = chain_company.predict(
        instructions=parser_company.get_format_instructions(), query=query
    )

    return result


def get_symbol(llm, vs, result_company_name):
    doc = vs.similarity_search(query=result_company_name, k=1)

    template_symbol = (
        """return the symbol from given document\n\n{instruction}\n\n{doc}"""
    )

    prompt_symbol = PromptTemplate(
        input_variables=["doc", "instruction"], template=template_symbol
    )

    class GetSymbol(BaseOutputParser):
        symbol: str = Field(description="company symbol")

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


def get_financial_summary(llm, result_symbol):
    result_symbol = result_symbol.replace('"', "")

    company = yt.Ticker(ticker=result_symbol + ".NS")

    balance_sheet = company.balance_sheet

    balance_sheet = balance_sheet.to_string()

    template_balancesheet = """You are expert in stock market. Your task is to give brief summary using below financial table. return all the important points.
    
    {balance_sheet}"""

    prompt_balance_sheet = PromptTemplate(
        input_variables=["balance_sheet"], template=template_balancesheet
    )

    chain_balance_sheet = LLMChain(llm=llm, prompt=prompt_balance_sheet)

    result = chain_balance_sheet.predict(balance_sheet=balance_sheet)

    return result


def get_tec_summary(llm, result_symbol):
    url = f"https://scanner.tradingview.com/symbol?symbol=NSE:{result_symbol}&fields=Recommend.Other,Recommend.All,Recommend.MA,RSI,RSI[1],Stoch.K,Stoch.D,Stoch.K[1],Stoch.D[1],CCI20,CCI20[1],ADX,ADX+DI,ADX-DI,ADX+DI[1],ADX-DI[1],AO,AO[1],AO[2],Mom,Mom[1],MACD.macd,MACD.signal,Rec.Stoch.RSI,Stoch.RSI.K,Rec.WR,W.R,Rec.BBPower,BBPower,Rec.UO,UO,EMA10,close,SMA10,EMA20,SMA20,EMA30,SMA30,EMA50,SMA50,EMA100,SMA100,EMA200,SMA200,Rec.Ichimoku,Ichimoku.BLine,Rec.VWMA,VWMA,Rec.HullMA9,HullMA9,Pivot.M.Classic.S3,Pivot.M.Classic.S2,Pivot.M.Classic.S1,Pivot.M.Classic.Middle,Pivot.M.Classic.R1,Pivot.M.Classic.R2,Pivot.M.Classic.R3,Pivot.M.Fibonacci.S3,Pivot.M.Fibonacci.S2,Pivot.M.Fibonacci.S1,Pivot.M.Fibonacci.Middle,Pivot.M.Fibonacci.R1,Pivot.M.Fibonacci.R2,Pivot.M.Fibonacci.R3,Pivot.M.Camarilla.S3,Pivot.M.Camarilla.S2,Pivot.M.Camarilla.S1,Pivot.M.Camarilla.Middle,Pivot.M.Camarilla.R1,Pivot.M.Camarilla.R2,Pivot.M.Camarilla.R3,Pivot.M.Woodie.S3,Pivot.M.Woodie.S2,Pivot.M.Woodie.S1,Pivot.M.Woodie.Middle,Pivot.M.Woodie.R1,Pivot.M.Woodie.R2,Pivot.M.Woodie.R3,Pivot.M.Demark.S1,Pivot.M.Demark.Middle,Pivot.M.Demark.R1&no_404=true"

    tec = requests.get(url=url)

    template_tec = """You are expert in stock market. Your task is to give brief summary using following technical data. return all the important points
    
    {tec}"""

    prompt_tec = PromptTemplate(input_variables=["tec"], template=template_tec)

    chain_tec = LLMChain(llm=llm, prompt=prompt_tec)

    result_tec = chain_tec.predict(tec=tec.json())

    return result_tec


def get_final_result(llm, result_company_name, result_finance, result_tec):
    template_final = """You are expert in stock market. Your task is to return one line recommendation weather to buy the stock or not using following details.
    company name: {result_company_name}

    Details:
    
    {result_finance}
    
    {result_tec}
    
    Your helpful answer:
    """

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
    # get company name
    result_company_name = get_company_name(llm, query)
    print(result_company_name)

    # get symbol name
    result_symbol = get_symbol(llm, vs, result_company_name)
    print(result_symbol)

    # get financial insight
    result_finance = get_financial_summary(llm, result_symbol)
    print(result_finance)

    # get technical summary
    result_tec = get_tec_summary(llm, result_symbol)
    print(result_tec)

    # get final result
    result_final = get_final_result(
        llm, result_company_name, result_finance, result_tec
    )
    return result_final


# load environment variables
dotenv.load_dotenv()

# creating local cache
langchain.llm_cache = InMemoryCache()

# initiating llm class
llm = ChatOpenAI(temperature=0, openai_api_key=os.getenv("openai_api_key"))

# create embedding class
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("openai_api_key"))

# load embedding for ticker retriever
vs = FAISS.load_local(folder_path="vectorstore", embeddings=embeddings)

# Query to start with
query = "is it good time to invest in Vodafone Idea Ltd"

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
    print(cb)

print(result)
