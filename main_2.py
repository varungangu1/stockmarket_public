import requests
import dotenv
import os
import langchain
from langchain.chat_models import ChatOpenAI
from langchain.cache import InMemoryCache
from langchain.prompts import ChatPromptTemplate
import yfinance as yt
import pandas as pd

# Define the filename of the CSV file containing the stock list
stock_list_file_name = "final_list.csv"


# Function to read the stock list from a CSV file and convert it into a dictionary
def get_stock_dict(stock_list_file_name: str) -> dict:
    """
    Read a stock list from a CSV file and convert it into a dictionary.

    Args:
        stock_list_file_name (str): The filename of the CSV file containing the stock list.

    Returns:
        dict: A dictionary where the keys are Company Names and the values are stock information.

    This function reads the CSV file, sets the "Company Name" column as the index, removes any rows with missing data, and
    converts the DataFrame into a dictionary.

    Example usage:
    stock_db = get_stock_dict("final_list.csv")
    """
    # Read the stock list from the CSV file, using "Company Name" as the index column
    df = pd.read_csv(filepath_or_buffer=stock_list_file_name, index_col="Company Name")

    # Remove rows with missing data (NaN values)
    df.dropna(inplace=True)

    # Convert the DataFrame to a dictionary
    stock_db = df.to_dict()

    return stock_db


# Function to retrieve the balance sheet for a given stock ticker
def get_balance_sheet(ticker: str) -> str:
    """
    Retrieve the balance sheet for a given stock ticker.

    Args:
        ticker (str): The stock ticker symbol for the company (e.g., AAPL for Apple Inc.).

    Returns:
        str: A JSON representation of the balance sheet data.

    This function uses the yfinance library to fetch the balance sheet information for the specified stock ticker.
    It converts the column names to a specific date format and returns the balance sheet data as a JSON string.

    Example usage:
    balance_sheet_data = get_balance_sheet("AAPL")
    """
    # Create a yfinance Ticker object for the specified stock ticker with .NS extension (for NSE)
    company = yt.Ticker(ticker=ticker + ".NS")

    # Retrieve the balance sheet data
    balance_sheet = company.balance_sheet

    # Format the column names to a specific date format ("%d-%b-%Y")
    balance_sheet.columns = balance_sheet.columns.strftime("%d-%b-%Y")

    # Convert the balance sheet data to a JSON representation
    balance_sheet_json = balance_sheet.to_json()

    return balance_sheet_json


# Function to generate a summary of a company's balance sheet using a language model
def get_balancesheet_summary(llm, balance_sheet_json):
    # Define a system template for the conversation
    system_template = """You are a helpful assistant who is an expert in the stock market. Your task is to summarize the company's balance sheet delimited by triple quotes.
    Provide your response in a bullet-point format, encompassing both favorable and unfavorable aspects."""

    # Define a user template with the balance sheet data to be summarized
    user_template = '"""{balance_sheet}"""'

    # Create a ChatPromptTemplate with system and user messages
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_template), ("human", user_template)]
    )

    # Define a conversation chain by combining the prompt and the language model (llm)
    chain = prompt | llm

    # Invoke the conversation chain by providing input, including the balance_sheet_json
    result = chain.invoke(input={"balance_sheet": balance_sheet_json})

    # Return the result of the summary generated by the language model
    return result


# Function to retrieve technical indicators for a stock given its ticker symbol
def get_technical_indicators(ticker: str) -> dict:
    """
    Retrieve technical indicators for a stock using its ticker symbol.

    Args:
        ticker (str): The stock's ticker symbol (e.g., AAPL for Apple Inc.).

    Returns:
        dict: A dictionary containing technical indicators data.

    This function constructs a URL to fetch technical indicators data from the TradingView API based on the stock's ticker symbol.
    It sends a GET request to the API, checks if the response status code is 200 (indicating success), and returns the response data in JSON format.

    Example usage:
    technical_indicators = get_technical_indicators("AAPL")
    """
    # Construct the URL to request technical indicators data for the specified stock
    tradingview_url = f"https://scanner.tradingview.com/symbol?symbol=NSE:{ticker}&fields=Recommend.Other,Recommend.All,Recommend.MA,RSI,RSI[1],Stoch.K,Stoch.D,Stoch.K[1],Stoch.D[1],CCI20,CCI20[1],ADX,ADX+DI,ADX-DI,ADX+DI[1],ADX-DI[1],AO,AO[1],AO[2],Mom,Mom[1],MACD.macd,MACD.signal,Rec.Stoch.RSI,Stoch.RSI.K,Rec.WR,W.R,Rec.BBPower,BBPower,Rec.UO,UO,EMA10,close,SMA10,EMA20,SMA20,EMA30,SMA30,EMA50,SMA50,EMA100,SMA100,EMA200,SMA200,Rec.Ichimoku,Ichimoku.BLine,Rec.VWMA,VWMA,Rec.HullMA9,HullMA9,Pivot.M.Classic.S3,Pivot.M.Classic.S2,Pivot.M.Classic.S1,Pivot.M.Classic.Middle,Pivot.M.Classic.R1,Pivot.M.Classic.R2,Pivot.M.Classic.R3,Pivot.M.Fibonacci.S3,Pivot.M.Fibonacci.S2,Pivot.M.Fibonacci.S1,Pivot.M.Fibonacci.Middle,Pivot.M.Fibonacci.R1,Pivot.M.Fibonacci.R2,Pivot.M.Fibonacci.R3,Pivot.M.Camarilla.S3,Pivot.M.Camarilla.S2,Pivot.M.Camarilla.S1,Pivot.M.Camarilla.Middle,Pivot.M.Camarilla.R1,Pivot.M.Camarilla.R2,Pivot.M.Camarilla.R3,Pivot.M.Woodie.S3,Pivot.M.Woodie.S2,Pivot.M.Woodie.S1,Pivot.M.Woodie.Middle,Pivot.M.Woodie.R1,Pivot.M.Woodie.R2,Pivot.M.Woodie.R3,Pivot.M.Demark.S1,Pivot.M.Demark.Middle,Pivot.M.Demark.R1&no_404=true"

    # Send a GET request to the TradingView API
    response = requests.get(url=tradingview_url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Return the response data in JSON format
        return response.json()


# Function to generate a summary of technical indicator values for a stock using a language model
def get_tech_summary(llm, technical_indicator_json):
    """
    Generate a summary of technical indicator values for a stock using a language model.

    Args:
        llm: An instance of the language model.
        technical_indicator_json (dict): A dictionary containing technical indicator values.

    Returns:
        str: A summary of the technical indicator values.

    This function creates a conversation prompt with a system message that introduces the assistant's task and a user message containing the technical indicator data.
    It uses a language model to generate a summary in a bullet-point format, encompassing both favorable and unfavorable aspects.

    Example usage:
    summary = get_tech_summary(llm, technical_indicator_data)
    """
    # Define a system template for the conversation
    system_template = """You are a helpful assistant who is an expert in the stock market. Your task is to summarize the company's technical indicator values delimited by triple quotes.
    Provide your response in a bullet-point format, encompassing both favorable and unfavorable aspects."""

    # Define a user template with the technical indicator data to be summarized
    user_template = '"""{technical_indicator}"""'

    # Create a ChatPromptTemplate with system and user messages
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_template), ("human", user_template)]
    )

    # Define a conversation chain by combining the prompt and the language model (llm)
    chain = prompt | llm

    # Invoke the conversation chain by providing input, including the technical_indicator_json
    result = chain.invoke(input={"technical_indicator": technical_indicator_json})

    # Return the result of the summary generated by the language model
    return result


# # Load environment variables from a .env file
# dotenv.load_dotenv()

# # Create an in-memory cache for the language model
# langchain.llm_cache = InMemoryCache()

# # Initialize the ChatOpenAI instance with your OpenAI API key and set the temperature parameter
# llm = ChatOpenAI(openai_api_key=os.getenv("openai_api_key"), temperature=0)

# # Call the get_stock_dict function to load the stock list into a dictionary
# stock_db = get_stock_dict(stock_list_file_name)

# # Get company names as a list using stock_db keys
# company_names = list(stock_db["Symbol"].keys())

# # Get the stock ticker for the first company
# ticker = stock_db["Symbol"][company_names[0]]

# # Call the get_balance_sheet function to retrieve the balance sheet data for the specified company
# balance_sheet_json = get_balance_sheet(ticker)

# # Generate a summary of the balance sheet data using the language model and store the result
# balancesheet_result = get_balancesheet_summary(llm, balance_sheet_json)

# # Print the content of the generated summary
# print(balancesheet_result.content)

# # Generate a summary of the technical indicator data using the language model and store the result
# technical_indicator_json = get_technical_indicators(ticker)

# # Call the get_tech_summary function to generate a summary of technical indicators using the language model
# technical_result = get_tech_summary(llm, technical_indicator_json)

# print(technical_result.content)
