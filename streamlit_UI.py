# Import necessary libraries
import streamlit as st
from langchain.cache import InMemoryCache
import langchain
from langchain.chat_models import ChatOpenAI
from main_2 import (
    get_stock_dict,
    get_balance_sheet,
    get_balancesheet_summary,
    get_technical_indicators,
    get_tech_summary,
    generate_stock_recommendation,
)

# Initialize an in-memory cache for the language model
langchain.llm_cache = InMemoryCache()

# Display the application title using Markdown
st.markdown("# Welcome to LLMStockMate")

# Provide an overview of the application's functionality using Markdown
st.markdown(
    "LLMStockMate is a comprehensive stock recommendation app that utilizes both financial and technical data to provide buy and sell recommendations."
)

# Check if the OpenAI API key is already stored in the session state
if "openaikey" not in st.session_state:
    # Create an input field for users to enter their OpenAI API key
    openaikey = st.text_input(
        label="Please enter your OpenAI API key:",
        type="password",
        placeholder="Your OpenAI API Key",
    )

    # Verify the provided API key format
    if openaikey and not openaikey.startswith("sk-"):
        # Display an error message if the API key format is incorrect
        st.error("Please provide a valid API key")
    elif openaikey and openaikey.startswith("sk-"):
        # If a valid key is provided, store it in the session state
        st.session_state["openaikey"] = openaikey

        # Initialize the ChatOpenAI instance with the API key and temperature parameter
        st.session_state["llm"] = ChatOpenAI(
            openai_api_key=st.session_state["openaikey"], temperature=0
        )

        # Rerun the application to proceed
        st.rerun()

# Check if the stock database is stored in the session state
if "stock_db" not in st.session_state:
    # Load the stock database using the get_stock_dict function
    st.session_state["stock_db"] = get_stock_dict(stock_list_file_name="final_list.csv")

# Get a list of company names from the stock database
company_names = list(st.session_state["stock_db"]["Symbol"].keys())

if "openaikey" in st.session_state:
    # Create a select box for users to choose a company to analyze
    company = st.selectbox(
        label="Select a stock to analyze",
        options=company_names,
        placeholder="Your favorite stock",
        index=None,
    )

    if company:
        # Retrieve the ticker symbol for the selected company
        ticker = st.session_state["stock_db"]["Symbol"][company]
        st.markdown(f"You have selected `{company}`")
        st.markdown(f"The ticker symbol for `{company}` is `{ticker}`")
        if st.button(label="Analyze"):
            with st.spinner("Processing balance sheet....."):
                # Retrieve the balance sheet data for the selected company
                balance_sheet_json = get_balance_sheet(ticker)

                # Get a summary of the balance sheet using the language model
                balancesheet_result = get_balancesheet_summary(
                    st.session_state["llm"], balance_sheet_json
                )
            with st.expander(label="Balance sheet summary:"):
                st.markdown(f"{balancesheet_result.content}")

            with st.spinner("Processing technical summary....."):
                # Retrieve technical indicators data for the selected company
                technical_indicator_json = get_technical_indicators(ticker)

                # Get a summary of technical indicators using the language model
                technical_result = get_tech_summary(
                    st.session_state["llm"], technical_indicator_json
                )
            with st.expander(label="Technical summary:"):
                st.markdown(f"{technical_result.content}")

            with st.spinner("Processing final recommendation....."):
                recommendation_result = generate_stock_recommendation(
                    financial_summary=balancesheet_result,
                    technical_summary=technical_result,
                    model=st.session_state["llm"],
                )
            st.markdown(recommendation_result.content)