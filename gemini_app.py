import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup
import requests
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from typing import List, Union, Dict
from langchain.agents import AgentOutputParser
from langchain.chains import LLMChain
import re
import pandas as pd
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

# Custom prompt template for the agent
class StockAnalysisPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += f"Action: {action.log}\nObservation: {observation}\n"
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        kwargs["thoughts"] = thoughts
        return self.template.format(**kwargs)

# Initialize Google Generative AI configuration
@st.cache_resource
def load_model():
    GOOGLE_API_KEY = "AIzaSyDKazkFS08O2Z6VCGDB9NBwgt3MM9nWZv8"
    
    # Configure the Google Generative AI client
    genai.configure(api_key=GOOGLE_API_KEY)
    
    # Create a LangChain compatible Gemini model
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.0-flash",
        temperature=0,
        google_api_key=GOOGLE_API_KEY,
        streaming=False
    )
    return llm

# Custom tools for the agent
def get_stock_data(ticker: str) -> Dict:
    """Get basic stock information for a given ticker"""
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "currentPrice": info.get("currentPrice"),
        "marketCap": info.get("marketCap"),
        "longName": info.get("longName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "website": info.get("website"),
        "longBusinessSummary": info.get("longBusinessSummary")
    }

def get_stock_news(ticker: str):
    """
    Fetches the latest news articles for a given stock ticker from Google News.
    
    :param ticker: Stock ticker symbol (e.g., "AAPL" for Apple)
    :return: A list of dictionaries containing news titles and URLs.
    """
    base_url = f"https://www.google.com/search?q={ticker}+stock+news&tbm=nws"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    response = requests.get(base_url, headers=headers)

    if response.status_code != 200:
        print(f"Error fetching data: {response.status_code}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    articles = soup.find_all("div", class_="SoaBEf")

    news_list = []

    for article in articles:
        title_tag = article.find("div", class_="n0jPhd ynAwRc MBeuO nDgy9d")
        if title_tag:
            title = title_tag.text.strip()
            link_tag = article.find("a")
            link = f"https://www.google.com{link_tag['href']}" if link_tag else None
            news_list.append({"title": title, "url": link})

    return news_list


def get_stock_financials(ticker: str) -> Dict:
    """Get key financial metrics for a given stock ticker"""
    stock = yf.Ticker(ticker)
    try:
        financials = stock.financials
        return {
            "Revenue": financials.loc["Total Revenue"].to_dict() if "Total Revenue" in financials.index else {},
            "Net Income": financials.loc["Net Income"].to_dict() if "Net Income" in financials.index else {},
            "Operating Income": financials.loc["Operating Income"].to_dict() if "Operating Income" in financials.index else {}
        }
    except:
        return {"error": "Unable to fetch financial data"}

# Create the agent
def create_stock_agent(llm):
    tools = [
        Tool(
            name="get_stock_info",
            func=get_stock_data,
            description="Get basic information about a stock. Input should be a stock ticker symbol."
        ),
        Tool(
            name="get_stock_news",
            func=get_stock_news,
            description="Get recent news articles about a stock. Input should be a stock ticker symbol."
        ),
        Tool(
            name="get_stock_financials",
            func=get_stock_financials,
            description="Get key financial metrics for a stock. Input should be a stock ticker symbol."
        )
    ]
    
    template = """You are a helpful AI stock analysis assistant. Your goal is to help users analyze stocks based on their questions.

                You have access to the following tools:
                {tools}

                Use the following format:
                Question: the input question you must answer
                Thought: you should always think about what to do
                Action: the action to take, should be one of [{tool_names}]
                Action Input: the input to the action
                Observation: the result of the action
                ... (this Thought/Action/Action Input/Observation can repeat N times)
                Thought: I now know the final answer
                Final Answer: the final answer to the original input question

                Previous conversation:
                {thoughts}

                Question: {input}"""

    prompt = StockAnalysisPromptTemplate(
        template=template,
        tools=tools,
        input_variables=["input", "intermediate_steps"]
    )
    
    class StockAgentOutputParser(AgentOutputParser):
        def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
            if "Final Answer:" in llm_output:
                return AgentFinish(
                    return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                    log=llm_output,
                )
            
            regex = r"Action: (.*?)[\n]*Action Input: (.*)"
            match = re.search(regex, llm_output, re.DOTALL)
            
            if not match:
                return AgentFinish(
                    return_values={"output": "I cannot determine what action to take. Please try rephrasing your question."},
                    log=llm_output,
                )
                
            action = match.group(1).strip()
            action_input = match.group(2).strip()
            
            return AgentAction(tool=action, tool_input=action_input, log=llm_output)
    
    output_parser = StockAgentOutputParser()
    
    agent = LLMSingleActionAgent(
        llm_chain=LLMChain(llm=llm, prompt=prompt),
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=[tool.name for tool in tools]
    )
    
    memory = ConversationBufferMemory(memory_key="thoughts")
    
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True
    )

# Streamlit UI
st.title('🤖 AI Stock Research Assistant')
st.write('Ask me anything about stocks!')

# Initialize session state
if 'agent' not in st.session_state:
    with st.spinner('Loading AI model...'):
        llm = load_model()
        st.session_state.agent = create_stock_agent(llm)

# User input
user_question = st.text_input('Ask a question about a stock (e.g., "What\'s the latest news about AAPL?")', '')

if user_question:
    with st.spinner('Analyzing...'):
        try:
            response = st.session_state.agent.run({"input": user_question})
            st.write(response)
        except Exception as e:
            st.error(f"Error: {str(e)}")

st.markdown("---")
st.caption("Powered by LangChain, Google Gemini Flash 2.0, and Streamlit")

# Example questions
st.sidebar.header("Example Questions")
st.sidebar.write("""
- What's the current stock price of AAPL?
- Tell me about Tesla's recent news and stock performance
- What are the key financial metrics for MSFT?
- Analyze the recent performance of NVDA
- What's the market sentiment for AMZN based on recent news?
""")