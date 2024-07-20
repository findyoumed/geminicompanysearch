import os
import streamlit as st
import requests
from duckduckgo_search import DDGS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import logging
from functools import lru_cache
from langchain.globals import set_verbose
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# verbose 설정
set_verbose(False)  # 또는 True, 필요에 따라 설정

# Google API 키 설정
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY is not set in the environment variables")

# Alpha Vantage API 키 설정
alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
if not alpha_vantage_api_key:
    raise ValueError("ALPHA_VANTAGE_API_KEY is not set in the environment variables")

# LangChain 모델 설정
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)

@lru_cache(maxsize=100)
def search_stock_symbol(query):
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=1)]
        return results[0]['body'] if results else None
    except Exception as e:
        logger.error(f"Error searching stock symbol: {e}")
        return None

@lru_cache(maxsize=100)
def get_alpha_vantage_data(function, symbol):
    try:
        r = requests.get(
            f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&apikey={alpha_vantage_api_key}"
        )
        r.raise_for_status()
        data = r.json()
        if "Error Message" in data:
            logger.error(f"Alpha Vantage API error for {function}: {data['Error Message']}")
            return None
        if "Information" in data and "standard API rate limit" in data["Information"]:
            logger.warning("Alpha Vantage API rate limit reached")
            return "RATE_LIMIT_EXCEEDED"
        return data
    except requests.RequestException as e:
        logger.error(f"Request failed for {function}: {e}")
        return None

def get_company_overview(symbol):
    return get_alpha_vantage_data("OVERVIEW", symbol)

def get_income_statement(symbol):
    data = get_alpha_vantage_data("INCOME_STATEMENT", symbol)
    return data.get("annualReports", []) if data and data != "RATE_LIMIT_EXCEEDED" else None

def get_stock_performance(symbol):
    data = get_alpha_vantage_data("TIME_SERIES_WEEKLY", symbol)
    if data and data != "RATE_LIMIT_EXCEEDED" and "Weekly Time Series" in data:
        return list(data["Weekly Time Series"].items())[:4]  # 최근 4주 데이터만 반환
    return None

def analyze_company(company_name):
    symbol_info = search_stock_symbol(f"{company_name} stock symbol")
    if not symbol_info:
        return "죄송합니다. 해당 회사의 주식 심볼을 찾을 수 없습니다."
    
    symbol = symbol_info.split()[-1]
    logger.info(f"Found symbol: {symbol} for company: {company_name}")

    overview = get_company_overview(symbol)
    income_statement = get_income_statement(symbol)
    performance = get_stock_performance(symbol)

    if overview == "RATE_LIMIT_EXCEEDED" or income_statement == "RATE_LIMIT_EXCEEDED" or performance == "RATE_LIMIT_EXCEEDED":
        return "죄송합니다. API 사용량 제한에 도달했습니다. 잠시 후 다시 시도해 주세요."

    if not overview or not income_statement or not performance:
        logger.error(f"Failed to retrieve data for {company_name} ({symbol})")
        return "죄송합니다. 회사 정보를 가져오는 데 문제가 발생했습니다. 잠시 후 다시 시도해 주세요."

    template = """
    당신은 헤지펀드 매니저입니다. 다음 정보를 바탕으로 {company_name}({symbol})의 주식이 매수할 만한지 평가해주세요:

    회사 개요: {overview}
    손익계산서: {income_statement}
    주가 성과 (최근 4주):
    {performance}

    주식의 성과, 회사 개요, 손익계산서를 고려하여 분석해주세요.
    판단에 있어 확신을 가지고 주식을 추천하거나 사용자에게 매수하지 말 것을 조언하세요.
    응답은 다음 형식으로 작성해주세요:
    1. 회사 소개 (1-2문장)
    2. 재무 상황 분석 (2-3문장)
    3. 최근 주가 동향 (1-2문장)
    4. 매수 추천 여부 및 이유 (2-3문장)
    """

    prompt = PromptTemplate(
        input_variables=["company_name", "symbol", "overview", "income_statement", "performance"],
        template=template,
    )

    chain = (
        RunnablePassthrough.assign(
            company_name=lambda x: x["company_name"],
            symbol=lambda x: x["symbol"],
            overview=lambda x: x["overview"],
            income_statement=lambda x: x["income_statement"][:2] if x["income_statement"] else [],
            performance=lambda x: x["performance"],
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    try:
        response = chain.invoke({
            "company_name": company_name,
            "symbol": symbol,
            "overview": overview,
            "income_statement": income_statement,
            "performance": performance
        })
        return response
    except Exception as e:
        logger.error(f"Error in LangChain processing: {e}")
        return "죄송합니다. 분석 과정에서 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."

# Streamlit 페이지 설정
st.set_page_config(page_title="InvestorGPT", page_icon="💼")

st.markdown("""
# InvestorGPT
            
InvestorGPT에 오신 것을 환영합니다.
            
관심 있는 회사의 이름을 적으면 우리의 AI가 당신을 위해 분석을 수행할 것입니다.
""")

company = st.text_input("관심 있는 회사의 이름을 적어주세요.")

if company:
    with st.spinner('분석 중입니다...'):
        result = analyze_company(company)
    st.write(result)

    # 디버깅 정보 표시 (개발 중에만 사용)
    if st.checkbox("Show debug info"):
        st.write("Debug Information:")
        st.write(f"Company: {company}")
        st.write(f"Symbol: {result.split('(')[-1].split(')')[0] if '(' in result else 'Not found'}")
        st.write("API Responses:")
        st.write(get_company_overview(result.split('(')[-1].split(')')[0] if '(' in result else ''))
        st.write(get_income_statement(result.split('(')[-1].split(')')[0] if '(' in result else ''))
        st.write(get_stock_performance(result.split('(')[-1].split(')')[0] if '(' in result else ''))