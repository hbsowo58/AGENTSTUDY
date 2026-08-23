import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import before_model
from langchain_tavily import TavilySearch

load_dotenv()

st.set_page_config(page_title="LangGraph AI 챗봇", layout="wide")
st.title("💬 LangGraph 기반 AI 챗봇")

# -------------------------
# 1) 모델과 검색 도구 설정
# -------------------------
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY가 설정되지 않았습니다. .env 또는 Streamlit secrets에 넣어주세요.")
    st.stop()

llm = ChatOpenAI(model="gpt-4o", api_key=api_key)

try:
    tavily_api_key = st.secrets["TAVILY_API_KEY"]
except Exception:
    tavily_api_key = os.getenv("TAVILY_API_KEY")

tools = []
if tavily_api_key:
    tools.append(TavilySearch(max_results=3, tavily_api_key=tavily_api_key))

BLOCKED_WORDS = ["바보", "멍청이", "나쁜말"]

# -------------------------
# 2) 입력 필터 미들웨어
# -------------------------
@before_model
def content_filter(state: AgentState, runtime):
    last_message = state["messages"][-1] if state["messages"] else None
    content = getattr(last_message, "content", str(last_message))

    if not content.strip():
        raise ValueError("질문을 입력해주세요.")

    for word in BLOCKED_WORDS:
        if word in content:
            raise ValueError("부적절한 표현이 포함되어 있어 요청을 처리할 수 없습니다.")


# -------------------------
# 3) 에이전트 구성
# -------------------------
if "graph" not in st.session_state:
    st.session_state.graph = create_agent(
        model=llm,
        tools=tools,
        middleware=[content_filter],
    )

# -------------------------
# 4) 세션 상태와 대화 화면
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 대화 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 입력
if prompt := st.chat_input("질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        result = st.session_state.graph.invoke(
            {"messages": st.session_state.messages}
        )
        answer = result["messages"][-1].content
        used_search = any(
            getattr(message, "type", "") == "tool"
            and getattr(message, "name", "") == "tavily_search"
            for message in result["messages"]
        )

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
        with st.chat_message("assistant"):
            if used_search:
                st.caption("웹 검색을 사용해 답변했습니다.")
            st.markdown(answer)
    except ValueError as error:
        st.warning(str(error))
    except Exception as error:
        st.error(f"에이전트 실행 중 오류가 발생했습니다: {error}")
