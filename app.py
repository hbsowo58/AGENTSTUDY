import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import AIMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.types import Command

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

tavily_tool = None
if tavily_api_key:
    tavily_tool = TavilySearch(max_results=3, tavily_api_key=tavily_api_key)

BLOCKED_WORDS = ["바보", "멍청이", "나쁜말"]


class State(MessagesState):
    plan: list[str]
    query: str = ""
    research_summary: str = ""


# -------------------------
# 2) 입력 검증
# -------------------------
def validate_user_input(content: str):
    content = (content or "").strip()
    if not content:
        raise ValueError("질문을 입력해주세요.")

    for word in BLOCKED_WORDS:
        if word in content:
            raise ValueError("부적절한 표현이 포함되어 있어 요청을 처리할 수 없습니다.")

    return content


# -------------------------
# 3) 그래프 노드 정의
# -------------------------
def planning_node(state: State):
    last_message = state["messages"][-1] if state["messages"] else None
    content = getattr(last_message, "content", str(last_message)) if last_message else ""
    content = validate_user_input(content)

    needs_search = any(
        keyword in content
        for keyword in ["최신", "최근", "오늘", "뉴스", "동향", "시장", "트렌드", "무엇", "어떤"]
    )

    if tavily_tool and needs_search:
        plan = ["웹검색", "답변생성"]
    else:
        plan = ["답변생성"]

    return {
        "plan": plan,
        "query": content,
        "current_task": plan[0],
    }


def supervisor_node(state: State):
    task = state.get("plan", ["답변생성"])[0]

    if task == "웹검색" and tavily_tool:
        return Command(goto="research")
    return Command(goto="final")


def research_node(state: State):
    query = state.get("query", "")
    search_result = tavily_tool.invoke(query)

    summary = (
        f"사용자 질문: {query}\n\n"
        f"검색 결과:\n{search_result}"
    )

    return {
        "research_summary": summary,
        "messages": [AIMessage(content="웹 검색을 수행해 관련 정보를 정리하고 있습니다.", name="research")],
    }


def final_node(state: State):
    query = state.get("query", "")
    search_summary = state.get("research_summary", "")

    if search_summary:
        prompt = (
            "다음 정보를 바탕으로 사용자의 질문에 답변하세요.\n\n"
            f"질문: {query}\n\n"
            f"검색 결과:\n{search_summary}\n\n"
            "답변은 사실 기반으로, 간결하고 도움이 되는 한국어로 작성하세요."
        )
    else:
        prompt = (
            "다음 질문에 대해 자연스럽고 정확하게 답변하세요.\n\n"
            f"질문: {query}"
        )

    response = llm.invoke(prompt)
    return {
        "messages": [AIMessage(content=response.content, name="assistant")],
    }


# -------------------------
# 4) 그래프 구성
# -------------------------
if "graph" not in st.session_state:
    graph_builder = StateGraph(State)
    graph_builder.add_node("planning", planning_node)
    graph_builder.add_node("supervisor", supervisor_node)
    graph_builder.add_node("research", research_node)
    graph_builder.add_node("final", final_node)

    graph_builder.add_edge(START, "planning")
    graph_builder.add_edge("planning", "supervisor")
    graph_builder.add_edge("supervisor", "research")
    graph_builder.add_edge("supervisor", "final")
    graph_builder.add_edge("research", "final")
    graph_builder.add_edge("final", END)

    st.session_state.graph = graph_builder.compile()


# -------------------------
# 5) 세션 상태와 대화 화면
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if prompt := st.chat_input("질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        validate_user_input(prompt)
        result = st.session_state.graph.invoke({"messages": st.session_state.messages})
        answer = result["messages"][-1].content

        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            if "검색" in answer or "검색 결과" in answer:
                st.caption("웹 검색을 사용해 답변했습니다.")
            st.markdown(answer)
    except ValueError as error:
        st.warning(str(error))
    except Exception as error:
        st.error(f"에이전트 실행 중 오류가 발생했습니다: {error}")
