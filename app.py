import os
import streamlit as st
from typing import TypedDict, Annotated
from operator import add
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv()

st.set_page_config(page_title="LangGraph AI 챗봇", layout="wide")
st.title("💬 LangGraph 기반 AI 챗봇")

# -------------------------
# 1) 상태 정의 (5.2)
# -------------------------
class State(TypedDict):
    messages: Annotated[list[str], add]
    question_length: int

# -------------------------
# 2) OpenAI 설정
# -------------------------
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY가 설정되지 않았습니다. .env 또는 Streamlit secrets에 넣어주세요.")
    st.stop()

llm = ChatOpenAI(model="gpt-4o", api_key=api_key)

# -------------------------
# 3) 그래프 노드 함수 (5.3)
# -------------------------
def guardrail(state: State) -> State:
    # 마지막 사용자 메시지를 기준으로 질문 길이를 판별
    last_message = state["messages"][-1]
    return {
        "question_length": len(last_message.strip())
    }


def chatbot(state: State) -> State:
    history = state["messages"]

    # 대화 기록 전체를 LLM에 전달해서 기억 유지
    langchain_messages = [("system", "당신은 친절하고 정확한 AI 어시스턴트입니다.")]

    for i, msg in enumerate(history):
        role = "human" if i % 2 == 0 else "ai"
        langchain_messages.append((role, msg))

    response = llm.invoke(langchain_messages)
    answer = response.content

    return {
        "messages": [answer]
    }


def route(state: State) -> str:
    # 질문 길이가 짧거나 비어 있으면 종료
    if state["question_length"] <= 1:
        return END
    return "chatbot"


# -------------------------
# 4) 그래프 구성
# -------------------------
if "graph" not in st.session_state:
    builder = StateGraph(State)

    builder.add_node("guardrail", guardrail)
    builder.add_node("chatbot", chatbot)

    builder.add_edge(START, "guardrail")
    builder.add_conditional_edges(
        "guardrail",
        route,
        {
            "chatbot": "chatbot",
            END: END,
        },
    )
    builder.add_edge("chatbot", END)

    st.session_state.graph = builder.compile()

# -------------------------
# 5) 세션 상태 초기화
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 대화 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 입력
if prompt := st.chat_input("질문을 입력하세요..."):
    # 사용자 메시지 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 그래프에 넘길 전체 대화 기록 만들기
    history_texts = [msg["content"] for msg in st.session_state.messages]
    current_state = {
        "messages": history_texts,
        "question_length": len(prompt.strip()),
    }

    result = st.session_state.graph.invoke(current_state)

    # 마지막 답변 추출
    answer = result["messages"][-1]

    # 답변 저장 및 표시
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)