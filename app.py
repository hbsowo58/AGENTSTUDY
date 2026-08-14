import streamlit as st
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI 챗봇", layout="wide")
st.title("💬 AI 챗봇")

if "messages" not in st.session_state:
    st.session_state.messages = []

llm = ChatOpenAI(model_name="gpt-4o")

# 대화 히스토리 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력
if prompt := st.chat_input("메시지를 입력하세요..."):
    # 사용자 메시지 저장 & 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # AI 응답 - 전체 대화 히스토리를 전달
    with st.chat_message("assistant"):
        # LangChain 형식으로 변환
        langchain_messages = [
            ("system", "당신은 친절한 AI 어시스턴트입니다."),
            *[(msg["role"], msg["content"]) for msg in st.session_state.messages]
        ]
        response = llm.invoke(langchain_messages)
        st.markdown(response.content)
    
    # 히스토리에 저장
    st.session_state.messages.append({"role": "assistant", "content": response.content})