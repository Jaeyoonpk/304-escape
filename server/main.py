import os
from fastapi import FastAPI
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import create_agent
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.documents import Document

load_dotenv()

embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("GMS_KEY"),
)

vector_store = Chroma(
    collection_name="books",
    embedding_function=embeddings,
    persist_directory="./chroma_store",
)


@tool(response_format="content_and_artifact")
def search_my_clasroom_info(query: str) -> tuple[str, list[Document]]:      # RAG사용할때 주석도 읽기 떄문에 주석도 수정해야함.
    """
    우리 반(SSAFY 16기 서울 7반) 정보를 검색합니다.
    반 정보(담임, 반장, CA, 교실, 식당, 총원)와 반 학생 23명의 정보
    (출생연도, 성별, 전공, 사는 곳, 역삼까지 거리, 취미, 요즘 상태, 최근 한 일,
    MBTI, 특이사항, 마니또 추측, 마니또에게 준 것)를 찾을 수 있습니다.
    사람 이름이 나오면 이름을 검색어에 꼭 포함하세요.

    Args:
        query (str): 검색할 핵심 질의.

    Returns:
        tuple[str, list[Document]]: 검색된 문서의 직렬화된 문자열과 원본 문서 객체 리스트(artifact)로 구성된 튜플(tuple)

    Examples:
        - "반장이 누구야?"
        - "우리반의 총 인원은 몇명이야?"
        - "동탄에 사는 사람"
        - "박현모 취미"
        - "이재원 마니또 추측"
    """

    retrieved_docs = vector_store.similarity_search(query, k=5)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


SYSTEM_PROMPT = """
너는 SSAFY 16기 서울 7반의 우리반 챗봇이다.
항상 한국어 존댓말로 답변한다.
다음 툴을 사용할 수 있다.
  - search_my_clasroom_info : 우리반 정보나 반 학생에 대한 질문일 때 반드시 사용
반 정보나 반 학생에 대한 질문은 검색 결과에 있는 내용으로만 답하고,
검색 결과에 없으면 모른다고 말한다. 추측해서 지어내지 않는다.
여러 사람을 묻는 질문(예: 동탄에 사는 사람)은 필요하면 검색을 여러 번 해서 빠짐없이 답한다.
일반적인 질문은 툴을 사용하지 않고 답변한다.
"""

llm = ChatOpenAI(
    model=os.getenv("API_MODEL"),
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("GMS_KEY"),
)

# 에이전트 생성
agent = create_agent(
    model=llm,
    tools=[search_my_clasroom_info],
    system_prompt=SYSTEM_PROMPT,
)

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    prompt: str


class ChatResponse(BaseModel):
    answer: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "documents": vector_store._collection.count()}


@app.post("/api/v1/chat", response_model=ChatResponse)
def create_user(chat_request: ChatRequest) -> ChatResponse:
    response = agent.invoke({"messages": [("user", chat_request.prompt)]})
    final_answer = response["messages"][-1].content
    if isinstance(final_answer, list):
        final_answer = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in final_answer
        )
    return {"answer": final_answer}