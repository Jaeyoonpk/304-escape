import json
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

load_dotenv()

DATA_PATH = "./class7_full.jsonl"
COLLECTION_NAME = "books"  # main.py의 collection_name과 같아야 함
PERSIST_DIR = "./chroma_store"


def flatten_metadata(meta: dict) -> dict:
    """Chroma 메타데이터는 str/int/float/bool만 받으므로 리스트는 문자열로 바꾸고 None은 뺀다."""
    flat = {}
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, list):
            flat[key] = ", ".join(str(v) for v in value)
        elif isinstance(value, (str, int, float, bool)):
            flat[key] = value
        else:
            flat[key] = str(value)
    return flat


def load_jsonl(path: str) -> tuple[list[Document], list[str]]:
    """jsonl 한 줄 = 문서 하나. 사람 한 명이 한 청크가 되도록 쪼개지 않는다."""
    docs = []
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            meta = flatten_metadata(row.get("metadata", {}))
            meta["doc_id"] = row["id"]
            meta["type"] = row["type"]
            if row.get("name"):
                meta["name"] = row["name"]
            docs.append(Document(page_content=row["text"], metadata=meta))
            ids.append(row["id"])
    return docs, ids


if __name__ == "__main__":
    docs, ids = load_jsonl(DATA_PATH)

    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL"),
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("GMS_KEY"),
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    # 예전에 books.txt로 넣은 청크가 남지 않도록 컬렉션을 비우고 다시 넣는다.
    vector_store.reset_collection()
    vector_store.add_documents(docs, ids=ids)
    print(f"{len(docs)}개 문서를 '{COLLECTION_NAME}' 컬렉션에 저장했습니다.")

    print("=" * 50)
    test_queries = [
        "우리반 반장은?",
        "동탄에 사는 사람은?",
        "박현모 취미는?",
        "이재원 마니또 추측은?",
    ]
    for query in test_queries:
        results = vector_store.similarity_search(query, k=3)
        print(f"[질문] {query}")
        for doc in results:
            print(f"  - {doc.metadata.get('name', doc.metadata.get('doc_id'))}: {doc.page_content[:60]}...")
        print()