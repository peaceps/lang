from langchain_core.vectorstores import InMemoryVectorStore
from chain.core.init_llmgw import get_embedding


def run() -> None:
    # Avoid langchain_openai.OpenAIEmbeddings: it pulls chat_models → transformers → torch (c10.dll on your machine).
    embedding = get_embedding()
    vectorstore = InMemoryVectorStore.from_texts(
        ["harrison worked at kensho", "bears like to eat honey"],
        embedding=embedding,
    )
    retriever = vectorstore.as_retriever()
    print(retriever.invoke("where did harrison work?"))
