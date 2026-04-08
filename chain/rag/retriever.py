from chain.core.init_llmgw import get_rag_retriever


def run() -> None:
    # Avoid langchain_openai.OpenAIEmbeddings: it pulls chat_models → transformers → torch (c10.dll on your machine).
    retriever = get_rag_retriever(["harrison worked at kensho", "bears like to eat honey"])
    print(retriever.invoke("where did harrison work?"))
