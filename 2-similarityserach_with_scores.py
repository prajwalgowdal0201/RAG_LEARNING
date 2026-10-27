"""Load markdown docs, chunk them, embed with OpenAI, and run similarity search with scores.

Requires OPENAI_API_KEY in .env.
"""
from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader,TextLoader
from pathlib import Path 
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
load_dotenv()

def similarity_search_with_scores():
    """Load docs with DirectoryLoader and return the top 4 chunks as (Document, distance) pairs."""
    docs_path = Path(__file__).parent/"docs"
    documents = DirectoryLoader(path=docs_path,glob="**/*.md",loader_cls=TextLoader,loader_kwargs={"encoding":"utf-8"})
    documents=documents.load()
    # Split so each vector covers a small, focused piece of text.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    # No persist_directory: the index lives in memory only.
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model
    )
    query = "which error code represents that connection already exists"
    # Lower distance means a closer match.
    results_with_scores = vector_store.similarity_search_with_score(query=query,k=4)
    return results_with_scores

def search_with_manual_loading():
    """Same search, but loading each file with a TextLoader loop instead of DirectoryLoader."""
    docs_path = Path(__file__).parent/"docs"

    documents = []
    for md_file in docs_path.glob("**/*.md"):
        # TextLoader.load() returns a list, so extend rather than append.
        documents.extend(TextLoader(md_file, autodetect_encoding=True).load())

    # Chunk so the LLM gets small relevant pieces instead of whole files.
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small")
    )
    query="What error code is returned when rfid reader radio is busy?"
    results_with_scores = vector_store.similarity_search_with_score(query=query,k=5)
    for i,(doc,score) in enumerate(results_with_scores):
        print(f"Result {i+1}:")
        print(f"Score: {score}")
        print(f"Content: {doc.page_content}")
    return results_with_scores

if __name__ == "__main__":
    search_with_manual_loading()

