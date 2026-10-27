"""Complete RAG pipeline: load markdown docs -> chunk -> embed -> Chroma -> retriever -> LLM chain.

Requires OPENAI_API_KEY in .env. Type exit, quit or bye to stop the chat loop.
"""
from dotenv import load_dotenv
from pathlib import Path 
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownTextSplitter
from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
load_dotenv()

def create_kb():
    """Build the vector store from the markdown files in docs/ and return it."""
    docs_path=Path(__file__).parent/"docs"
    documents: list[Document] = []
    for md_file in docs_path.glob("**/*.md"):
        # TextLoader.load() returns a list of documents, so use extend, not append.
        documents.extend(TextLoader(md_file.absolute()).load())
    
    # Markdown-aware splitter cuts on headings and paragraphs before falling back to size.
    md_splitter = MarkdownTextSplitter(chunk_size=1000,chunk_overlap=100)
    chunks = md_splitter.split_documents(documents)
    # Saved to disk, but every run re-embeds all chunks and calls the OpenAI API again.
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
        persist_directory="vector_store"
    )
    return vector_store

def demo_basic_rag():
    """Run an interactive question-answer loop over the knowledge base."""
    vector_store = create_kb()

    # Retriever: embeds the question and returns the 5 most similar chunks.
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k":5}
    )

    llm = init_chat_model(model="gpt-5.4-nano")

    prompt = ChatPromptTemplate.from_template(
        """Answer the question based on only the context provided.
        If you dont get info from context , tell "I cant draw  any conclusion from the context."
        context: {context}
        question:{question}
        Answer:

        """
    )

    def format_retrieved_docs(docs):
        """Join the retrieved Documents into one string for the {context} slot."""
        return "\n\n".join([doc.page_content for doc in docs])

    # Each `|` passes the left output to the right step (all are Runnables).
    # Flow: question -> {context: retriever+format, question: passthrough} -> prompt -> llm -> str.
    rag_chain = ({"context":retriever | format_retrieved_docs,"question":RunnablePassthrough()}
                 | prompt 
                 | llm 
                 | StrOutputParser()
    )
    while True:
        query = input("Enter you question to the bot\n")
        if query.strip().lower() in ['exit','quit','bye']:
            break
        response = rag_chain.invoke(query)
        print("AI response: ",response)     

if __name__ == "__main__":
    demo_basic_rag()

