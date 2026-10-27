"""Common LangChain document loaders run on dummy files in sample_data/.

Every loader's .load() returns list[Document]; each Document has page_content (str) and metadata (dict).
"""
from pathlib import Path
from pprint import pprint

import truststore

truststore.inject_into_ssl()

from langchain_community.document_loaders import (
    CSVLoader,
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)

SAMPLE_DIR = Path(__file__).parent / "sample_data"


def show(title, docs):
    """Print the loader output type, the document count, and the first document."""
    print(f"\n=== {title} ===")
    print(f"type: {type(docs).__name__} of {type(docs[0]).__name__}, count: {len(docs)}")
    print("page_content:", repr(docs[0].page_content[:80]))
    print("metadata:")
    pprint(docs[0].metadata)


def demo_text_loader():
    """One Document per file."""
    show("TextLoader", TextLoader(SAMPLE_DIR / "notes.txt", encoding="utf-8").load())


def demo_pdf_loader():
    """One Document per page; metadata includes the page number (0-based)."""
    show("PyPDFLoader", PyPDFLoader(str(SAMPLE_DIR / "sample.pdf")).load())


def demo_csv_loader():
    """One Document per row; page_content is the row as 'column: value' lines."""
    show("CSVLoader", CSVLoader(str(SAMPLE_DIR / "employees.csv")).load())


def demo_directory_loader():
    """Applies an inner loader to every file matching the glob."""
    loader = DirectoryLoader(
        str(SAMPLE_DIR), glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
    )
    show("DirectoryLoader (*.md)", loader.load())


def demo_lazy_load():
    """lazy_load() yields Documents one at a time instead of building the whole list."""
    print("\n=== lazy_load ===")
    for doc in CSVLoader(str(SAMPLE_DIR / "employees.csv")).lazy_load():
        print(doc.metadata["row"], doc.page_content.replace("\n", " | "))


def demo_web_loader():
    """One Document per URL; needs network access and beautifulsoup4."""
    show("WebBaseLoader", WebBaseLoader("https://example.com").load())


if __name__ == "__main__":
    demo_text_loader()
    demo_pdf_loader()
    demo_csv_loader()
    demo_directory_loader()
    demo_lazy_load()
    demo_web_loader()
