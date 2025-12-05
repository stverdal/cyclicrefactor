"""
PDF Indexer for RAG functionality.

Loads PDFs from a data directory, chunks them, and stores embeddings
in a ChromaDB vector store for retrieval by agents.

Usage:
    python -m rag.pdf_indexer --data-dir data/pdf --persist-dir cache/chroma_db
"""

import os
import argparse
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Try langchain_chroma first (newer), fall back to langchain_community
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from utils.logging import get_logger

logger = get_logger("pdf_indexer")


def load_pdfs(data_dir: str) -> List[Document]:
    """
    Load all PDF files from the specified directory.
    
    Args:
        data_dir: Path to directory containing PDF files
        
    Returns:
        List of Document objects with page content and metadata
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.warning(f"Data directory does not exist: {data_dir}")
        return []
    
    documents = []
    pdf_files = list(data_path.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {data_dir}")
        return []
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_path in pdf_files:
        try:
            logger.debug(f"Loading: {pdf_path.name}")
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            
            # Add source filename to metadata
            for doc in docs:
                doc.metadata["source_file"] = pdf_path.name
            
            documents.extend(docs)
            logger.info(f"Loaded {len(docs)} pages from {pdf_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to load {pdf_path.name}: {e}")
            continue
    
    logger.info(f"Total documents loaded: {len(documents)}")
    return documents


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Split documents into smaller chunks for embedding.
    
    Args:
        documents: List of Document objects to chunk
        chunk_size: Target size of each chunk in characters
        chunk_overlap: Overlap between chunks to preserve context
        
    Returns:
        List of chunked Document objects
    """
    if not documents:
        return []
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
    
    return chunks


def get_embedding_function(provider: str = "ollama", model: str = "nomic-embed-text"):
    """
    Get an embedding function based on the specified provider.
    
    Args:
        provider: Either "ollama" or "huggingface"
        model: Model name for the embedding provider
        
    Returns:
        Embedding function compatible with Chroma
    """
    if provider == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
            logger.info(f"Using Ollama embeddings with model: {model}")
            return OllamaEmbeddings(model=model)
        except ImportError:
            logger.warning("langchain_ollama not available, falling back to HuggingFace")
            provider = "huggingface"
    
    if provider == "huggingface":
        try:
            # Try newer langchain-huggingface package first
            from langchain_huggingface import HuggingFaceEmbeddings
            # Use a small, fast model for local embeddings
            hf_model = model if model != "nomic-embed-text" else "all-MiniLM-L6-v2"
            logger.info(f"Using HuggingFace embeddings with model: {hf_model}")
            return HuggingFaceEmbeddings(model_name=hf_model)
        except ImportError:
            try:
                # Fall back to langchain_community (deprecated)
                from langchain_community.embeddings import HuggingFaceEmbeddings
                hf_model = model if model != "nomic-embed-text" else "all-MiniLM-L6-v2"
                logger.info(f"Using HuggingFace embeddings (legacy) with model: {hf_model}")
                return HuggingFaceEmbeddings(model_name=hf_model)
            except ImportError:
                logger.error("No embedding providers available. Install langchain-ollama or langchain-huggingface")
                raise RuntimeError("No embedding provider available")
    
    raise ValueError(f"Unknown embedding provider: {provider}")


def create_vector_store(
    chunks: List[Document],
    persist_dir: str,
    embedding_provider: str = "ollama",
    embedding_model: str = "nomic-embed-text",
    collection_name: str = "architecture_docs"
) -> Optional[Chroma]:
    """
    Create a ChromaDB vector store from document chunks.
    
    Args:
        chunks: List of Document chunks to embed
        persist_dir: Directory to persist the vector store
        embedding_provider: Provider for embeddings ("ollama" or "huggingface")
        embedding_model: Model name for embeddings
        collection_name: Name for the Chroma collection
        
    Returns:
        Chroma vector store instance, or None if creation failed
    """
    if not chunks:
        logger.warning("No chunks to index")
        return None
    
    # Ensure persist directory exists
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)
    
    try:
        embedding_fn = get_embedding_function(embedding_provider, embedding_model)
        
        logger.info(f"Creating vector store with {len(chunks)} chunks...")
        logger.info(f"Persist directory: {persist_dir}")
        logger.info(f"Collection: {collection_name}")
        
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_fn,
            persist_directory=str(persist_path),
            collection_name=collection_name
        )
        
        logger.info("Vector store created successfully")
        return vectorstore
        
    except Exception as e:
        logger.error(f"Failed to create vector store: {e}")
        raise


def index_pdfs(
    data_dir: str = "data/pdf",
    persist_dir: str = "cache/chroma_db",
    embedding_provider: str = "ollama",
    embedding_model: str = "nomic-embed-text",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    collection_name: str = "architecture_docs"
) -> Optional[Chroma]:
    """
    Main indexing function - loads PDFs, chunks them, and creates vector store.
    
    Args:
        data_dir: Directory containing PDF files
        persist_dir: Directory to persist the vector store
        embedding_provider: Provider for embeddings
        embedding_model: Model name for embeddings
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        collection_name: Name for the Chroma collection
        
    Returns:
        Chroma vector store instance, or None if indexing failed
    """
    logger.info("="*50)
    logger.info("Starting PDF indexing")
    logger.info("="*50)
    
    # Step 1: Load PDFs
    documents = load_pdfs(data_dir)
    if not documents:
        logger.error("No documents loaded, aborting indexing")
        return None
    
    # Step 2: Chunk documents
    chunks = chunk_documents(documents, chunk_size, chunk_overlap)
    if not chunks:
        logger.error("No chunks created, aborting indexing")
        return None
    
    # Step 3: Create vector store
    vectorstore = create_vector_store(
        chunks,
        persist_dir,
        embedding_provider,
        embedding_model,
        collection_name
    )
    
    if vectorstore:
        logger.info("="*50)
        logger.info("PDF indexing complete!")
        logger.info(f"Indexed {len(documents)} pages as {len(chunks)} chunks")
        logger.info(f"Vector store persisted to: {persist_dir}")
        logger.info("="*50)
    
    return vectorstore


def main():
    """CLI entry point for PDF indexing."""
    parser = argparse.ArgumentParser(
        description="Index PDF documents for RAG retrieval"
    )
    parser.add_argument(
        "--data-dir",
        default="data/pdf",
        help="Directory containing PDF files (default: data/pdf)"
    )
    parser.add_argument(
        "--persist-dir",
        default="cache/chroma_db",
        help="Directory to persist vector store (default: cache/chroma_db)"
    )
    parser.add_argument(
        "--embedding-provider",
        choices=["ollama", "huggingface"],
        default="ollama",
        help="Embedding provider to use (default: ollama)"
    )
    parser.add_argument(
        "--embedding-model",
        default="nomic-embed-text",
        help="Embedding model name (default: nomic-embed-text)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Size of text chunks (default: 1000)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between chunks (default: 200)"
    )
    parser.add_argument(
        "--collection",
        default="architecture_docs",
        help="Chroma collection name (default: architecture_docs)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    else:
        import logging
        logging.basicConfig(level=logging.INFO)
    
    # Run indexing
    vectorstore = index_pdfs(
        data_dir=args.data_dir,
        persist_dir=args.persist_dir,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        collection_name=args.collection
    )
    
    if vectorstore:
        # Quick verification
        print("\nVerification - Testing retrieval:")
        results = vectorstore.similarity_search("dependency inversion principle", k=2)
        for i, doc in enumerate(results):
            print(f"\n--- Result {i+1} ---")
            print(f"Source: {doc.metadata.get('source_file', 'unknown')}")
            print(f"Content preview: {doc.page_content[:200]}...")
    else:
        print("Indexing failed. Check logs for details.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
