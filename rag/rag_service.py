"""
RAG Service for querying the indexed documents.

Provides a simple interface for agents to retrieve relevant context
from the indexed PDF documents.

Usage:
    from rag.rag_service import RAGService
    
    service = RAGService(config.retriever)
    context = service.query("dependency inversion principle")
    formatted = service.format_context(context)
"""

from typing import List, Optional, Dict, Any
from pathlib import Path

from langchain_core.documents import Document

# Try langchain_chroma first (newer), fall back to langchain_community
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from config import RetrieverConfig
from rag.pdf_indexer import get_embedding_function
from utils.logging import get_logger

logger = get_logger("rag_service")


class RAGService:
    """
    Service for retrieving relevant documents from the vector store.
    
    Provides a query interface for agents to get context from indexed PDFs.
    """
    
    def __init__(
        self,
        config: Optional[RetrieverConfig] = None,
        persist_dir: Optional[str] = None,
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
        collection_name: Optional[str] = None
    ):
        """
        Initialize the RAG service.
        
        Args:
            config: RetrieverConfig object with all settings
            persist_dir: Override persist directory (if no config)
            embedding_provider: Override embedding provider (if no config)
            embedding_model: Override embedding model (if no config)
            collection_name: Override collection name (if no config)
        """
        # Use config or individual parameters
        if config:
            self.persist_dir = config.persist_dir
            self.embedding_provider = config.embedding_provider
            self.embedding_model = config.embedding_model
            self.collection_name = config.collection_name
            self.search_type = config.search_type
            self.search_kwargs = config.search_kwargs
        else:
            self.persist_dir = persist_dir or "cache/chroma_db"
            self.embedding_provider = embedding_provider or "huggingface"
            self.embedding_model = embedding_model or "all-MiniLM-L6-v2"
            self.collection_name = collection_name or "architecture_docs"
            self.search_type = "similarity"
            self.search_kwargs = {"k": 4}
        
        self._vectorstore: Optional[Chroma] = None
        self._embedding_fn = None
        
    def _get_embedding_function(self):
        """Get the embedding function based on provider (delegates to pdf_indexer)."""
        if self._embedding_fn is not None:
            return self._embedding_fn
        
        self._embedding_fn = get_embedding_function(self.embedding_provider, self.embedding_model)
        return self._embedding_fn
    
    def _get_vectorstore(self) -> Optional[Chroma]:
        """Get or create the vector store connection."""
        if self._vectorstore is not None:
            return self._vectorstore
        
        persist_path = Path(self.persist_dir)
        if not persist_path.exists():
            logger.warning(f"Vector store not found at {self.persist_dir}. Run pdf_indexer first.")
            return None
        
        try:
            embedding_fn = self._get_embedding_function()
            self._vectorstore = Chroma(
                persist_directory=str(persist_path),
                embedding_function=embedding_fn,
                collection_name=self.collection_name
            )
            logger.info(f"Connected to vector store at {self.persist_dir}")
            return self._vectorstore
        except Exception as e:
            logger.error(f"Failed to connect to vector store: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if the RAG service is available (vector store exists)."""
        vectorstore = self._get_vectorstore()
        return vectorstore is not None
    
    def query(
        self,
        query_text: str,
        k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Query the vector store for relevant documents.
        
        Args:
            query_text: The query to search for
            k: Number of results to return (default from config)
            filter_dict: Optional metadata filter
            
        Returns:
            List of Document objects with relevant content
        """
        vectorstore = self._get_vectorstore()
        if vectorstore is None:
            logger.warning("Vector store not available, returning empty results")
            return []
        
        # Use provided k or default from config
        num_results = k or self.search_kwargs.get("k", 4)
        
        try:
            if self.search_type == "mmr":
                # Maximum Marginal Relevance for diversity
                results = vectorstore.max_marginal_relevance_search(
                    query_text,
                    k=num_results,
                    filter=filter_dict
                )
            else:
                # Standard similarity search
                results = vectorstore.similarity_search(
                    query_text,
                    k=num_results,
                    filter=filter_dict
                )
            
            logger.debug(f"Query '{query_text[:50]}...' returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []
    
    def query_with_scores(
        self,
        query_text: str,
        k: Optional[int] = None
    ) -> List[tuple]:
        """
        Query with similarity scores.
        
        Args:
            query_text: The query to search for
            k: Number of results to return
            
        Returns:
            List of (Document, score) tuples
        """
        vectorstore = self._get_vectorstore()
        if vectorstore is None:
            logger.warning("Vector store not available for scored query")
            return []
        
        num_results = k or self.search_kwargs.get("k", 4)
        
        try:
            results = vectorstore.similarity_search_with_score(
                query_text,
                k=num_results
            )
            
            if results:
                # Log score range for debugging
                scores = [score for _, score in results]
                logger.debug(f"Query returned {len(results)} results, score range: {min(scores):.3f} - {max(scores):.3f}")
            else:
                logger.debug("Query returned 0 results")
            
            return results
        except Exception as e:
            logger.error(f"Query with scores failed: {e}")
            return []
    
    def format_context(
        self,
        documents: List[Document],
        include_source: bool = True,
        max_length: Optional[int] = None
    ) -> str:
        """
        Format retrieved documents into a context string for prompts.
        
        Args:
            documents: List of Document objects from query
            include_source: Whether to include source file names
            max_length: Optional max length to truncate to
            
        Returns:
            Formatted context string
        """
        if not documents:
            return ""
        
        parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source_file", "unknown")
            content = doc.page_content.strip()
            
            if include_source:
                parts.append(f"[Source {i}: {source}]\n{content}")
            else:
                parts.append(content)
        
        result = "\n\n---\n\n".join(parts)
        
        if max_length and len(result) > max_length:
            result = result[:max_length] + "\n\n[... truncated]"
        
        return result
    
    def get_relevant_context(
        self,
        query_text: str,
        k: Optional[int] = None,
        max_length: Optional[int] = None
    ) -> str:
        """
        Convenience method to query and format in one call.
        
        Args:
            query_text: The query to search for
            k: Number of results
            max_length: Max context length
            
        Returns:
            Formatted context string ready for prompts
        """
        documents = self.query(query_text, k=k)
        return self.format_context(documents, max_length=max_length)


# Singleton instance for convenience
_default_service: Optional[RAGService] = None


def get_rag_service(config: Optional[RetrieverConfig] = None) -> RAGService:
    """
    Get the RAG service instance.
    
    Uses a singleton pattern for efficiency. If called with a different config
    after initialization, logs a warning but returns the existing instance.
    
    Args:
        config: Config to use. If None and no instance exists, uses defaults.
        
    Returns:
        RAGService instance
    """
    global _default_service
    
    if _default_service is None:
        _default_service = RAGService(config)
        logger.info("RAG service singleton created")
    elif config is not None:
        # Warn if trying to use different config
        logger.warning("RAG service already initialized - ignoring new config. "
                      "Call reset_service() first to use a different config.")
    
    return _default_service


def reset_service():
    """Reset the singleton service (useful for testing)."""
    global _default_service
    _default_service = None
