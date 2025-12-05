"""Reranker retriever module for improved document ranking."""

from typing import List
from FlagEmbedding import FlagReranker
from langchain_core.runnables import RunnableLambda


class RerankerRetriever:
    """Retriever that uses a cross-encoder reranker to improve document ranking.
    
    This class combines a base retriever with a reranker model to retrieve
    and rank documents by relevance. It retrieves more candidates initially
    and then reranks them to return the most relevant documents.
    """
    
    def __init__(self,
                 vector_store,
                 reranker_model: str = 'BAAI/bge-reranker-v2-m3',
                 initial_k: int = 20,
                 top_k: int = 5,
                 use_fp16: bool = True):
        """Initialize the reranker retriever.
        
        Args:
            base_retriever: Base retriever to fetch initial candidates.
            reranker_model: Model identifier for the FlagReranker.
            initial_k: Number of initial candidates to retrieve.
            top_k: Number of top documents to return after reranking.
            use_fp16: Whether to use FP16 for faster computation.
        """
        self.base_retriever = vector_store
        self.reranker_model = reranker_model
        self.initial_k = initial_k
        self.top_k = top_k
        self.use_fp16 = use_fp16
        
        # Initialize the reranker
        self.reranker = FlagReranker(
            reranker_model,
            use_fp16=use_fp16
        )
        
        # Configure base retriever with initial k
        self.retriever = vector_store.as_retriever(
            search_kwargs={"k": initial_k}
        )
    
    def _extract_text_from_documents(self, documents: List) -> List[str]:
        """Extract text content from documents.
        
        Handles both LangChain Document objects and plain strings.
        
        Args:
            documents: List of documents or strings.
            
        Returns:
            List of text strings extracted from documents.
        """
        texts = []
        for doc in documents:
            if hasattr(doc, 'page_content'):
                texts.append(doc.page_content)
            else:
                texts.append(str(doc))
        return texts
    
    def retrieve_and_rerank(self, query) -> List:
        """Retrieve documents using base retriever and rerank them.
        
        This method:
        1. Retrieves initial candidates using the base retriever
        2. Extracts text from document objects
        3. Creates query-document pairs
        4. Computes reranking scores using the cross-encoder
        5. Returns top-k most relevant documents
        
        Args:
            query: The input query string or dict with 'user_question' key.
            
        Returns:
            List of top-k reranked documents, sorted by relevance (highest first).
        """
        # Handle dict input with 'user_question' key
        if isinstance(query, dict):
            query_text = query.get("user_question", "")
        else:
            query_text = query
        
        # Step 1: Retrieve candidate documents
        candidates = self.retriever.invoke(query_text)
        if not candidates:
            print("[INFO] No candidates retrieved.")
            return []
        print(f"[INFO] Retrieved {len(candidates)} candidate documents for reranking.")
        
        # Step 2: Extract text from documents
        candidate_texts = self._extract_text_from_documents(candidates)
        
        # Step 3: Create query-document pairs
        pairs = [[query_text, text] for text in candidate_texts]
        
        # Step 4: Compute reranking scores
        scores = self.reranker.compute_score(pairs)
        if not isinstance(scores, list):
            scores = [scores]  # Handle single-document edge case
        
        # Step 5: Sort by score (descending) and select top-k
        scored_docs = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True
        )
        top_k_docs = [doc for _, doc in scored_docs[:self.top_k]]
        print(f"[INFO] Reranked and selected top {len(top_k_docs)} documents.")
        
        return top_k_docs
    
    def as_runnable(self):
        """Convert retriever to a LangChain Runnable.
        
        Returns:
            RunnableLambda that wraps the retrieve_and_rerank method.
        """
        return RunnableLambda(self.retrieve_and_rerank)


def create_reranked_retriever(vector_store,
                             reranker_model: str = 'BAAI/bge-reranker-v2-m3',
                             initial_k: int = 20,
                             top_k: int = 5,
                             use_fp16: bool = True):
    """Convenience function to create a reranked retriever as a Runnable.
    
    Args:
        base_retriever: Base retriever to fetch initial candidates.
        reranker_model: Model identifier for the FlagReranker.
        initial_k: Number of initial candidates to retrieve.
        top_k: Number of top documents to return after reranking.
        use_fp16: Whether to use FP16 for faster computation.
        
    Returns:
        RunnableLambda that performs reranked retrieval.
    """
    retriever = RerankerRetriever(
        vector_store=vector_store,
        reranker_model=reranker_model,
        initial_k=initial_k,
        top_k=top_k,
        use_fp16=use_fp16
    )
    return retriever.as_runnable()
