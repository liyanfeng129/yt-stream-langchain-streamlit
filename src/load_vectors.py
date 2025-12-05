"""Vector store management module for document retrieval."""

from langchain_community.vectorstores.chroma import Chroma
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class QueryOnlyEmbeddings(Embeddings):
    """Embedding function that only encodes queries (not documents).
    
    Used when loading vectorstore from persistence where documents
    already have embeddings stored.
    """
    
    def __init__(self, model):
        """Initialize with a SentenceTransformer model.
        
        Args:
            model: SentenceTransformer model instance for encoding queries.
        """
        super().__init__()
        self.model = model
    
    def embed_documents(self, texts):
        """Not used when loading from persistence."""
        raise NotImplementedError("Use only for queries")
    
    def embed_query(self, text):
        """Embed a query on the fly.
        
        Args:
            text: Query text to embed.
            
        Returns:
            List of float values representing the embedding.
        """
        # Ensure text is not None or empty, and encode as a list
        if not text or not isinstance(text, str):
            raise ValueError(f"Invalid query text: {text}")
        
        # SentenceTransformer expects a list of strings
        embedding = self.model.encode([text], convert_to_tensor=False)
        return embedding[0].tolist()


class VectorStoreLoader:
    """Load and manage Chroma vectorstore from persistence."""
    
    def __init__(self, 
                 embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
                 #collection_name: str = "atoss_collection",
                 #persist_directory: str = "./chroma_langchain_db",
                 collection_name: str = "bossard_whitepaper",
                 persist_directory: str = "./chroma/chroma_bossard_whitepaper_db",
                 device: str = "cuda",
                 k: int = 10):
        """Initialize the vector store loader.
        
        Args:
            embedding_model_name: Name of the SentenceTransformer model.
            collection_name: Name of the Chroma collection.
            persist_directory: Path to the persisted vectorstore directory.
            device: Device to use for embeddings (cuda or cpu).
            k: Number of documents to retrieve.
        """
        self.embedding_model_name = embedding_model_name
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.device = device
        self.k = k
        
        self.emb_model = None
        self.embedding_function = None
        self.vectorstore = None
        self.retriever = None
    
    def load(self):
        """Load the vectorstore and retriever from persistence.
        
        Raises:
            Exception: If vectorstore cannot be loaded.
        """
        print("[OK] Loading existing vectorstore from persistence...")
        
        try:
            # Load embedding model for query encoding only
            self.emb_model = SentenceTransformer(
                self.embedding_model_name, 
                device=self.device
            )
            
            # Create embedding function for queries only
            self.embedding_function = QueryOnlyEmbeddings(self.emb_model)
            
            # Load vectorstore from persisted directory
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_function,
                persist_directory=self.persist_directory,
            )
            
            print(f"[OK] Vectorstore loaded successfully")
            print(f"[OK] Collection contains {self.vectorstore._collection.count()} documents")
            
            # Create retriever
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.k}
            )
            print(f"[OK] Retriever initialized with k={self.k}")
            print(f"[OK] Ready to query without needing embeddings in memory")
            
        except Exception as e:
            print(f"[ERROR] Failed to load vectorstore: {e}")
            raise
    
    def get_retriever(self):
        """Get the initialized retriever.
        
        Returns:
            Retriever object for document retrieval.
            
        Raises:
            RuntimeError: If vectorstore has not been loaded yet.
        """
        if self.retriever is None:
            raise RuntimeError("Vectorstore not loaded. Call load() first.")
        return self.retriever
    
    def get_vectorstore(self):
        """Get the initialized vectorstore.
        
        Returns:
            Chroma vectorstore object.
            
        Raises:
            RuntimeError: If vectorstore has not been loaded yet.
        """
        if self.vectorstore is None:
            raise RuntimeError("Vectorstore not loaded. Call load() first.")
        return self.vectorstore


def load_vectorstore(embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
                     collection_name: str = "atoss_collection",
                     persist_directory: str = "./chroma_langchain_db",
                     device: str = "cuda",
                     k: int = 10):
    """Convenience function to load vectorstore in one call.
    
    Args:
        embedding_model_name: Name of the SentenceTransformer model.
        collection_name: Name of the Chroma collection.
        persist_directory: Path to the persisted vectorstore directory.
        device: Device to use for embeddings (cuda or cpu).
        k: Number of documents to retrieve.
        
    Returns:
        Tuple of (retriever, vectorstore) objects.
    """
    loader = VectorStoreLoader(
        embedding_model_name=embedding_model_name,
        collection_name=collection_name,
        persist_directory=persist_directory,
        device=device,
        k=k
    )
    loader.load()
    return loader.get_retriever(), loader.get_vectorstore()