"""Vector store module using ChromaDB for semantic search."""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm


class VectorStore:
    """Manages embeddings and vector database for semantic search."""

    def __init__(self, db_path: Path, collection_name: str = "epstein_files"):
        """
        Initialize the vector store.

        Args:
            db_path: Path to ChromaDB storage directory
            collection_name: Name of the ChromaDB collection
        """
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.embedding_model = None
        self.client = None
        self.collection = None

    def initialize(self):
        """Initialize embedding model and ChromaDB client."""
        print("Loading embedding model (all-MiniLM-L6-v2)...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        print(f"Initializing ChromaDB at {self.db_path}...")
        self.db_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Epstein Files document chunks"}
        )

    def create_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed
            batch_size: Batch size for encoding

        Returns:
            List of embedding vectors
        """
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings.tolist()

    def build_index(self, chunks: List[Dict[str, Any]], batch_size: int = 100):
        """
        Build the vector index from document chunks.

        Args:
            chunks: List of chunk dictionaries with 'text' and 'metadata'
            batch_size: Number of chunks to process at once
        """
        if self.collection.count() > 0:
            print(f"Collection already contains {self.collection.count()} documents")
            response = input("Rebuild index? (y/n): ").strip().lower()
            if response != 'y':
                print("Using existing index")
                return

            print("Clearing existing collection...")
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Epstein Files document chunks"}
            )

        print(f"Building index for {len(chunks)} chunks...")

        # Process in batches
        for i in tqdm(range(0, len(chunks), batch_size), desc="Building index"):
            batch = chunks[i:i + batch_size]

            texts = [chunk["text"] for chunk in batch]
            metadatas = [chunk["metadata"] for chunk in batch]
            ids = [f"chunk_{i + j}" for j in range(len(batch))]

            # Generate embeddings
            embeddings = self.create_embeddings(texts, batch_size=32)

            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )

        print(f"Index built successfully with {self.collection.count()} chunks")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks using semantic similarity.

        Args:
            query: Search query text
            top_k: Number of results to return

        Returns:
            List of relevant chunks with metadata and similarity scores
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0].tolist()

        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        # Format results
        chunks = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                chunks.append({
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if 'distances' in results else None
                })

        return chunks

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count() if self.collection else 0
        return {
            "total_chunks": count,
            "collection_name": self.collection_name,
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dimension": 384,
            "storage_format": "ChromaDB"
        }
