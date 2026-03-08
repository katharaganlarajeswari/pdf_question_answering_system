"""
Vector Store Module
Handles embeddings creation and similarity search using ChromaDB
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import logging
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """Handles vector embeddings and similarity search"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", persist_directory: str = "./chroma_db"):
        self.model_name = model_name
        self.persist_directory = persist_directory
        self.embedding_model = None
        self.client = None
        self.collection = None
        
    def initialize(self):
        """Initialize the embedding model and ChromaDB client"""
        logger.info(f"Initializing embedding model: {self.model_name}")
        self.embedding_model = SentenceTransformer(self.model_name)
        
        logger.info(f"Initializing ChromaDB at: {self.persist_directory}")
        self.client = chromadb.Client(Settings(
            persist_directory=self.persist_directory,
            anonymized_telemetry=False
        ))
        
    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a text chunk"""
        if not self.embedding_model:
            self.initialize()
        return self.embedding_model.encode(text).tolist()
    
    def create_collection(self, collection_name: str):
        """Create or get a collection for storing embeddings"""
        if not self.client:
            self.initialize()
        
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "PDF Document Q&A Collection"}
            )
            logger.info(f"Collection '{collection_name}' ready with {self.collection.count()} existing documents")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def add_chunks(self, chunks: List[Dict[str, Any]], document_id: str):
        """Add text chunks to the vector store"""
        if not self.collection:
            raise ValueError("Collection not initialized. Call create_collection first.")
        
        ids = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            chunk_id = f"{document_id}_chunk_{chunk['id']}"
            ids.append(chunk_id)
            documents.append(chunk['text'])
            metadatas.append({
                "document_id": document_id,
                "chunk_id": chunk['id'],
                "char_count": chunk.get('char_count', 0)
            })
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = self.embedding_model.encode(documents).tolist()
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(chunks)} chunks to collection")
    
    def similarity_search(self, query: str, n_results: int = 5, document_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Perform similarity search for the query"""
        if not self.collection:
            raise ValueError("Collection not initialized")
        
        # Generate query embedding
        query_embedding = self._get_embedding(query)
        
        # Build where filter if document_id is specified
        where = {"document_id": document_id} if document_id else None
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        if results and results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                formatted_results.append({
                    "text": doc,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "distance": results['distances'][0][i] if results['distances'] else 0
                })
        
        logger.info(f"Found {len(formatted_results)} relevant chunks")
        return formatted_results
    
    def delete_document(self, document_id: str):
        """Delete a document and its chunks from the store"""
        if not self.collection:
            return
        
        # Get all chunk IDs for this document
        results = self.collection.get(where={"document_id": document_id})
        
        if results and results['ids']:
            self.collection.delete(ids=results['ids'])
            logger.info(f"Deleted document {document_id} with {len(results['ids'])} chunks")
    
    def clear_collection(self):
        """Clear all data from the collection"""
        if self.collection:
            self.client.delete_collection(name=self.collection.name)
            logger.info("Cleared collection")

