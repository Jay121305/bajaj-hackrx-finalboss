import os
import numpy as np
import faiss
import time
import logging
import torch
from typing import Dict, List, Any, Tuple, Optional, Set
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Vector embedding and semantic search engine"""

    def __init__(self):
        self.model_name = "all-MiniLM-L6-v2"  # Lightweight, efficient embedding model
        self.model = None
        self.index_dimensions = 384  # Dimensions for the model
        
        # Store indexes by domain for better performance
        self.domain_indexes = {}
        
        # Document metadata and mapping
        self.document_to_chunks = {}  # document_id -> List[chunk_ids]
        self.chunk_metadata = {}      # chunk_id -> metadata
        self.document_metadata = {}   # document_id -> metadata
        
        # Track document domains
        self.document_to_domain = {}  # document_id -> domain
        
        # Performance stats
        self.stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "embedding_time": 0,
            "search_count": 0,
            "avg_search_time": 0,
        }

    async def initialize(self):
        """Initialize embedding model and indexes"""
        logger.info(f"Initializing embedding model: {self.model_name}")
        
        # Load the model (will download if not present)
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Embedding model loaded successfully")
            
            # Create base FAISS index for each supported domain
            base_domains = ["insurance", "legal", "hr", "compliance", "finance", "medical", "general"]
            for domain in base_domains:
                self.domain_indexes[domain] = faiss.IndexFlatL2(self.index_dimensions)
                
            logger.info(f"FAISS indexes initialized for {len(self.domain_indexes)} domains")
                
        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            raise RuntimeError(f"Failed to initialize embedding engine: {str(e)}")

    async def index_document(self, document_id: str, content: str, domain: str) -> int:
        """Index document content into vector database"""
        start_time = time.time()
        
        try:
            # Store domain information
            self.document_to_domain[document_id] = domain
            
            # Create domain index if it doesn't exist
            if domain not in self.domain_indexes:
                self.domain_indexes[domain] = faiss.IndexFlatL2(self.index_dimensions)
                
            # Split document into chunks for better retrieval
            chunks = self._split_into_chunks(content)
            
            # Generate embeddings for chunks
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = self.model.encode(chunk_texts, convert_to_tensor=True)
            
            # Convert to numpy for FAISS
            embeddings_np = embeddings.cpu().numpy().astype('float32')
            
            # Add to FAISS index
            self.domain_indexes[domain].add(embeddings_np)
            
            # Track the last index for this document
            current_index = self.stats["total_chunks"]
            chunk_ids = []
            
            # Store chunk metadata
            for i, chunk in enumerate(chunks):
                chunk_id = f"{document_id}_{i}"
                chunk_ids.append(chunk_id)
                
                # Store metadata with vector index
                self.chunk_metadata[chunk_id] = {
                    "document_id": document_id,
                    "chunk_index": i,
                    "vector_index": current_index + i,
                    "text": chunk["text"],
                    "metadata": chunk.get("metadata", {})
                }
            
            # Map document to its chunks
            self.document_to_chunks[document_id] = chunk_ids
            
            # Update stats
            self.stats["total_documents"] += 1
            self.stats["total_chunks"] += len(chunks)
            self.stats["embedding_time"] += time.time() - start_time
            
            # Estimate token count
            token_count = len(content.split()) * 1.3  # Rough estimate
            
            logger.info(f"Indexed document {document_id} with {len(chunks)} chunks in {time.time() - start_time:.2f}s")
            
            return int(token_count)
            
        except Exception as e:
            logger.error(f"Error indexing document: {str(e)}")
            raise RuntimeError(f"Failed to index document: {str(e)}")

    async def semantic_search(
        self,
        query: str,
        document_id: Optional[str] = None,
        domain: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Perform semantic search on indexed documents"""
        start_time = time.time()
        
        try:
            # Generate query embedding
            query_embedding = self.model.encode(query, convert_to_tensor=True)
            query_embedding_np = query_embedding.cpu().numpy().astype('float32').reshape(1, -1)
            
            # Determine search scope
            if document_id:
                # Search within specific document
                domain = self.document_to_domain.get(document_id, "general")
                results = await self._search_document(query_embedding_np, document_id, domain, top_k)
            elif domain:
                # Search within domain
                results = await self._search_domain(query_embedding_np, domain, top_k)
            else:
                # Search across all domains (more expensive)
                results = await self._search_all_domains(query_embedding_np, top_k)
            
            # Update stats
            search_time = time.time() - start_time
            self.stats["search_count"] += 1
            self.stats["avg_search_time"] = ((self.stats["avg_search_time"] * (self.stats["search_count"] - 1)) + search_time) / self.stats["search_count"]
            
            logger.info(f"Semantic search completed in {search_time:.2f}s with {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Error performing semantic search: {str(e)}")
            raise RuntimeError(f"Failed to perform search: {str(e)}")

    async def _search_document(self, query_vector: np.ndarray, document_id: str, domain: str, top_k: int) -> List[Dict[str, Any]]:
        """Search within a specific document"""
        if domain not in self.domain_indexes:
            return []
            
        if document_id not in self.document_to_chunks:
            return []
            
        # Get chunk IDs for this document
        chunk_ids = self.document_to_chunks[document_id]
        
        # Get vector indexes for these chunks
        vector_indexes = [self.chunk_metadata[chunk_id]["vector_index"] for chunk_id in chunk_ids]
        
        # Search in the domain index
        distances, indices = self.domain_indexes[domain].search(query_vector, len(vector_indexes))
        
        # Filter to only include vectors from this document
        results = []
        for i, idx in enumerate(indices[0]):
            if idx in vector_indexes:
                # Find the corresponding chunk_id
                for chunk_id in chunk_ids:
                    if self.chunk_metadata[chunk_id]["vector_index"] == idx:
                        results.append({
                            "chunk_id": chunk_id,
                            "document_id": document_id,
                            "text": self.chunk_metadata[chunk_id]["text"],
                            "metadata": self.chunk_metadata[chunk_id]["metadata"],
                            "score": float(1.0 / (1.0 + distances[0][i])),  # Convert distance to similarity score
                        })
                        break
                        
                if len(results) >= top_k:
                    break
                    
        return results

    async def _search_domain(self, query_vector: np.ndarray, domain: str, top_k: int) -> List[Dict[str, Any]]:
        """Search within a specific domain"""
        if domain not in self.domain_indexes:
            return []
            
        # Search in the domain index
        distances, indices = self.domain_indexes[domain].search(query_vector, top_k * 2)  # Get more results for filtering
        
        # Map indices back to chunk metadata
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= len(self.chunk_metadata):
                continue
                
            # Find the chunk with this vector index
            found = False
            for chunk_id, metadata in self.chunk_metadata.items():
                if metadata["vector_index"] == idx:
                    results.append({
                        "chunk_id": chunk_id,
                        "document_id": metadata["document_id"],
                        "text": metadata["text"],
                        "metadata": metadata.get("metadata", {}),
                        "score": float(1.0 / (1.0 + distances[0][i])),
                    })
                    found = True
                    break
                    
            if len(results) >= top_k:
                break
                
        return results[:top_k]

    async def _search_all_domains(self, query_vector: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Search across all domains"""
        all_results = []
        
        for domain, index in self.domain_indexes.items():
            if index.ntotal == 0:
                continue
                
            # Search in this domain
            distances, indices = index.search(query_vector, min(top_k, index.ntotal))
            
            # Map indices to chunk metadata for this domain
            domain_results = []
            for i, idx in enumerate(indices[0]):
                # Find the chunk with this vector index in this domain
                found = False
                for chunk_id, metadata in self.chunk_metadata.items():
                    if metadata["vector_index"] == idx:
                        document_id = metadata["document_id"]
                        
                        # Check if document belongs to this domain
                        if self.document_to_domain.get(document_id) == domain:
                            domain_results.append({
                                "chunk_id": chunk_id,
                                "document_id": document_id,
                                "text": metadata["text"],
                                "metadata": metadata.get("metadata", {}),
                                "score": float(1.0 / (1.0 + distances[0][i])),
                                "domain": domain,
                            })
                            found = True
                            break
            
            all_results.extend(domain_results)
        
        # Sort by score and return top-k
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]

    def _split_into_chunks(self, text: str, chunk_size: int = 512, overlap: int = 128) -> List[Dict]:
        """Split document into overlapping chunks for better retrieval"""
        words = text.split()
        chunks = []
        
        # Create sliding window of words
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            if not chunk_words:
                continue
                
            # Create chunk
            chunk_text = " ".join(chunk_words)
            
            # Add metadata about the chunk position
            chunk_metadata = {
                "start_idx": i,
                "end_idx": i + len(chunk_words),
                "position": len(chunks),
            }
            
            # Add structural information
            structure_info = self._detect_chunk_structure(chunk_text)
            if structure_info:
                chunk_metadata.update(structure_info)
            
            chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })
        
        return chunks

    def _detect_chunk_structure(self, text: str) -> Dict[str, Any]:
        """Detect structural elements in chunk for better context"""
        metadata = {}
        
        # Detect headings
        lines = text.split("\n")
        first_line = lines[0] if lines else ""
        
        # Check for section/heading patterns
        if "SECTION" in first_line or "CLAUSE" in first_line or "ARTICLE" in first_line:
            metadata["contains_section_header"] = True
            metadata["section_header"] = first_line.strip()
        
        # Check for list items
        list_items = [line for line in lines if line.strip().startswith(("•", "-", "* ", "1.", "2.", "a)", "b)"))]
        if list_items:
            metadata["contains_list"] = True
            metadata["list_items_count"] = len(list_items)
        
        # Check for definitions
        if " means " in text or "defined as" in text:
            metadata["contains_definition"] = True
        
        # Check for numeric values (important in contracts/policies)
        if any(c.isdigit() for c in text):
            metadata["contains_numbers"] = True
            
        # Check for table-like content
        if any("|" in line or "\t" in line for line in lines):
            metadata["contains_table"] = True
        
        return metadata

    async def get_document_domain(self, document_id: str) -> str:
        """Get domain for a document"""
        return self.document_to_domain.get(document_id, "general")

    async def get_status(self) -> Dict[str, Any]:
        """Get engine status and statistics"""
        return {
            "model": self.model_name,
            "documents_indexed": self.stats["total_documents"],
            "total_chunks": self.stats["total_chunks"],
            "searches_performed": self.stats["search_count"],
            "avg_search_time": round(self.stats["avg_search_time"], 3) if self.stats["search_count"] > 0 else 0,
            "domains_indexed": list(self.domain_indexes.keys()),
            "domain_doc_counts": {domain: len([d for d, v in self.document_to_domain.items() if v == domain]) for domain in self.domain_indexes.keys()}
        }