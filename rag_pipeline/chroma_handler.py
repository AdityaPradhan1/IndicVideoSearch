#!/usr/bin/env python3
"""
ChromaDB Handler - Handles all ChromaDB operations for video summaries using langchain_chroma
"""

from typing import List, Dict, Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import hashlib
import traceback
from rank_bm25 import BM25Okapi
import numpy as np
from collections import defaultdict
class SentenceTransformerEmbeddings(Embeddings):
    """Wrapper for sentence_transformers to use with LangChain"""
    
    def __init__(self, sentence_transformer):
        """Initialize with a sentence_transformer model"""
        self.sentence_transformer = sentence_transformer
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using the sentence transformer"""
        print(f"Embedding {len(texts)} documents")
        return self.sentence_transformer.encode(texts).tolist()
        
    def embed_query(self, text: str) -> List[float]:
        """Embed a query using the sentence transformer"""
        print(f"Embedding query: {text}")
        return self.sentence_transformer.encode(text).tolist()

class ChromaDBHandler:
    def __init__(self, embeddings, persist_directory="chroma_db"):
        """Initialize ChromaDB client with LangChain integration
        
        Args:
            embeddings: SentenceTransformer or embeddings function to use
            persist_directory: Directory to persist ChromaDB data
        """
        self.persist_directory = persist_directory
        self.bm25_index = None
        self.bm25_documents = []
        self.bm25_doc_ids = []
        
        # Create embeddings wrapper if it's not already a LangChain Embeddings object
        if not isinstance(embeddings, Embeddings):
            print("Using SentenceTransformer for embeddings")
            self.embeddings = SentenceTransformerEmbeddings(embeddings)
        else:
            print("Using provided embeddings object")
            self.embeddings = embeddings
    
    def create_collection(self, collection_name: str, metadata: Optional[Dict] = None):
        """Create a new collection"""
        try:
            # LangChain Chroma will create a new collection or get existing one
            collection = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            # Delete all existing documents if the collection exists
            try:
                collection.delete_collection()
                collection = Chroma(
                    collection_name=collection_name, 
                    embedding_function=self.embeddings,
                    persist_directory=self.persist_directory
                )
            except:
                pass
                
            return collection
        except Exception as e:
            print(f"Error creating collection: {e}")
            return None
    
    def get_collection(self, collection_name: str):
        """Get existing collection"""
        try:
            return Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
        except Exception as e:
            print(f"Collection '{collection_name}' not found: {e}")
            return None
    
    def get_or_create_collection(self, collection_name: str, metadata: Optional[Dict] = None):
        """Get existing collection or create new one"""
        collection = self.get_collection(collection_name)
        print(f"Got collection: {collection}")
        if collection is None:
            collection = self.create_collection(collection_name, metadata)
            print(f"Created collection: {collection}")
        return collection
    
    def add_documents(self, collection, documents: List[str], metadatas: List[Dict], ids: List[str]):
        """Add documents to collection"""
        try:
            # Convert to LangChain Document format
            langchain_docs = [
                Document(page_content=doc, metadata=meta)
                for doc, meta in zip(documents, metadatas)
            ]
            
            # Add documents to collection
            collection.add_documents(
                documents=langchain_docs,
                ids=ids
            )
            return True
        except Exception as e:
            print(f"Error adding documents: {e}")
            return False
    
    def get_collection_info(self, collection_name: str):
        """Get collection information"""
        try:
            collection = self.get_collection(collection_name)
            if not collection:
                return None
            
            # Access the underlying Chroma collection
            chroma_collection = collection._collection
            count = chroma_collection.count()
            
            return {
                "name": collection_name,
                "count": count,
                "metadata": {}  # LangChain doesn't expose collection metadata directly
            }
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return None
    
    def delete_collection(self, collection_name: str):
        """Delete collection"""
        try:
            collection = self.get_collection(collection_name)
            if collection:
                collection.delete_collection()
            return True
        except Exception as e:
            print(f"Error deleting collection: {e}")
            return False
    
    def list_collections(self):
        """List all collections"""
        try:
            # Create temporary client to list collections
            from chromadb import PersistentClient
            client = PersistentClient(path=self.persist_directory)
            return client.list_collections()
        except Exception as e:
            print(f"Error listing collections: {e}")
            return []
        
    def search(self, collection_name: str, query: str, n_results: int = 5, search_method: str = "hybrid"):
        """Search in collection using different methods"""
        if search_method == "similarity":
            final_results = self.similarity_search(collection_name, query, n_results)
        elif search_method == "mmr":
            final_results = self.max_marginal_relevance_search(collection_name, query, n_results)
        elif search_method == "bm25":
            final_results = self.bm25_search(collection_name, query, n_results)
        elif search_method == "hybrid":
            # Hybrid search can be implemented as a combination of similarity and MMR
            # This is a simple approach. I have tried to combine results from other methods
            # such as MMR+Keyword or Similarity+Keyword and Similarity+MMR+Keyword
            results = self.similarity_search(collection_name, query, n_results)
            if results:
                mmr_results = self.max_marginal_relevance_search(collection_name, query, n_results)
                # RRF fusion for hybrid search
                print("Combining results using RRF")
                final_results = self.reciprocal_rank_fusion([results, mmr_results], k=60)
        elif search_method == "bm25_mmr":
            # Hybrid search combining BM25 and MMR
            bm25_results = self.bm25_search(collection_name, query, n_results)
            if bm25_results:
                mmr_results = self.max_marginal_relevance_search(collection_name, query, n_results)
                # RRF fusion for hybrid search
                print("Combining results using RRF (BM25 + MMR)")
                final_results = self.reciprocal_rank_fusion([bm25_results, mmr_results], k=60)
        else:
            print(f"Unknown search method: {search_method}")
            return None
        
        return final_results
    
    def similarity_search(self, collection_name: str, query: str, n_results: int = 5):
        """Search in collection"""
        try:
            collection = self.get_collection(collection_name)
            if not collection:
                return None
            
            results = collection.similarity_search_with_score(
                query=query,
                k=n_results
            )
            
            # Format results to match original format
            formatted_results = {
                "ids": [],
                "documents": [],
                "metadatas": [],
                "distances": []
            }
            
            for doc, score in results:
                formatted_results["documents"].append(doc.page_content)
                formatted_results["metadatas"].append(doc.metadata)
                formatted_results["ids"].append(doc.metadata.get("id", ""))
                formatted_results["distances"].append(score)
                
            return formatted_results
        except Exception as e:
            print(f"Error searching: {e}")
            return None

    def max_marginal_relevance_search(self, collection_name: str, query: str, n_results: int = 5, diversity: float = 0.3):
        """Perform MMR search in collection"""
        try:
            collection = self.get_collection(collection_name)
            if not collection:
                return None
            
            results = collection.max_marginal_relevance_search(
                query=query,
                k=n_results,
                fetch_k=2*n_results,  # Fetch more candidates than needed
                lambda_mult=diversity  # Diversity factor (0-1)
            )
            # Format results to match original format
            formatted_results = {
                "ids": [],
                "documents": [],
                "metadatas": [],
                "distances": []
            }
            
            for doc in results:
                formatted_results["documents"].append(doc.page_content)
                formatted_results["metadatas"].append(doc.metadata)
                formatted_results["ids"].append(doc.metadata.get("id", ""))
                formatted_results["distances"].append(1.0)  # MMR doesn't provide scores, use 1.0 as placeholder
                
            return formatted_results
        except Exception as e:
            print(f"Error performing MMR search: {e}")
            return None
    
    def bm25_search(self, collection_name: str, query: str, n_results: int = 5):
        """Perform BM25 lexical search on the collection"""
        try:
            # Get the collection
            collection = self.get_collection(collection_name)
            if not collection:
                return None
            
            # Build or retrieve BM25 index if not already created
            if self.bm25_index is None:
                self._build_bm25_index(collection)
                
            # Tokenize the query the same way we did for documents
            tokenized_query = query.lower().split()
            
            # Get BM25 scores
            scores = self.bm25_index.get_scores(tokenized_query)
            top_indices = np.argsort(scores)[::-1][:n_results]
            
            # Format results
            formatted_results = {
                "ids": [],
                "documents": [],
                "metadatas": [],
                "distances": []
            }
            
            for idx in top_indices:
                # Skip documents with zero score
                if scores[idx] <= 0:
                    continue
                    
                doc_id = self.bm25_doc_ids[idx]
                doc_content = self.bm25_documents[idx]
                
                # Find the original metadata from collection
                langchain_doc = collection.get(ids=[doc_id])
                metadata = langchain_doc['metadatas'][0] if langchain_doc['metadatas'] else {}
                
                formatted_results["ids"].append(doc_id)
                formatted_results["documents"].append(doc_content)
                formatted_results["metadatas"].append(metadata)
                # Convert score to a distance-like metric (lower is better)
                formatted_results["distances"].append(1.0 / (1.0 + scores[idx]))
                
            return formatted_results
        except Exception as e:
            print(f"Error in BM25 search: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _build_bm25_index(self, collection):
        """Build BM25 index from collection documents"""
        # Get all documents from collection
        results = collection.get()
        
        # Store the original documents and their IDs
        self.bm25_documents = []
        self.bm25_doc_ids = []
        
        # Process documents for BM25
        tokenized_docs = []
        
        for doc, doc_id in zip(results['documents'], results['ids']):
            if hasattr(doc, 'page_content'):
                content = doc.page_content
            else:
                content = doc
                
            # Store the original content and ID
            self.bm25_documents.append(content)
            self.bm25_doc_ids.append(doc_id)
            
            # Tokenize for BM25 (simple lowercase and split)
            tokenized_docs.append(content.lower().split())
        
        # Create BM25 index
        self.bm25_index = BM25Okapi(tokenized_docs)
        print(f"Built BM25 index with {len(tokenized_docs)} documents")
    
    def reciprocal_rank_fusion(self, results_list, k=60):
        """
        Perform RRF on multiple result sets
        
        Args:
            results_list: List of search result dictionaries
            k: Constant to prevent division by zero and reduce impact of high rankings
        """
        try:
            # Track document scores across all result sets
            doc_scores = {}
            
            # Process each result set
            for results_idx, results in enumerate(results_list):
                if not results or not isinstance(results, dict):
                    continue
                    
                # Process each document by its rank position
                for rank, (doc, meta) in enumerate(zip(
                    results.get("documents", []),
                    results.get("metadatas", [])
                )):
                    # Create a unique key for this document
                    doc_id = meta.get("id", "")
                    if doc_id:
                        key = doc_id
                    else:
                        # Fallback to using document content hash if no ID
                        key = hashlib.md5(doc.encode()).hexdigest()
                        
                    # Initialize document entry if needed
                    if key not in doc_scores:
                        doc_scores[key] = {
                            "score": 0,
                            "document": doc,
                            "metadata": meta,
                            "id": doc_id,
                            "sources": []
                        }
                    
                    # Add RRF score: 1/(k + rank)
                    rrf_score = 1.0 / (k + rank)
                    doc_scores[key]["score"] += rrf_score
                    doc_scores[key]["sources"].append(f"{results_idx}:{rank}:{rrf_score:.4f}")
            
            # Sort documents by RRF score (descending)
            sorted_docs = sorted(
                doc_scores.values(), 
                key=lambda x: x["score"], 
                reverse=True
            )
            
            # Format results
            formatted_results = {
                "ids": [],
                "documents": [],
                "metadatas": [],
                "distances": [],  # Using RRF scores as inverse distances
                "rrf_scores": [],  # Keep original RRF scores for reference
                "rrf_sources": []  # Track which strategies contributed to each result
            }
            
            for doc in sorted_docs:
                formatted_results["ids"].append(doc["id"])
                formatted_results["documents"].append(doc["document"])
                formatted_results["metadatas"].append(doc["metadata"])
                # Use 1/score as a distance-like metric (lower is better)
                formatted_results["distances"].append(1.0 / doc["score"] if doc["score"] > 0 else float('inf'))
                formatted_results["rrf_scores"].append(doc["score"])
                formatted_results["rrf_sources"].append(doc["sources"])
                
            return formatted_results
        except Exception as e:
            print(f"Error in RRF fusion: {e}")
            traceback.print_exc()