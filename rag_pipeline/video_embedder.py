#!/usr/bin/env python3
"""
Video Embedder - Handles embedding and vectorization of video summaries
"""

import os
import json
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from rag_pipeline.chroma_handler import ChromaDBHandler
from sentence_transformers import CrossEncoder

class VideoEmbedder:
    def __init__(self, persist_directory="chroma_db", model_name="all-MiniLM-L6-v2"):
        """Initialize the embedder with ChromaDB handler and SentenceTransformer"""
        self.persist_directory = persist_directory
        self.model_name = model_name
        self.embedder = SentenceTransformer(model_name, device='cpu')
        
        # Initialize database handler (easily swappable)
        self.db_handler = ChromaDBHandler(self.embedder, self.persist_directory)
        self.reranking = True

        if self.reranking:
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    def load_summary_json(self, json_path):
        """Load video summary from JSON file"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                summary_data = json.load(f)
            return summary_data
        except Exception as e:
            print(f"Error loading summary JSON: {e}")
            return None
    
    def create_embeddings(self, summary_data):
        """Create embeddings for video summary data"""
        print("Creating embeddings for summaries...")
        
        # Create full text representation
        full_text = f"Video: {summary_data['video_name']}\n"
        
        for chunk in summary_data['chunks']:
            full_text += f"Time {chunk['timestamp']}: {chunk['summary']}\n"
        
        # Add embedding info to summary
        embedding_info = {
            'full_text': full_text,
            'text_length': len(full_text),
            'embedding_ready': True,
            'embedding_date': datetime.now().isoformat(),
            'model_used': self.model_name
        }
        
        return embedding_info
    
    def vectorize_summary(self, summary_data, collection_name="video_summaries"):
        """Vectorize summary and store in database"""
        print(f"Vectorizing summary: {summary_data['video_name']}")
        
        # Get or create collection
        collection = self.db_handler.get_or_create_collection(collection_name)
        print(f"Using collection: {collection}")
        if not collection:
            raise Exception("Failed to create/get collection")
        
        # Prepare documents for vectorization
        documents = []
        metadatas = []
        ids = []
        
        for chunk in summary_data["chunks"]:
            summary_text = chunk["summary"]
            metadata = {
                "chunk_number": str(chunk["chunk_number"]),
                "start_time": str(chunk["start_time"]),
                "end_time": str(chunk["end_time"]),
                "timestamp": chunk["timestamp"],
                "video_name": summary_data["video_name"],
                "video_path": summary_data["video_path"],
                "duration": str(chunk["duration"])
            }
            doc_id = f"{summary_data['video_name']}_chunk_{chunk['chunk_number']}"
            
            documents.append(summary_text)
            metadatas.append(metadata)
            ids.append(doc_id)
        
        # Add documents to collection
        success = self.db_handler.add_documents(collection, documents, metadatas, ids)
        if not success:
            raise Exception("Failed to add documents to collection")
        
        print(f"✅ Summary vectorized and stored in database!")
        return collection
    
    def process_summary_json(self, json_path, collection_name="video_summaries"):
        """Main function to process JSON summary and create embeddings"""
        print("=" * 50)
        print("VIDEO EMBEDDING STARTING")
        print("=" * 50)
        
        # Load summary data
        summary_data = self.load_summary_json(json_path)
        if not summary_data:
            raise Exception("Failed to load summary data")
        
        # Create embeddings info
        embedding_info = self.create_embeddings(summary_data)
        
        # Add embedding info to summary
        summary_data['embedding_info'] = embedding_info
        
        # Vectorize and store in database
        collection = self.vectorize_summary(summary_data, collection_name)
        
        # Update the JSON file with embedding info
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        print("=" * 50)
        print("VIDEO EMBEDDING COMPLETED!")
        print(f"Updated JSON with embedding info: {json_path}")
        print(f"Data stored in database collection: {collection_name}")
        print("=" * 50)
        
        return summary_data, collection
    
    def search_videos(self, query, collection_name="video_summaries", n_results=5):
        """Search through vectorized video summaries"""
        results = self.db_handler.search(collection_name, query, n_results)
        if self.reranking and results:
            pairs = [(query, doc) for doc in results["documents"]]
            # Get scores from reranker
            rerank_scores = self.reranker.predict(pairs)
            
            # Create reranked results
            reranked_indices = sorted(range(len(rerank_scores)), 
                                    key=lambda i: rerank_scores[i], reverse=True)[:n_results]
            
            # Format reranked results
            formatted_results = {"ids": [], "documents": [], "metadatas": [], "distances": []}
            
            for idx in reranked_indices:
                formatted_results["documents"].append(results["documents"][idx])
                formatted_results["metadatas"].append(results["metadatas"][idx])
                formatted_results["ids"].append(results["ids"][idx])
                formatted_results["distances"].append(rerank_scores[idx])  # Use reranker score
                
            return formatted_results
        return results
    
    def get_collection_info(self, collection_name="video_summaries"):
        """Get information about the collection"""
        return self.db_handler.get_collection_info(collection_name)
    
    def list_collections(self):
        """List all available collections"""
        return self.db_handler.list_collections()
    
    def delete_collection(self, collection_name):
        """Delete a collection"""
        return self.db_handler.delete_collection(collection_name)
