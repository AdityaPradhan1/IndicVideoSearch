from rag_pipeline.video_embedder import VideoEmbedder
from chat_app.query_transformation import QueryTransformer
from chat_app.client_manager import ClientManager
import pandas as pd
import argparse
from collections import defaultdict
import numpy as np
import math
import time
import os

def create_unique_chunk_id(video_name, chunk_number):
    """Create a unique ID for a chunk by combining video name and chunk number"""
    # Clean video name to create a valid identifier
    video_slug = video_name.lower().replace(' ', '_').replace('|', '').replace('-', '_')
    return f"{video_slug}_chunk_{chunk_number}"

def create_evaluation_dataset():
    """
    Creates a dataset of test queries and their ground truth relevant chunks
    based on the 'Deep Learning in Tamil Part 5' summary.
    """
    video_name = "Deep Learning in Tamil | Chain Rule in Back Propagation | Deep Learning for Beginners | Part 5.mp4"

    test_queries = [
        "What is the main topic of this video?",
        "What is the formula for the weight update rule in gradient descent?",
        "What is the Sum of Squared Errors loss function?",
        "How is the chain rule used to update weights in a neural network?",
        "How do you calculate the gradient for a weight in the final layer?",
        "What is the difference between updating weights in the final layer versus an earlier hidden layer?",
        "How does a weight in a hidden layer influence the final loss?",
        "What does it mean to 'backpropagate' the error?",
        "Can you show the full chain rule formula for calculating the gradient of a weight in a hidden layer?",
        "What is the overall goal of training a neural network as described in the video?"
    ]
    
    # Each query maps to video name + chunk number combinations
    relevant_chunks = {
        "What is the main topic of this video?": [
            (video_name, 1),
            (video_name, 3),
            (video_name, 10),
        ],
        "What is the formula for the weight update rule in gradient descent?": [
            (video_name, 2),
            (video_name, 3),
            (video_name, 5),
            (video_name, 6),
        ],
        "What is the Sum of Squared Errors loss function?": [
            (video_name, 2),
            (video_name, 5),
            (video_name, 8),
        ],
        "How is the chain rule used to update weights in a neural network?": [
            (video_name, 3),
            (video_name, 4),
            (video_name, 5),
            (video_name, 6),
        ],
        "How do you calculate the gradient for a weight in the final layer?": [
            (video_name, 3),
            (video_name, 4),
            (video_name, 6),
        ],
        "What is the difference between updating weights in the final layer versus an earlier hidden layer?": [
            (video_name, 7),
            (video_name, 8),
        ],
        "How does a weight in a hidden layer influence the final loss?": [
            (video_name, 7),
            (video_name, 8),
            (video_name, 9),
        ],
        "What does it mean to 'backpropagate' the error?": [
            (video_name, 8),
            (video_name, 9),
            (video_name, 10),
        ],
        "Can you show the full chain rule formula for calculating the gradient of a weight in a hidden layer?": [
            (video_name, 9),
        ],
        "What is the overall goal of training a neural network as described in the video?": [
            (video_name, 2),
        ]
    }
    
    # Convert to unique IDs
    relevant_ids = {}
    for query, chunks in relevant_chunks.items():
        relevant_ids[query] = [create_unique_chunk_id(video, chunk_num) for video, chunk_num in chunks]
    
    return test_queries, relevant_ids

def load_evaluation_dataset_csv(dataset_id="deep_learning_tamil", csv_path="data/evaluation_datasets.csv"):
    """Load an evaluation dataset from CSV by ID"""
    df = pd.read_csv(csv_path)
    df_filtered = df[df['dataset_id'] == dataset_id]
    
    if len(df_filtered) == 0:
        raise ValueError(f"Dataset ID {dataset_id} not found in CSV")
    
    # Get unique queries
    test_queries = df_filtered['query'].unique().tolist()
    
    # Group relevant chunks by query
    relevant_chunks = defaultdict(list)
    for _, row in df_filtered.iterrows():
        relevant_chunks[row['query']].append((row['video_name'], row['chunk_number']))
    
    # Convert to unique IDs
    relevant_ids = {}
    for query, chunks in relevant_chunks.items():
        relevant_ids[query] = [create_unique_chunk_id(video, chunk_num) for video, chunk_num in chunks]
    
    return test_queries, relevant_ids

def load_all_evaluation_datasets_csv(csv_path="rag_eval/custom_dataset.csv"):
    """Load ALL evaluation datasets from CSV regardless of dataset_id"""
    df = pd.read_csv(csv_path)
    
    if len(df) == 0:
        raise ValueError(f"No data found in CSV at {csv_path}")
    
    # Get unique queries across all datasets
    test_queries = df['query'].unique().tolist()
    
    # Group relevant chunks by query
    relevant_chunks = defaultdict(list)
    for _, row in df.iterrows():
        relevant_chunks[row['query']].append((row['video_name'], int(row['chunk_number'])))
    
    # Convert to unique IDs
    relevant_ids = {}
    for query, chunks in relevant_chunks.items():
        relevant_ids[query] = [create_unique_chunk_id(video, chunk_num) for video, chunk_num in chunks]
    
    return test_queries, relevant_ids

def evaluate_rag_pipeline(embedder, test_queries, relevant_ids,
                          collection_name="video_summaries", search_method="bm25_mmr", transformation_method="rag_fusion"):
    results = []
    client_manager = ClientManager()
    client_manager.initialize_llm("gemini-2.0-flash-lite", 0.7)
    query_transformer = QueryTransformer(client_manager.llm)
    
    # Initialize performance metrics
    total_transformation_time = 0
    total_search_time = 0
    total_queries_generated = 0
    
    for query in test_queries:
        # Get the relevant chunk IDs for this query
        true_relevant = relevant_ids[query]
        
        # Measure query transformation time
        transform_start = time.time()
        transformed_queries = query_transformer.transform_query(query, transformation_method)
        transform_end = time.time()
        transformation_time = transform_end - transform_start
        total_transformation_time += transformation_time
        
        # Handle different return types from transform_query
        if transformation_method == "hyde":
            # HyDE returns a tuple of (original_query, document)
            original_query, hyde_doc = transformed_queries
            transformed_queries = [hyde_doc]  # Use generated document for retrieval
            total_queries_generated += 1
        else:
            # Other methods return a list of queries
            if not transformed_queries:
                transformed_queries = [query]
            total_queries_generated += len(transformed_queries)

        # Initialize combined results
        search_results = {
            'documents': [],
            'metadatas': [],
            'distances': []
        }
        
        # Measure search time
        search_start = time.time()
        for i, transformed_query in enumerate(transformed_queries, 1):
            qresults = embedder.search_videos(transformed_query, collection_name=collection_name, n_results=5, search_method=search_method)
            
            if qresults and isinstance(qresults, dict) and 'documents' in qresults:
                search_results['documents'].extend(qresults['documents'])
                search_results['metadatas'].extend(qresults['metadatas'])
                search_results['distances'].extend(qresults['distances'])
        search_end = time.time()
        search_time = search_end - search_start
        total_search_time += search_time
        
        
        # Extract unique IDs from the returned results (PREVENT DUPLICATES)
        retrieved_ids = []
        seen_ids = set()
        
        for metadata in search_results["metadatas"]:
            video_name = metadata["video_name"]
            chunk_number = metadata["chunk_number"]
            unique_id = create_unique_chunk_id(video_name, chunk_number)
            
            # Only add if we haven't seen this ID yet
            if unique_id not in seen_ids:
                retrieved_ids.append(unique_id)
                seen_ids.add(unique_id)
        
        # Calculate metrics (with safeguard)
        relevant_retrieved = [id for id in retrieved_ids if id in true_relevant]
        
        # Ensure we don't count more relevant items than exist
        if len(relevant_retrieved) > len(true_relevant):
            print(f"WARNING: Found more relevant items than exist for query: {query}")
            print(f"Retrieved: {relevant_retrieved}")
            print(f"True relevant: {true_relevant}")
            relevant_retrieved = relevant_retrieved[:len(true_relevant)]
        
        # Basic metrics
        precision_at_5 = len(relevant_retrieved) / min(5, len(retrieved_ids))
        recall_at_5 = len(relevant_retrieved) / len(true_relevant)
        
        # F1 score - harmonic mean of precision and recall
        f1_score = 0
        if precision_at_5 + recall_at_5 > 0:  # Avoid division by zero
            f1_score = 2 * (precision_at_5 * recall_at_5) / (precision_at_5 + recall_at_5)
            
        # Mean Reciprocal Rank (MRR) - position of first relevant result
        mrr = 0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in true_relevant:
                # +1 because ranks start at 1, not 0
                mrr = 1.0 / (i + 1)
                break
        
        # Normalized Discounted Cumulative Gain (nDCG)
        # For each position, relevance is 1 if document is relevant, 0 otherwise
        dcg = 0
        for i, doc_id in enumerate(retrieved_ids[:5]):
            rel = 1 if doc_id in true_relevant else 0
            # Position i+1 because positions start at 1
            dcg += rel / math.log2(i + 2)  
            
        # Ideal DCG - if all relevant docs came first (up to k=5)
        idcg = 0
        for i in range(min(len(true_relevant), 5)):
            idcg += 1 / math.log2(i + 2)
            
        ndcg = 0
        if idcg > 0:  # Avoid division by zero
            ndcg = dcg / idcg
        
        # Calculate efficiency metrics - relevance per second
        total_time = transformation_time + search_time
        
        results.append({
            "query": query,
            "precision@5": precision_at_5,
            "recall@5": recall_at_5,
            "f1_score": f1_score,
            "mrr": mrr,
            "ndcg@5": ndcg,
            "retrieved_ids": retrieved_ids,
            "relevant_ids": true_relevant,
            "matched_ids": relevant_retrieved,
            # Performance metrics
            "transformation_time": transformation_time,
            "search_time": search_time,
            "total_time": total_time,
            })
    
    # Calculate average metrics
    avg_precision = sum(r["precision@5"] for r in results) / len(results)
    avg_recall = sum(r["recall@5"] for r in results) / len(results)
    avg_f1 = sum(r["f1_score"] for r in results) / len(results)
    avg_mrr = sum(r["mrr"] for r in results) / len(results)
    avg_ndcg = sum(r["ndcg@5"] for r in results) / len(results)
    
    # Calculate average performance metrics
    avg_transformation_time = total_transformation_time / len(results)
    avg_search_time = total_search_time / len(results)
    avg_total_time = avg_transformation_time + avg_search_time
    
    # Return the complete evaluation results
    return {
        "detailed_results": results,
        "average_precision@5": avg_precision,
        "average_recall@5": avg_recall,
        "average_f1_score": avg_f1,
        "average_mrr": avg_mrr,
        "average_ndcg@5": avg_ndcg,
        # Performance averages
        "average_transformation_time": avg_transformation_time,
        "average_search_time": avg_search_time, 
        "average_total_time": avg_total_time,
        "efficiency_score": (avg_f1 / avg_total_time) if avg_total_time > 0 else 0
    }

if __name__ == "__main__":   
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline")
    parser.add_argument("--dataset", default="deep_learning_tamil", help="Dataset ID to evaluate")
    parser.add_argument("--csv-path", default="rag_eval/custom_dataset.csv", help="Path to the CSV file")
    args = parser.parse_args() 
    
    # Initialize the embedder
    embedder = VideoEmbedder()
    
    # Load the evaluation dataset
    if args.dataset == "all":
        # Use all questions from the CSV
        test_queries, relevant_ids = load_all_evaluation_datasets_csv(args.csv_path)
        print(f"Loaded {len(test_queries)} questions from all datasets")
    else:
        # Use only questions from the specified dataset
        test_queries, relevant_ids = load_evaluation_dataset_csv(args.dataset, args.csv_path)
        print(f"Loaded {len(test_queries)} questions from dataset '{args.dataset}'")    
    
    # Evaluate the RAG pipeline
    transformation_methods = ["rag_fusion", "hyde", "decomposition", "expansion"]
    results = {}
    
    for method in transformation_methods:
        print(f"\nEvaluating with transformation method: {method}")
        evaluation_results = evaluate_rag_pipeline(embedder, test_queries, relevant_ids, 
                                                  collection_name="video_summaries", 
                                                  search_method="bm25_mmr", 
                                                  transformation_method=method)
        
        # Print the results
        print("\nEffectiveness Metrics:")
        print(f"Average Precision@5: {evaluation_results['average_precision@5']:.4f}")
        print(f"Average Recall@5: {evaluation_results['average_recall@5']:.4f}")
        print(f"Average F1 Score: {evaluation_results['average_f1_score']:.4f}")
        print(f"Average MRR: {evaluation_results['average_mrr']:.4f}")
        print(f"Average nDCG@5: {evaluation_results['average_ndcg@5']:.4f}")
        
        print("\nEfficiency Metrics:")
        print(f"Average Transformation Time: {evaluation_results['average_transformation_time']:.3f} sec")
        print(f"Average Search Time: {evaluation_results['average_search_time']:.3f} sec")
        print(f"Average Total Time: {evaluation_results['average_total_time']:.3f} sec")
        print(f"Efficiency Score (F1/sec): {evaluation_results['efficiency_score']:.5f}")
        
        results[method] = {
            # Effectiveness metrics
            "average_precision@5": evaluation_results["average_precision@5"],
            "average_recall@5": evaluation_results["average_recall@5"],
            "average_f1_score": evaluation_results["average_f1_score"],
            "average_mrr": evaluation_results["average_mrr"],
            "average_ndcg@5": evaluation_results["average_ndcg@5"],
            # Efficiency metrics
            "average_transformation_time": evaluation_results["average_transformation_time"],
            "average_search_time": evaluation_results["average_search_time"],
            "average_total_time": evaluation_results["average_total_time"],
            "efficiency_score": evaluation_results["efficiency_score"]
        }


    print("\n\nFinal Results Summary:")

    # Print effectiveness metrics
    print("\nEffectiveness Metrics:")
    print("Method      | Precision@5 | Recall@5 | F1 Score | MRR     | nDCG@5")
    print("------------|-------------|----------|----------|---------|--------")
    for method, metrics in results.items():
        print(f"{method.ljust(12)}| "
              f"{metrics['average_precision@5']:.4f}    | "
              f"{metrics['average_recall@5']:.4f}  | "
              f"{metrics['average_f1_score']:.4f}  | "
              f"{metrics['average_mrr']:.4f} | "
              f"{metrics['average_ndcg@5']:.4f}")
    
    # Print efficiency metrics
    print("\nEfficiency Metrics:")
    print("Method      | Transform Time | Search Time | Total Time | Eff. Score")
    print("------------|----------------|-------------|------------|------------")
    for method, metrics in results.items():
        print(f"{method.ljust(12)}| "
              f"{metrics['average_transformation_time']:.3f} sec      | "
              f"{metrics['average_search_time']:.3f} sec   | "
              f"{metrics['average_total_time']:.3f} sec  | "
              f"{metrics['efficiency_score']:.5f}")