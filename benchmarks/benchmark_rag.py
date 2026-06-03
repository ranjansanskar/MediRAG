import os
import json
import sys

# Ensure parent directory is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from empathyagentpipeline import retrieve_hybrid

def run_benchmark():
    test_file = os.path.join(os.path.dirname(__file__), "rag_test_set.json")
    if not os.path.exists(test_file):
        print(f"Error: Could not find {test_file}. Please run generate_test_data.py first.")
        return
        
    with open(test_file, 'r') as f:
        test_set = json.load(f)
        
    print(f"Starting Benchmark on {len(test_set)} synthetic queries...\n")
    
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    mrr_sum = 0.0
    
    for i, test_case in enumerate(test_set):
        query = test_case['query']
        expected_text = test_case['expected_text']
        
        # We retrieve top 5 results
        # We need to temporarily patch the k in the retriever if we want more, 
        # but let's just use the default retrieve_hybrid which returns top docs
        retrieved_docs = retrieve_hybrid(query)
        
        # Find the rank of the expected document
        rank = None
        for j, doc in enumerate(retrieved_docs):
            # Check if expected text is in retrieved text (or vice versa due to overlapping/chunking differences)
            # A simple substring check or exact match since we used the exact chunk
            if expected_text.strip() == doc.page_content.strip():
                rank = j + 1
                break
                
        if rank is not None:
            if rank == 1: hits_at_1 += 1
            if rank <= 3: hits_at_3 += 1
            if rank <= 5: hits_at_5 += 1
            mrr_sum += (1.0 / rank)
            
        status = f"HIT (Rank {rank})" if rank else "MISS"
        print(f"Query {i+1}: {query[:50]}... -> {status}")
        
    # Calculate final metrics
    total = len(test_set)
    print("\n" + "="*40)
    print("🏆 MEDIRAG RAG BENCHMARK RESULTS 🏆")
    print("="*40)
    print(f"Total Queries Tested : {total}")
    print(f"Hit Rate @ 1         : {(hits_at_1 / total) * 100:.1f}%")
    print(f"Hit Rate @ 3         : {(hits_at_3 / total) * 100:.1f}%")
    print(f"Hit Rate @ 5         : {(hits_at_5 / total) * 100:.1f}%")
    print(f"MRR (Mean Rec. Rank) : {(mrr_sum / total):.3f}")
    print("="*40)

if __name__ == "__main__":
    run_benchmark()
