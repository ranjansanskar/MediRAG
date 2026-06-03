import os
import json
import random
import sys

# Ensure parent directory is in path so we can import empathyagentpipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from empathyagentpipeline import db, llm_temperature_low
from langchain_core.prompts import PromptTemplate

def generate_dataset(num_samples=20):
    print("Fetching chunks from Chroma DB...")
    # Get all documents from Chroma
    data = db.get()
    
    ids = data['ids']
    docs = data['documents']
    
    if not ids:
        print("No documents found in the database. Ensure the server ran at least once.")
        return
        
    # Zip and randomly sample
    zipped = list(zip(ids, docs))
    # Filter out chunks that are too short (likely junk or titles)
    zipped = [z for z in zipped if len(z[1]) > 200]
    
    samples = random.sample(zipped, min(num_samples, len(zipped)))
    
    print(f"Generating synthetic queries for {len(samples)} chunks...")
    
    prompt = PromptTemplate.from_template(
        "You are an expert at creating RAG evaluation datasets.\n"
        "Given the following text from a clinical mental health workbook, write a single, realistic user query (1-2 sentences) "
        "that a person struggling with their mental health might say, where this text would be the PERFECT answer.\n"
        "Make it sound like a natural message sent to a therapy chatbot.\n\n"
        "TEXT:\n{context}\n\n"
        "QUERY:"
    )
    
    test_set = []
    
    for i, (chunk_id, text) in enumerate(samples):
        print(f"Processing {i+1}/{len(samples)}...")
        try:
            chain = prompt | llm_temperature_low
            res = chain.invoke({"context": text})
            query = res.content.strip().replace('"', '')
            
            test_set.append({
                "id": chunk_id,
                "query": query,
                "expected_text": text
            })
        except Exception as e:
            print(f"Error generating query for chunk {chunk_id}: {e}")
            
    # Save to file
    out_path = os.path.join(os.path.dirname(__file__), "rag_test_set.json")
    with open(out_path, "w") as f:
        json.dump(test_set, f, indent=4)
        
    print(f"\nSuccessfully generated {len(test_set)} test cases and saved to {out_path}!")

if __name__ == "__main__":
    # We will generate 30 queries for a solid benchmark
    generate_dataset(30)
