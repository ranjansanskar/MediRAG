import os
import json
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("Importing empathyagentpipeline...", flush=True)
from empathyagentpipeline import retrieve_hybrid, llm_temperature_low, doctor_prompt
print("Imported empathyagentpipeline successfully.", flush=True)

from langchain_core.prompts import PromptTemplate

faithfulness_prompt = PromptTemplate.from_template(
    """You are an expert judge evaluating a RAG pipeline.
Given the original context and the generated answer, score the Faithfulness from 0.0 to 1.0.
Context: {context}
Generated Answer: {answer}
Output ONLY a float number between 0.0 and 1.0."""
)

relevancy_prompt = PromptTemplate.from_template(
    """You are an expert judge evaluating a RAG pipeline.
Given the original user query and the generated answer, score the Answer Relevancy from 0.0 to 1.0.
User Query: {query}
Generated Answer: {answer}
Output ONLY a float number between 0.0 and 1.0."""
)

def evaluate_faithfulness(context, answer):
    chain = faithfulness_prompt | llm_temperature_low
    print("      Invoking faithfulness...", flush=True)
    res = chain.invoke({"context": context, "answer": answer})
    print("      Faithfulness done.", flush=True)
    try:
        return float(res.content.strip())
    except ValueError:
        return 0.5

def evaluate_relevancy(query, answer):
    chain = relevancy_prompt | llm_temperature_low
    print("      Invoking relevancy...", flush=True)
    res = chain.invoke({"query": query, "answer": answer})
    print("      Relevancy done.", flush=True)
    try:
        return float(res.content.strip())
    except ValueError:
        return 0.5

def run_evaluation():
    print("Entering run_evaluation()", flush=True)
    test_file = os.path.join(os.path.dirname(__file__), "rag_test_set.json")
    with open(test_file, 'r') as f:
        test_set = json.load(f)
        
    test_set = test_set[:1]
    
    print(f"Evaluating {len(test_set)} queries...", flush=True)
    
    for i, test_case in enumerate(test_set):
        q = test_case['query']
        print(f"  Retrieving docs for: {q[:20]}...", flush=True)
        docs = retrieve_hybrid(q)
        context_texts = [d.page_content for d in docs]
        context_str = "\n\n".join(context_texts)
        
        print("  Invoking doctor_prompt...", flush=True)
        chain = doctor_prompt | llm_temperature_low
        res = chain.invoke({
            "query": q,
            "context": context_str,
            "symptoms": "unknown",
            "past_history_context": "None"
        })
        answer = res.content
        print("  Doctor prompt done.", flush=True)
        
        f_score = evaluate_faithfulness(context_str, answer)
        r_score = evaluate_relevancy(q, answer)
        print(f"  Results: F={f_score}, R={r_score}", flush=True)

if __name__ == "__main__":
    run_evaluation()
