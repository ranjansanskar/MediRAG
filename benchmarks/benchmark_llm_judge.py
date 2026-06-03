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

import time

def run_evaluation():
    print("Entering run_evaluation()", flush=True)
    test_file = os.path.join(os.path.dirname(__file__), "rag_test_set.json")
    with open(test_file, 'r') as f:
        test_set = json.load(f)
        
    test_set = test_set[:20]
    
    print(f"Evaluating {len(test_set)} queries...", flush=True)
    
    total_faithfulness = 0.0
    total_relevancy = 0.0
    
    with open(os.path.join(os.path.dirname(__file__), "evaluation_results.txt"), "w") as out:
        out.write("RAG Pipeline LLM-as-a-judge Evaluation\\n")
        out.write("="*40 + "\\n\\n")
        
        for i, test_case in enumerate(test_set):
            q = test_case['query']
            print(f"\\n[{i+1}/20] Query: {q[:40]}...", flush=True)
            
            # 1. Retrieve & Generate
            docs = retrieve_hybrid(q)
            context_texts = [d.page_content for d in docs]
            context_str = "\\n\\n".join(context_texts)
            
            chain = doctor_prompt | llm_temperature_low
            res = chain.invoke({
                "query": q,
                "context": context_str,
                "symptoms": "unknown",
                "past_history_context": "None"
            })
            answer = res.content
            
            # 2. Evaluate
            f_score = evaluate_faithfulness(context_str, answer)
            r_score = evaluate_relevancy(q, answer)
            
            total_faithfulness += f_score
            total_relevancy += r_score
            
            log_str = f"Q{i+1}: F={f_score:.2f}, R={r_score:.2f} | {q}\\n"
            print("  -> " + log_str.strip(), flush=True)
            out.write(log_str)
            out.flush()
            
            if i < len(test_set) - 1:
                print("  Waiting 15 seconds to respect OpenRouter rate limits...", flush=True)
                time.sleep(15)

        avg_f = total_faithfulness / len(test_set)
        avg_r = total_relevancy / len(test_set)
        
        summary = f"\\nFinal Average:\\nFaithfulness: {avg_f:.2f}\\nAnswer Relevancy: {avg_r:.2f}\\n"
        print(summary, flush=True)
        out.write(summary)

if __name__ == "__main__":
    run_evaluation()
