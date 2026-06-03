import os
import json
import sys

# Ensure parent directory is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import Faithfulness, AnswerRelevancy
from empathyagentpipeline import retrieve_hybrid, llm_temperature_low, doctor_prompt, embeddings

from openai import OpenAI
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings

# Create OpenAI client pointing to OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY")
)
ragas_llm = llm_factory("gpt-4o-mini", client=client)

# Native Ragas HuggingFace Embeddings
ragas_emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def run_ragas():
    test_file = os.path.join(os.path.dirname(__file__), "rag_test_set.json")
    if not os.path.exists(test_file):
        print(f"Error: Could not find {test_file}.")
        return
        
    with open(test_file, 'r') as f:
        test_set = json.load(f)
        
    # Limit to 2 questions to prevent OpenRouter rate limits (429) causing exponential backoff hangs
    test_set = test_set[:2]
    
    questions = []
    answers = []
    contexts_list = []
    
    print(f"Generating Answers for {len(test_set)} queries to evaluate with Ragas...")
    
    for i, test_case in enumerate(test_set):
        q = test_case['query']
        print(f"Processing Query {i+1}: {q[:50]}...")
        
        docs = retrieve_hybrid(q)
        context_texts = [d.page_content for d in docs]
        context_str = "\n\n".join(context_texts)
        
        chain = doctor_prompt | llm_temperature_low
        res = chain.invoke({
            "query": q,
            "context": context_str,
            "symptoms": "unknown",
            "past_history_context": "None"
        })
        
        questions.append(q)
        answers.append(res.content)
        contexts_list.append(context_texts)
        
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
    }
    
    dataset = Dataset.from_dict(data)
    
    print("\nRunning LLM-as-a-judge Evaluation (Faithfulness & Answer Relevance)...")
    
    try:
        result = evaluate(
            dataset,
            metrics=[Faithfulness(llm=ragas_llm), AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb)],
            llm=ragas_llm,
            embeddings=ragas_emb,
            raise_exceptions=False
        )
        
        print("\n" + "="*40)
        print("🧠 RAGAS EVALUATION RESULTS 🧠")
        print("="*40)
        print(result)
        print("="*40)
        
    except Exception as e:
        print(f"\nError during Ragas evaluation: {e}")

if __name__ == "__main__":
    run_ragas()
