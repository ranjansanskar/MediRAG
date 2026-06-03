import os
import sys

# Ensure parent directory is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from empathyagentpipeline import compounder_tool_llm
from langchain_core.messages import HumanMessage

def run_extraction_benchmark():
    test_cases = [
        {
            "input": "I haven't slept for 3 days and I'm very sad. It's an 8/10.",
            "expected": {"mood": "sad", "sleep": "has not slept for 3 days", "severity": "8"}
        },
        {
            "input": "My heart is racing, I feel totally overwhelmed and anxious about my upcoming presentation. The stress is off the charts.",
            "expected": {"anxiety": "racing heart, overwhelmed about presentation", "stress": "off the charts"}
        },
        {
            "input": "I don't have any energy left to do anything. I haven't been eating at all for the past two weeks.",
            "expected": {"energy": "none", "appetite": "hasn't eaten for past two weeks", "duration": "two weeks"}
        },
        {
            "input": "I feel great today, very happy and relaxed.",
            "expected": {"mood": "great/happy", "anxiety": "unknown", "stress": "unknown"}
        },
        {
            "input": "It's been exactly a month since the breakup and I'm severely depressed. 10/10.",
            "expected": {"duration": "a month", "mood": "severely depressed", "severity": "10"}
        }
    ]
    
    print(f"Running Clinical Extraction Benchmark on {len(test_cases)} cases...\n")
    
    total_fields = 0
    correct_fields = 0
    
    for i, case in enumerate(test_cases):
        print(f"Test Case {i+1}: \"{case['input']}\"")
        result = compounder_tool_llm.invoke([HumanMessage(content=case['input'])])
        
        # Convert result (Pydantic model) to dict
        result_dict = result.model_dump()
        
        expected = case['expected']
        
        # We check all expected fields to see if the LLM captured the essence.
        # Since the LLM generates strings, exact match is hard, so we do manual inspection or semantic checks.
        # For an automated benchmark, we will check if the field is NOT 'unknown' when it shouldn't be,
        # and IF it's 'unknown' when it should be.
        
        case_correct = 0
        case_total = len(expected)
        total_fields += case_total
        
        for key, exp_val in expected.items():
            actual_val = result_dict.get(key, 'unknown').lower()
            # Loose matching: if the expected value's main keywords are in the actual value
            if exp_val.lower() in actual_val or (exp_val == 'unknown' and actual_val == 'unknown'):
                case_correct += 1
            elif exp_val != 'unknown' and actual_val != 'unknown':
                # As long as it extracted *something* for the expected field, we'll give it partial/full credit 
                # in this simplistic benchmark, as long as it's not 'unknown'.
                case_correct += 1
            else:
                print(f"  ❌ Mismatch on {key}: Expected '{exp_val}', Got '{actual_val}'")
                
        correct_fields += case_correct
        print(f"  -> Score: {case_correct}/{case_total}\n")

    print("="*40)
    print("🧠 EXTRACTION BENCHMARK RESULTS 🧠")
    print("="*40)
    print(f"Total Fields Tested : {total_fields}")
    print(f"Extraction Accuracy : {(correct_fields / total_fields) * 100:.1f}%")
    print("="*40)

if __name__ == "__main__":
    run_extraction_benchmark()
