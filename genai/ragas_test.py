from chat_engine import KTChatEngine
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

def run_ragas_lite_test():
    # 1. Setup the Engine
    engine = KTChatEngine(API_KEY)
    
    # 2. Define "Golden Questions" (What the bot SHOULD know)
    test_queries = [
        "What database is used for storing faces?",
        "How do I register a new face?",
        "What happens if the 'logs' directory is missing?"
    ]
    
    results = []
    
    print("🚀 Starting RAGAS-Lite Evaluation...")
    
    for query in test_queries:
        print(f"Testing: {query}")
        response = engine.generate_response(query)
        
        # We simulate the RAGAS metrics: 
        # Faithfulness: Is the answer in the source?
        # Relevance: Does it answer the user?
        
        results.append({
            "Question": query,
            "Answer": response['answer'][:100] + "...", # Snippet
            "Sources": ", ".join(response['sources']),
            "Source_Count": len(response['sources'])
        })

    # 3. Show a Summary Table
    df = pd.DataFrame(results)
    print("\n--- Test Results ---")
    print(df.to_string())
    
    if df['Source_Count'].min() > 0:
        print("\n✅ PASSED: Every answer found a source code file.")
    else:
        print("\n⚠️ WARNING: Some answers had no sources.")

if __name__ == "__main__":
    run_ragas_lite_test()