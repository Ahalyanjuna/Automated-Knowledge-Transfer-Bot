import json
import os
from dotenv import load_dotenv
from genai.chat_engine import KTChatEngine

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

class KTEvaluator:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("❌ API Key not found. Please check your .env file.")
        self.engine = KTChatEngine(api_key)

    def evaluate_answer(self, query, response, context):
        """
        Asks the LLM to grade its own response on a scale of 1-5.
        """
        eval_prompt = f"""
        Analyze the following RAG interaction and provide a score from 1 to 5 
        for 'Faithfulness' (accuracy to code) and 'Relevance' (usefulness).
        
        QUERY: {query}
        CONTEXT PROVIDED: {context}
        AI RESPONSE: {response}
        
        Return ONLY a JSON object: {{"faithfulness": score, "relevance": score, "reason": "short explanation"}}
        """
        
        completion = self.engine.client.chat.completions.create(
            messages=[{"role": "user", "content": eval_prompt}],
            model="llama-3.1-8b-instant",
            response_format={ "type": "json_object" }
        )
        return json.loads(completion.choices[0].message.content)

if __name__ == "__main__":
    # Initialize using the Secure API Key from environment variables
    evaluator = KTEvaluator(API_KEY)
    
    sample_query = "How is face encoding stored?"
    res = evaluator.engine.generate_response(sample_query)
    
    # Run the self-evaluation
    score = evaluator.evaluate_answer(
        sample_query, 
        res['answer'], 
        f"Source code snippets: {', '.join(res['sources'])}"
    )
    
    print("\n--- 📊 AI Self-Evaluation ---")
    print(json.dumps(score, indent=4))