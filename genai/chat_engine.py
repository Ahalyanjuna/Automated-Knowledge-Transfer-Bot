import os
from groq import Groq
from genai.retriever import KTRetriever
from genai.rl_logger import RLExperienceLogger
from genai.rl_agent import RLAgent
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

class KTChatEngine:
    def __init__(self, groq_api_key: str):
        # 1. Initialize the LLM client
        self.client = Groq(api_key=groq_api_key)
        
        # 2. Initialize our Retriever
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_path = os.path.join(base_dir, "chroma_visioncortex")

        self.retriever = KTRetriever(db_path=db_path)
        
        # 3. Memory storage for the conversation
        self.history = []

        # This loads the brain (DQN) and the feedback database
        self.rl_agent = RLAgent(model_path="rl_model.pth")
        self.logger = RLExperienceLogger()

    def generate_response(self, user_query: str):
        # 1. Get a larger pool of candidates (Top 10 instead of 3)
        raw_hits = self.retriever.search(user_query, top_k=10)
        
        # 2. Let the RL Agent re-score these 10 candidates
        query_vec = self.retriever.model.encode(user_query)
        
        scored_hits = []
        for hit in raw_hits:
            # Get the embedding for the specific code chunk
            doc_text = hit['content']
            doc_vec = self.retriever.model.encode(doc_text)
            
            # Ask the RL Agent: "How helpful is this?"
            rl_score = self.rl_agent.get_q_value(query_vec, doc_vec)
            hit['rl_score'] = rl_score
            scored_hits.append(hit)

        # 3. Sort by the RL Score (Highest first) and take top 3
        # This is where the "Intelligence" happens
        final_hits = sorted(scored_hits, key=lambda x: x['rl_score'], reverse=True)[:3]
        
        # 4. Prepare context for the LLM
        context = ""
        sources = []
        for i, hit in enumerate(final_hits, 1):
            file_name = hit['metadata'].get('source_file', 'Unknown')
            sources.append(file_name)
            # We show the RL score in the console for debugging
            print(f"DEBUG: Selected {file_name} with RL Score: {hit['rl_score']:.4f}")
            context += f"--- SOURCE {i}: {file_name} ---\n{hit['content']}\n\n"

        # 5. Call Llama-3 (Same as before)
        system_prompt = f"You are an expert Software Assistant. Use this context:\n{context}"
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.history[-4:]) 
        messages.append({"role": "user", "content": user_query})

        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant",
            temperature=0.2,
        )

        answer = chat_completion.choices[0].message.content
        
        # Save to history for memory
        self.history.append({"role": "user", "content": user_query})
        self.history.append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            "sources": list(set(sources))
        }

if __name__ == "__main__":
    from rl_logger import RLExperienceLogger # Import the logger
    
    engine = KTChatEngine(API_KEY)
    logger = RLExperienceLogger() # Initialize logger
    
    print("--- VisionCortex KT Bot with RL Feedback (Type 'exit' to stop) ---")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        # Get response and the query vector (we'll need to modify the engine to return this)
        # For now, let's just get the answer
        result = engine.generate_response(user_input)
        print(f"\nAI: {result['answer']}")
        
        # --- NEW: Feedback Collection ---
        feedback = input("\nWas this helpful? (y/n): ").lower()
        reward = 1.0 if feedback == 'y' else -1.0
        
        # Log the experience
        # We need the query vector from the retriever
        query_vec = engine.retriever.model.encode(user_input).tolist()
        chunk_ids = [hit['id'] for hit in engine.retriever.search(user_input)]
        
        logger.log_experience(user_input, query_vec, chunk_ids, reward)