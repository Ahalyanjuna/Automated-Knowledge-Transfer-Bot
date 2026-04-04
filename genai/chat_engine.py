# import os
# from groq import Groq
# from genai.retriever import KTRetriever
# from genai.rl_logger import RLExperienceLogger
# from genai.rl_agent import RLAgent
# from dotenv import load_dotenv

# load_dotenv()
# API_KEY = os.getenv("GROQ_API_KEY")

# class KTChatEngine:
#     def __init__(self, groq_api_key: str):
#         # 1. Initialize the LLM client
#         self.client = Groq(api_key=groq_api_key)
        
#         # 2. Initialize our Retriever
#         base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#         db_path = os.path.join(base_dir, "chroma_visioncortex")

#         self.retriever = KTRetriever(db_path=db_path)
        
#         # 3. Memory storage for the conversation
#         self.history = []

#         # This loads the brain (DQN) and the feedback database
#         self.rl_agent = RLAgent(model_path="rl_model.pth")
#         self.logger = RLExperienceLogger()

#     def generate_response(self, user_query: str):
#         # 1. Get a larger pool of candidates (Top 10 instead of 3)
#         raw_hits = self.retriever.search(user_query, top_k=10)
        
#         # 2. Let the RL Agent re-score these 10 candidates
#         query_vec = self.retriever.model.encode(user_query)
        
#         scored_hits = []
#         for hit in raw_hits:
#             # Get the embedding for the specific code chunk
#             doc_text = hit['content']
#             doc_vec = self.retriever.model.encode(doc_text)
            
#             # Ask the RL Agent: "How helpful is this?"
#             rl_score = self.rl_agent.get_q_value(query_vec, doc_vec)
#             hit['rl_score'] = rl_score
#             scored_hits.append(hit)

#         # 3. Sort by the RL Score (Highest first) and take top 3
#         # This is where the "Intelligence" happens
#         final_hits = sorted(scored_hits, key=lambda x: x['rl_score'], reverse=True)[:3]
        
#         # 4. Prepare context for the LLM
#         context = ""
#         sources = []
#         for i, hit in enumerate(final_hits, 1):
#             file_name = hit['metadata'].get('source_file', 'Unknown')
#             sources.append(file_name)
#             # We show the RL score in the console for debugging
#             print(f"DEBUG: Selected {file_name} with RL Score: {hit['rl_score']:.4f}")
#             context += f"--- SOURCE {i}: {file_name} ---\n{hit['content']}\n\n"

#         # 5. Call Llama-3 (Same as before)
#         system_prompt = f"You are an expert Software Assistant. Use this context:\n{context}"
#         messages = [{"role": "system", "content": system_prompt}]
#         messages.extend(self.history[-4:]) 
#         messages.append({"role": "user", "content": user_query})

#         chat_completion = self.client.chat.completions.create(
#             messages=messages,
#             model="llama-3.1-8b-instant",
#             temperature=0.2,
#         )

#         answer = chat_completion.choices[0].message.content
        
#         # Save to history for memory
#         self.history.append({"role": "user", "content": user_query})
#         self.history.append({"role": "assistant", "content": answer})

#         return {
#             "answer": answer,
#             "sources": list(set(sources))
#         }

# if __name__ == "__main__":
#     from rl_logger import RLExperienceLogger # Import the logger
    
#     engine = KTChatEngine(API_KEY)
#     logger = RLExperienceLogger() # Initialize logger
    
#     print("--- VisionCortex KT Bot with RL Feedback (Type 'exit' to stop) ---")
    
#     while True:
#         user_input = input("\nYou: ")
#         if user_input.lower() in ["exit", "quit"]:
#             break
            
#         # Get response and the query vector (we'll need to modify the engine to return this)
#         # For now, let's just get the answer
#         result = engine.generate_response(user_input)
#         print(f"\nAI: {result['answer']}")
        
#         # --- NEW: Feedback Collection ---
#         feedback = input("\nWas this helpful? (y/n): ").lower()
#         reward = 1.0 if feedback == 'y' else -1.0
        
#         # Log the experience
#         # We need the query vector from the retriever
#         query_vec = engine.retriever.model.encode(user_input).tolist()
#         chunk_ids = [hit['id'] for hit in engine.retriever.search(user_input)]
        
#         logger.log_experience(user_input, query_vec, chunk_ids, reward)

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
        self.client = Groq(api_key=groq_api_key)

        # Project root
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Caches
        self.retriever_cache = {}
        self.rl_agent_cache = {}
        self.history_store = {}

        # Shared logger
        self.logger = RLExperienceLogger()

    def _get_db_path(self, project_name: str) -> str:
        return os.path.join(self.base_dir, f"chroma_{project_name}")

    def _get_rl_model_path(self, project_name: str) -> str:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), f"rl_model_{project_name}.pth")

    def _get_retriever(self, project_name: str) -> KTRetriever:
        if project_name not in self.retriever_cache:
            db_path = self._get_db_path(project_name)
            if not os.path.exists(db_path):
                raise FileNotFoundError(f"Chroma DB not found for project '{project_name}' at: {db_path}")

            self.retriever_cache[project_name] = KTRetriever(db_path=db_path)

        return self.retriever_cache[project_name]

    def _get_rl_agent(self, project_name: str) -> RLAgent:
        if project_name not in self.rl_agent_cache:
            model_path = self._get_rl_model_path(project_name)
            self.rl_agent_cache[project_name] = RLAgent(model_path=model_path)

        return self.rl_agent_cache[project_name]

    def _normalize_role(self, role: str) -> str:
        if not role:
            return "user"

        role_l = role.strip().lower()

        if "developer" in role_l or "engineer" in role_l:
            return "developer"
        if "project manager" in role_l or "manager" in role_l:
            return "project manager"
        if "data scientist" in role_l or "data science" in role_l or "ml" in role_l:
            return "data scientist"

        return role_l

    def _get_role_template(self, role: str) -> str:
        normalized_role = self._normalize_role(role)

        templates = {
            "developer": """
You are answering for a Developer.
Focus on:
- code flow
- classes, functions, and modules
- APIs and integration points
- debugging causes and fixes
- implementation details
- step-by-step logic when useful

Keep the answer technical and implementation-oriented.
""",
            "project manager": """
You are answering for a Project Manager.
Focus on:
- feature purpose
- workflow
- dependencies
- risks and blockers
- high-level implementation summary
- business and delivery impact

Avoid unnecessary low-level code details unless essential.
""",
            "data scientist": """
You are answering for a Data Scientist.
Focus on:
- data flow
- preprocessing
- embeddings/vector DB usage
- retrieval logic
- ranking logic
- model behavior and evaluation
- pipeline-level reasoning

Highlight how the data moves through the system.
"""
        }

        return templates.get(
            normalized_role,
            f"""
You are answering for a user whose role is: {role}.
Adjust explanation depth, focus, and wording to suit this role.
"""
        )

    def generate_response(self, user_query: str, project_name: str, user_role: str):
        retriever = self._get_retriever(project_name)
        rl_agent = self._get_rl_agent(project_name)

        # 1. Retrieve more candidates
        raw_hits = retriever.search(user_query, top_k=10)

        # 2. Re-rank with RL
        query_vec = retriever.model.encode(user_query)

        scored_hits = []
        for hit in raw_hits:
            doc_text = hit["content"]
            doc_vec = retriever.model.encode(doc_text)
            rl_score = rl_agent.get_q_value(query_vec, doc_vec)
            hit["rl_score"] = rl_score
            scored_hits.append(hit)

        # 3. Final top results
        final_hits = sorted(scored_hits, key=lambda x: x["rl_score"], reverse=True)[:3]

        # 4. Build context
        context = ""
        sources = []

        for i, hit in enumerate(final_hits, 1):
            file_name = hit["metadata"].get("source_file", "Unknown")
            sources.append(file_name)
            print(f"DEBUG: [{project_name}] Selected {file_name} with RL Score: {hit['rl_score']:.4f}")
            context += f"--- SOURCE {i}: {file_name} ---\n{hit['content']}\n\n"

        role_template = self._get_role_template(user_role)

        system_prompt = f"""
You are an expert Software Assistant.

Selected project: {project_name}
The person asking is of role: {user_role}

{role_template}

Rules:
- Answer ONLY from the supplied project context.
- Be specific to the selected project only.
- Do not mix knowledge from other projects.
- If the answer is not available in the retrieved context, say so clearly.
- Mention important file/module names where useful.
- If the role is non-technical, adapt the explanation accordingly.

Context:
{context}
"""

        history_key = f"{project_name}::{self._normalize_role(user_role)}"
        if history_key not in self.history_store:
            self.history_store[history_key] = []

        history = self.history_store[history_key]

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history[-4:])
        messages.append({
            "role": "user",
            "content": f"The person is of role: {user_role}\nProject: {project_name}\nQuestion: {user_query}"
        })

        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant",
            temperature=0.2,
        )

        answer = chat_completion.choices[0].message.content

        history.append({"role": "user", "content": user_query})
        history.append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            "sources": list(set(sources))
        }


if __name__ == "__main__":
    engine = KTChatEngine(API_KEY)

    print("--- Project-aware KT Bot with RL Feedback (Type 'exit' to stop) ---")
    project_name = input("Enter project name (folder should be chroma_<project>): ").strip()
    user_role = input("Enter role: ").strip() or "Developer"

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        result = engine.generate_response(
            user_query=user_input,
            project_name=project_name,
            user_role=user_role
        )
        print(f"\nAI: {result['answer']}")

        feedback = input("\nWas this helpful? (y/n): ").lower()
        reward = 1.0 if feedback == "y" else -1.0

        retriever = engine._get_retriever(project_name)
        query_vec = retriever.model.encode(user_input).tolist()
        chunk_ids = [hit["id"] for hit in retriever.search(user_input)]

        engine.logger.log_experience(project_name, user_input, query_vec, chunk_ids, reward)