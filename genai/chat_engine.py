import os
from groq import Groq
from genai.retriever import KTRetriever
from genai.rl_logger import RLExperienceLogger
from genai.rl_agent import RLAgent
from genai.query_translator import detect_query_language, translate_to_english, translate_from_english
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
- code flow, classes, functions, and modules
- APIs and integration points
- debugging causes and fixes
- implementation details
- step-by-step logic when useful
Keep the answer technical and implementation-oriented.
""",
            "project manager": """
You are answering for a Project Manager.
Focus on:
- feature purpose and workflow
- dependencies, risks and blockers
- high-level implementation summary
- business and delivery impact
Avoid unnecessary low-level code details unless essential.
""",
            "data scientist": """
You are answering for a Data Scientist.
Focus on:
- data flow and preprocessing
- embeddings/vector DB usage
- retrieval and ranking logic
- model behavior and evaluation
- pipeline-level reasoning
Highlight how the data moves through the system.
"""
        }
        return templates.get(
            normalized_role,
            f"You are answering for a user whose role is: {role}. Adjust explanation depth to suit this role."
        )

    def generate_response(
        self,
        user_query: str,
        project_name: str,
        user_role: str,
        respond_in_original_lang: bool = True,
    ) -> dict:
        """
        Generate a response with full multilingual support.

        Returns a dict with:
          answer             - response in user's original language (default)
          answer_english     - response in English (always present)
          original_query     - raw query as typed
          translated_query   - English translation of query (None if already English)
          detected_lang      - ISO 639-1 code e.g. "ta"
          detected_lang_name - Human readable e.g. "Tamil"
          is_translated      - bool: True if query was not in English
          sources            - list of source file names
        """

        # ── Step 1: Detect & translate query ──────────────────────────────
        detected_lang, detected_lang_name, lang_conf = detect_query_language(user_query)
        is_translated = detected_lang != "en" and lang_conf >= 0.6

        if is_translated:
            english_query = translate_to_english(user_query, detected_lang)
            if not english_query:
                # Translation failed — use original as-is
                english_query = user_query
                is_translated = False
        else:
            english_query = user_query

        print(f"DEBUG: [{project_name}] lang={detected_lang} ({lang_conf:.2f}), translated={is_translated}")
        print(f"DEBUG: original='{user_query[:80]}' → english='{english_query[:80]}'")

        # ── Step 2: Retrieve using English query ───────────────────────────
        retriever = self._get_retriever(project_name)
        rl_agent = self._get_rl_agent(project_name)

        raw_hits = retriever.search(english_query, top_k=10)

        query_vec = retriever.model.encode(english_query)

        scored_hits = []
        for hit in raw_hits:
            doc_text = hit["content"]
            doc_vec = retriever.model.encode(doc_text)
            rl_score = rl_agent.get_q_value(query_vec, doc_vec)
            hit["rl_score"] = rl_score
            scored_hits.append(hit)

        final_hits = sorted(scored_hits, key=lambda x: x["rl_score"], reverse=True)[:3]

        # ── Step 3: Build context & call LLM ──────────────────────────────
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
- Always respond in ENGLISH regardless of the question language.

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
            "content": f"Role: {user_role}\nProject: {project_name}\nQuestion: {english_query}"
        })

        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant",
            temperature=0.2,
        )

        answer_english = chat_completion.choices[0].message.content

        # ── Step 4: Translate response back if needed ──────────────────────
        answer_original = answer_english  # default
        if is_translated and respond_in_original_lang:
            translated_back = translate_from_english(answer_english, detected_lang)
            if translated_back:
                answer_original = translated_back

        # ── Step 5: Update history (always store English) ─────────────────
        history.append({"role": "user", "content": english_query})
        history.append({"role": "assistant", "content": answer_english})

        return {
            "answer": answer_original,          # in user's language
            "answer_english": answer_english,   # always English
            "original_query": user_query,
            "translated_query": english_query if is_translated else None,
            "detected_lang": detected_lang,
            "detected_lang_name": detected_lang_name,
            "is_translated": is_translated,
            "sources": list(set(sources)),
        }


if __name__ == "__main__":
    engine = KTChatEngine(API_KEY)

    print("--- Project-aware KT Bot with RL Feedback + Multilingual Support ---")
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

        if result["is_translated"]:
            print(f"\n🌐 Detected: {result['detected_lang_name']}")
            print(f"📝 Translated query: {result['translated_query']}")

        print(f"\nAI ({result['detected_lang_name']}): {result['answer']}")
        print(f"\nAI (English): {result['answer_english']}")

        feedback = input("\nWas this helpful? (y/n): ").lower()
        reward = 1.0 if feedback == "y" else -1.0

        retriever = engine._get_retriever(project_name)
        query_vec = retriever.model.encode(result["translated_query"] or user_input).tolist()
        chunk_ids = [hit["id"] for hit in retriever.search(result["translated_query"] or user_input)]
        engine.logger.log_experience(project_name, user_input, query_vec, chunk_ids, reward)