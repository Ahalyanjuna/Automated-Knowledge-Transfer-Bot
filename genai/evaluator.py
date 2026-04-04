import json
import os
from dotenv import load_dotenv
from genai.chat_engine import KTChatEngine

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")


class KTEvaluator:
    def __init__(self, api_key, project_name: str, user_role: str = "Developer"):
        if not api_key:
            raise ValueError("❌ API Key not found. Please check your .env file.")

        self.engine = KTChatEngine(api_key)
        self.project_name = project_name
        self.user_role = user_role

    def evaluate_answer(self, query, response, context):
        eval_prompt = f"""
Analyze the following RAG interaction and provide a score from 1 to 5
for:
- Faithfulness (accuracy to retrieved code/project context)
- Relevance (usefulness for the user’s question)

PROJECT: {self.project_name}
USER ROLE: {self.user_role}
QUERY: {query}
CONTEXT PROVIDED: {context}
AI RESPONSE: {response}

Return ONLY a JSON object in this format:
{{"faithfulness": score, "relevance": score, "reason": "short explanation"}}
""".strip()

        completion = self.engine.client.chat.completions.create(
            messages=[{"role": "user", "content": eval_prompt}],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)

    def ask_and_evaluate(self, query):
        res = self.engine.generate_response(
            user_query=query,
            project_name=self.project_name,
            user_role=self.user_role
        )

        score = self.evaluate_answer(
            query=query,
            response=res["answer"],
            context=f"Source code snippets: {', '.join(res['sources'])}"
        )

        return {
            "query": query,
            "answer": res["answer"],
            "sources": res["sources"],
            "evaluation": score
        }


if __name__ == "__main__":
    evaluator = KTEvaluator(
        api_key=API_KEY,
        project_name="visioncortex",
        user_role="Developer"
    )

    sample_query = "How is face encoding stored?"
    result = evaluator.ask_and_evaluate(sample_query)

    print("\n--- 🤖 AI Answer ---")
    print(result["answer"])

    print("\n--- 📂 Sources ---")
    print(", ".join(result["sources"]))

    print("\n--- 📊 AI Self-Evaluation ---")
    print(json.dumps(result["evaluation"], indent=4))