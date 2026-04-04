import os
import pandas as pd
from dotenv import load_dotenv
from genai.chat_engine import KTChatEngine

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")


def run_ragas_lite_test(
    project_name: str = "visioncortex",
    user_role: str = "Developer",
    test_queries=None
):
    engine = KTChatEngine(API_KEY)

    if test_queries is None or len(test_queries) == 0:
        test_queries = [
            "What database is used for storing faces?",
            "How do I register a new face?",
            "What happens if the 'logs' directory is missing?"
        ]

    results = []

    print(f"🚀 Starting RAGAS-Lite Evaluation for project='{project_name}', role='{user_role}'...")

    for query in test_queries:
        print(f"Testing: {query}")

        response = engine.generate_response(
            user_query=query,
            project_name=project_name,
            user_role=user_role
        )

        results.append({
            "Project": project_name,
            "Role": user_role,
            "Question": query,
            "Answer": response["answer"][:200] + ("..." if len(response["answer"]) > 200 else ""),
            "Sources": ", ".join(response["sources"]),
            "Source_Count": len(response["sources"])
        })

    df = pd.DataFrame(results)

    if not df.empty and df["Source_Count"].min() > 0:
        print("\n✅ PASSED: Every answer found at least one source code file.")
    else:
        print("\n⚠️ WARNING: Some answers had no sources.")

    return df


if __name__ == "__main__":
    df = run_ragas_lite_test(
        project_name="visioncortex",
        user_role="Developer"
    )
    print(df.to_string(index=False))