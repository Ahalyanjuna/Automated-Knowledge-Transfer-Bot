import os
import requests
import zipfile
import io
import shutil
from genai.chat_engine import KTChatEngine
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")


class AutoDocEngine:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("❌ API Key missing! Check your .env file.")
        self.engine = KTChatEngine(api_key)

    def generate_full_manual(self, project_path, audience_role="user"):
        """Analyzes local files and generates documentation."""
        files_to_analyze = []

        for root, dirs, files in os.walk(project_path):
            if any(skip in root for skip in ["venv", "chroma", "__pycache__", ".git", "node_modules"]):
                continue
            for file in files:
                if file.endswith((".py", ".js", ".java", ".cpp")):
                    files_to_analyze.append(os.path.join(root, file))

        if not files_to_analyze:
            return False, f"❌ No code files found in {project_path}", None

        print(f"📄 Found {len(files_to_analyze)} files. Synthesizing documentation...")

        audience_instruction = (
            "Write for an admin/technical audience. Include implementation details, architecture, and internal logic."
            if audience_role == "admin"
            else "Write for a general/user audience. Keep explanations clear, useful, and lighter on low-level implementation."
        )

        full_manual = "# 🧠 Automated Technical Manual\n\n"

        for file_path in files_to_analyze:
            file_name = os.path.basename(file_path)
            print(f"--- 🧠 Analyzing: {file_name} ---")

            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()[:4000]

                prompt = f"""
Analyze this source code from '{file_name}'.

Audience: {audience_role}
Instruction: {audience_instruction}

Code:
{content}

Provide:
1. Purpose of this file
2. Key functions/classes/logic
3. How it fits into the project
4. Important notes for this audience
""".strip()

                completion = self.engine.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.1-8b-instant",
                )

                full_manual += f"## 📄 File: {file_name}\n"
                full_manual += f"{completion.choices[0].message.content}\n\n---\n\n"

            except Exception as e:
                print(f"⚠️ Skip {file_name}: {e}")

        
        return True, "✨ Report generated successfully!", full_manual
    
    def generate_from_github(self, github_url,audience_role="user"):
        """Downloads a GitHub repo and documents it."""
        print(f"🌐 Fetching repository: {github_url}")

        zip_url = github_url.rstrip("/") + "/archive/refs/heads/main.zip"

        try:
            response = requests.get(zip_url, timeout=30)
            if response.status_code != 200:
                zip_url = github_url.rstrip("/") + "/archive/refs/heads/master.zip"
                response = requests.get(zip_url, timeout=30)

            if response.status_code == 200:
                extract_path = "temp_repo"

                if os.path.exists(extract_path):
                    shutil.rmtree(extract_path)

                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    z.extractall(extract_path)

                status, msg, content = self.generate_full_manual(
                    extract_path,
                    audience_role=audience_role
                )

                shutil.rmtree(extract_path)
                return status, msg,content

            return False, "❌ Could not find main or master branch. Is the repo public?", None

        except Exception as e:
            return False, f"❌ Error downloading GitHub repo: {e}", None


if __name__ == "__main__":
    autodoc = AutoDocEngine(API_KEY)

    print("--- 🤖 KT-Bot AutoDoc Engine ---")
    path_or_url = input("Enter Local Path OR GitHub Link: ").strip()

    if "github.com" in path_or_url:
        ok, msg, path = autodoc.generate_from_github(path_or_url)
    else:
        ok, msg, path = autodoc.generate_full_manual(path_or_url)

    print(msg)