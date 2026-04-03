import os
from genai.chat_engine import KTChatEngine
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

class AutoDocEngine:
    def __init__(self, api_key):
        self.engine = KTChatEngine(api_key)

    def generate_full_manual(self, project_path):
        files_to_analyze = []
        # Walk through the project to find code files
        for root, dirs, files in os.walk(project_path):
            if "venv" in root or "chroma" in root or "__pycache__" in root:
                continue
            for file in files:
                if file.endswith((".py", ".js")):
                    files_to_analyze.append(os.path.join(root, file))

        print(f"📄 Found {len(files_to_analyze)} files. Starting auto-documentation...")
        
        full_manual = "# VisionCortex Technical Manual\n\n"
        
        for file_path in files_to_analyze:
            file_name = os.path.basename(file_path)
            print(f"--- Processing {file_name} ---")
            
            # Ask the AI to summarize this specific file
            query = f"Explain the purpose, main functions, and logic of the file: {file_name}"
            result = self.engine.generate_response(query)
            
            full_manual += f"## File: {file_name}\n"
            full_manual += f"{result['answer']}\n\n"
            full_manual += "---\n\n"

        # Save to file
        with open("TECHNICAL_GUIDE.md", "w") as f:
            f.write(full_manual)
        
        return "✨ TECHNICAL_GUIDE.md has been generated successfully!"

if __name__ == "__main__":
    autodoc = AutoDocEngine(API_KEY)
    # Point it to your root directory
    status = autodoc.generate_full_manual("../") 
    print(status)