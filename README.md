# Automated-Knowledge-Transfer-Bot

python nlp_pipeline.py --input ../chunks.json --output ../output/nlp_chunks.json

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
2. Configuration
Open app.py and genai/chat_engine.py and ensure your Groq API Key is set:

API_KEY = "your-api-key"

Step 1: Ingest the Codebase
python genai/ingest.py
Output: Creates a folder named chroma_visioncortex.

Step 2: Initialize the RL Brain
Before the bot can learn, it needs a "blank slate" neural network model.
python genai/rl_agent.py
Output: Generates genai/rl_model.pth.

Step 3: Launch the Dashboard
Now, start the interactive Streamlit UI to chat with your code.
python -m streamlit run app.py --server.fileWatcherType none
Usage: Ask questions about the code and provide 👍/👎 feedback to generate training data.

Step 4: Train the Intelligence
After providing feedback in the UI, you can "teach" the bot to be more accurate. You can do this via the Sidebar button in the UI or manually:

Bash
python genai/train_rl.py
Output: Updates rl_model.pth based on your feedback stored in rl_feedback.db.

📂 Project Structure
app.py: The main Streamlit dashboard.

genai/ingest.py: Handles code scraping and vector embedding.

genai/retriever.py: The search engine (ChromaDB + Semantic Search).

genai/rl_agent.py: The Deep Q-Network model for re-ranking results.

genai/rl_logger.py: Saves user feedback to a local SQLite database.

genai/train_rl.py: The training loop that improves the bot's accuracy.

genai/auto_doc.py: Automatically generates a full TECHNICAL_GUIDE.md for the repo.

📊 Evaluation (RAGAS)
To test the performance of the retrieval system against "Golden Questions," run the evaluation script:

export PYTHONPATH=$PYTHONPATH:.
python genai/ragas_test.py

pip install python-dotenv