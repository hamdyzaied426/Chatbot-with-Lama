# 🗨️ Chatbot with Ollama, FAISS, and Streamlit

![Chatbot Architecture Diagram](https://via.placeholder.com/800x400.png?text=Architecture+Diagram) <!-- Replace with actual diagram -->

A conversational AI chatbot with persistent memory and semantic caching, powered by Ollama's LLMs.

## 🌟 Features
| Feature                | Technology Used       | Benefit                              |
|------------------------|-----------------------|--------------------------------------|
| AI Responses           | Ollama (LLaMA3.2)     | Local, privacy-first AI processing   |
| Chat History           | SQLite                | Persistent conversation storage      |
| Response Cache         | FAISS                 | Instant answer retrieval             |
| UI Interface           | Streamlit             | User-friendly web interface          |
| Semantic Search        | Sentence Transformers | Context-aware response matching      |

## 📦 Installation
```bash
# Clone repository
git clone https://github.com/yourusername/chatbot.git
cd chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Ollama (in separate terminal)
ollama serve

🚀 Usage
streamlit run app.py
🧠 Code Structure
.
├── app.py                 # Main Streamlit application
├── chat_history.db        # SQLite database (auto-created)
├── requirements.txt       # Dependency list
└── README.md              # This documentation

🔧 Components
1.User Interface (Streamlit)

  Chat input panel
  
  Sidebar with chat history

  Temperature control
  
2.AI Backend (Ollama)

  Local LLM inference
  
  Context-aware generation
  
3.Memory System

  SQLite for conversation history

  FAISS vector cache

4.Semantic Search

  Sentence Transformers
  
  Similarity threshold: 0.6

