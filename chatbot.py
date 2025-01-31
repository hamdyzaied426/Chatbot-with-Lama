import streamlit as st
import requests
import faiss
import sqlite3
import pickle
import datetime
from collections import Counter
from sentence_transformers import SentenceTransformer

# Database configuration
def get_db_connection():
    conn = sqlite3.connect('chat_history.db')
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def initialize_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT UNIQUE,
            embedding BLOB,
            response TEXT,
            usage_count INTEGER DEFAULT 1
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            title TEXT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT,
            role TEXT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chat_id) REFERENCES chats (id) ON DELETE CASCADE
        )
    """)
    
    conn.commit()
    conn.close()

# Session state initialization
def initialize_session_state():
    defaults = {
        'current_chat_id': None,
        'index': faiss.IndexFlatIP(384),
        'cache': {},
        'embedder': SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
        'ollama_status': {}
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Database operations
def store_query(query, embedding, response):
    conn = get_db_connection()
    try:
        conn.execute("""
            INSERT INTO queries (query, embedding, response)
            VALUES (?, ?, ?)
            ON CONFLICT(query) DO UPDATE SET usage_count = usage_count + 1
        """, (query, pickle.dumps(embedding), response))
        conn.commit()
    except Exception as e:
        st.error(f"Database error: {str(e)}")
    finally:
        conn.close()

def get_cached_response(query):
    try:
        if st.session_state.index.ntotal == 0:
            return None
            
        query_vector = st.session_state.embedder.encode(query).reshape(1, -1).astype('float32')
        similarities, indices = st.session_state.index.search(query_vector, k=5)
        
        valid_responses = [
            st.session_state.cache.get(indices[0][i]) 
            for i in range(len(similarities[0])) 
            if similarities[0][i] > 0.60
        ]
        
        if not valid_responses:
            return None
            
        most_common = Counter(valid_responses).most_common(1)[0][0]
        store_query(query, query_vector, most_common)
        return most_common
        
    except Exception as e:
        st.error(f"Cache error: {str(e)}")
        return None

# Chat management
def create_chat():
    chat_id = datetime.datetime.now().isoformat()
    conn = get_db_connection()
    conn.execute("INSERT INTO chats (id, title) VALUES (?, ?)", (chat_id, "New Chat"))
    conn.commit()
    conn.close()
    return chat_id

def delete_chat(chat_id):
    conn = get_db_connection()
    conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    conn.commit()
    conn.close()

# Enhanced Ollama connection handling
def check_ollama_connection():
    """Check Ollama service status with detailed diagnostics"""
    try:
        # Basic connectivity check
        ping_response = requests.get("http://localhost:11434/", timeout=5)
        if ping_response.status_code != 200:
            return {
                "connected": False,
                "error": f"Unexpected status code: {ping_response.status_code}"
            }

        # Model availability check
        models_response = requests.get("http://localhost:11434/api/tags", timeout=5)
        models_data = models_response.json()
        models = [m["name"] for m in models_data.get("models", [])]
        
        return {
            "connected": True,
            "model_available": "llama3.2" in models,
            "models": models,
            "model_details": models_data
        }
        
    except requests.exceptions.ConnectionError:
        return {"connected": False, "error": "Connection refused - Is Ollama running?"}
    except Exception as e:
        return {"connected": False, "error": str(e)}

def generate_response(prompt, history, temperature=0.7):
    """Enhanced response generation with detailed error handling"""
    connection_status = check_ollama_connection()
    st.session_state.ollama_status = connection_status
    
    if not connection_status["connected"]:
        error_msg = f"Ollama Connection Error: {connection_status.get('error', 'Unknown error')}"
        st.toast("🔴 Connection Failed", icon="❌")
        return error_msg, False
        
    if not connection_status.get("model_available", False):
        st.toast("🟠 Model Missing", icon="⚠️")
        return (
            f"Model 'llama3.2' not found. Available models: {', '.join(connection_status['models'])}\n"
            "Install with: `ollama pull llama3.2`", False
        )

    try:
        formatted_history = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in history]
        )
        full_prompt = f"{formatted_history}\nuser: {prompt}\nassistant:"
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": full_prompt,
                "stream": False,
                "options": {"temperature": temperature}
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()["response"], True
        
    except requests.exceptions.RequestException as e:
        st.toast("🔴 Generation Failed", icon="❌")
        return f"API Error: {str(e)}", False
    except Exception as e:
        return f"Unexpected Error: {str(e)}", False

# UI components
def sidebar():
    with st.sidebar:
        st.header("Chat Management")
        
        if st.button("➕ New Chat"):
            st.session_state.current_chat_id = create_chat()
            st.rerun()
        
        st.divider()
        st.header("Model Settings")
        temperature = st.slider("Temperature", 0.0, 2.0, 0.3, 0.1)
        
        st.divider()
        st.header("Chat History")
        
        conn = get_db_connection()
        chats = conn.execute("SELECT * FROM chats ORDER BY created_at DESC").fetchall()
        
        for chat in chats:
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(chat["title"], key=f"btn_{chat['id']}"):
                    st.session_state.current_chat_id = chat["id"]
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{chat['id']}"):
                    delete_chat(chat["id"])
                    st.rerun()
        
        conn.close()

def main_interface():
    st.title("Chatbot with Ollama")
    
    # Connection status display
    connection_status = st.session_state.ollama_status
    status_color = "green" if connection_status.get("connected", False) else "red"
    status_text = "● Connected" if connection_status.get("connected", False) else "● Disconnected"
    
    st.markdown(f"""
    **Ollama Status**: <span style='color:{status_color}'>{status_text}</span>  
    {f"**Available Models**: {', '.join(connection_status.get('models', []))}" if connection_status.get("connected", False) else ""}
    """, unsafe_allow_html=True)

    # Show troubleshooting guide if needed
    if not connection_status.get("connected", True):
        with st.expander("🔍 Troubleshooting Guide", expanded=True):
            st.markdown("""
            ### Connection Issues Detected
            1. **Start Ollama Service**  
               Run in a terminal:  
               ```bash
               ollama serve
               ```
            2. **Verify Installation**  
               Check version:  
               ```bash
               ollama --version
               ```
            3. **Check Port Access**  
               Ensure port 11434 is allowed through your firewall
            4. **Validate Model Installation**  
               List installed models:  
               ```bash
               ollama list
               ```
            """)

    # Load current chat
    messages = []
    if st.session_state.current_chat_id:
        conn = get_db_connection()
        messages = conn.execute(
            "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY timestamp",
            (st.session_state.current_chat_id,)
        ).fetchall()
        conn.close()
    
    # Display messages
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Handle input
    if prompt := st.chat_input("Type your message..."):
        if not st.session_state.current_chat_id:
            st.session_state.current_chat_id = create_chat()
        
        # Save user message
        conn = get_db_connection()
        conn.execute(
            "INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)",
            (st.session_state.current_chat_id, "user", prompt)
        )
        
        # Get response
        cached_response = get_cached_response(prompt)
        if cached_response:
            response = cached_response
        else:
            response, success = generate_response(prompt, messages)
            if success:
                embedding = st.session_state.embedder.encode(prompt).reshape(1, -1).astype('float32')
                st.session_state.index.add(embedding)
                st.session_state.cache[st.session_state.index.ntotal - 1] = response
                store_query(prompt, embedding, response)
        
        # Save assistant response
        conn.execute(
            "INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)",
            (st.session_state.current_chat_id, "assistant", response)
        )
        conn.commit()
        conn.close()
        
        st.rerun()

if __name__ == "__main__":
    initialize_db()
    initialize_session_state()
    check_ollama_connection()  # Initial connection check
    sidebar()
    main_interface()
