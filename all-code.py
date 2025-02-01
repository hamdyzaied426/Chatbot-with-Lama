import streamlit as st
import requests
import faiss
import sqlite3
import pickle
import numpy as np
import datetime
from collections import Counter
from sentence_transformers import SentenceTransformer

# Initialize database connection with foreign keys enabled
def get_db_connection():
    conn = sqlite3.connect('chat_history.db')
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

# Initialize database tables
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

# Initialize or load session state
def initialize_session_state():
    if 'current_chat_id' not in st.session_state:
        st.session_state.current_chat_id = None
        
    if 'index' not in st.session_state:
        # Initialize FAISS index with persisted data
        st.session_state.index = faiss.IndexFlatIP(384)
        conn = get_db_connection()
        try:
            stored_queries = conn.execute('SELECT embedding FROM queries').fetchall()
            for q in stored_queries:
                embedding = pickle.loads(q['embedding'])
                st.session_state.index.add(embedding)
        except Exception as e:
            st.error(f"Error loading embeddings: {e}")
        finally:
            conn.close()
        
    if 'cache' not in st.session_state:
        # Initialize cache with persisted data
        st.session_state.cache = {}
        conn = get_db_connection()
        try:
            cached_responses = conn.execute('SELECT query, response FROM queries').fetchall()
            for idx, row in enumerate(cached_responses):
                st.session_state.cache[idx] = row['response']
        except Exception as e:
            st.error(f"Error loading cache: {e}")
        finally:
            conn.close()
        
    if 'embedder' not in st.session_state:
        st.session_state.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Database functions
def get_embedding(text):
    return st.session_state.embedder.encode(text).reshape(1, -1).astype('float32')

def store_query_in_db(query, embedding, response):
    conn = get_db_connection()
    try:
        # Update or insert query with incrementing usage count
        conn.execute("""
            INSERT INTO queries (query, embedding, response)
            VALUES (?, ?, ?)
            ON CONFLICT(query) DO UPDATE SET 
                response = excluded.response,
                usage_count = usage_count + 1
        """, (query, pickle.dumps(embedding), response))
        conn.commit()
        
        # Add to current session's index and cache if new entry
        if conn.total_changes > 0:
            st.session_state.index.add(embedding)
            new_index = st.session_state.index.ntotal - 1
            st.session_state.cache[new_index] = response
    except Exception as e:
        st.error(f"Error storing query: {e}")
    finally:
        conn.close()

def get_cached_response(query):
    try:
        query_vector = get_embedding(query)
        
        # Search in both database and current session's index
        if st.session_state.index.ntotal > 0:
            similarities, indices = st.session_state.index.search(query_vector, k=5)
            valid_responses = [
                st.session_state.cache.get(indices[0][i]) 
                for i in range(len(similarities[0])) 
                if similarities[0][i] > 0.75
            ]
            if valid_responses:
                most_common_response = Counter(valid_responses).most_common(1)[0][0]
                return most_common_response
                
        # Fallback to database search
        conn = get_db_connection()
        all_queries = conn.execute('SELECT query, embedding, response FROM queries').fetchall()
        for q in all_queries:
            db_embedding = pickle.loads(q['embedding'])
            similarity = np.dot(query_vector, db_embedding.T)[0][0]
            if similarity > 0.60:
                # Add to current session's index and cache
                st.session_state.index.add(db_embedding)
                new_index = st.session_state.index.ntotal - 1
                st.session_state.cache[new_index] = q['response']
                return q['response']
        
        return None
    except Exception as e:
        st.error(f"Error in get_cached_response: {e}")
        return None

# Chat history management (unchanged from previous version)
def load_chat_history():
    conn = get_db_connection()
    chats = conn.execute('SELECT * FROM chats ORDER BY created_at DESC').fetchall()
    messages = {}
    
    for chat in chats:
        chat_messages = conn.execute(
            'SELECT role, content FROM messages WHERE chat_id = ? ORDER BY timestamp',
            (chat['id'],)
        ).fetchall()
        messages[chat['id']] = [dict(msg) for msg in chat_messages]
    
    conn.close()
    return {chat['id']: {'title': chat['title'], 'messages': messages[chat['id']]} for chat in chats}

def save_message(chat_id, role, content):
    conn = get_db_connection()
    conn.execute(
        'INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)',
        (chat_id, role, content)
    )
    conn.commit()
    conn.close()

def create_new_chat():
    chat_id = datetime.datetime.now().isoformat()
    title = "New Chat"
    conn = get_db_connection()
    conn.execute(
        'INSERT INTO chats (id, title) VALUES (?, ?)',
        (chat_id, title)
    )
    conn.commit()
    conn.close()
    return chat_id

def delete_chat(chat_id):
    conn = get_db_connection()
    conn.execute('DELETE FROM chats WHERE id = ?', (chat_id,))
    conn.commit()
    conn.close()
    
    if st.session_state.current_chat_id == chat_id:
        st.session_state.current_chat_id = None

def delete_all_chats():
    conn = get_db_connection()
    conn.execute('DELETE FROM chats')
    conn.commit()
    conn.close()
    st.session_state.current_chat_id = None

def update_chat_title(chat_id, title):
    conn = get_db_connection()
    conn.execute(
        'UPDATE chats SET title = ? WHERE id = ?',
        (title, chat_id)
    )
    conn.commit()
    conn.close()

# AI generation with history
def ollama_generate(prompt, history, temperature=0.9):
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
            }
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        st.error(f"🚨 Error connecting to Ollama: {e}")
        return "I'm having trouble connecting to the AI model."

# UI Components (unchanged from previous version)
def chat_history_sidebar():
    with st.sidebar:
        st.header("Chat History")
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state.current_chat_id = create_new_chat()
        
        st.divider()
        st.header("Model Settings")
        temperature = st.slider("Model Temperature", 0.0, 2.0, 0.3, step=0.1)
        
        st.divider()
        st.header("Previous Chats")
        
        chat_history = load_chat_history()
        today = datetime.date.today()
        
        for chat_id in sorted(chat_history.keys(), reverse=True):
            chat = chat_history[chat_id]
            dt = datetime.datetime.fromisoformat(chat_id).date()
            
            if dt == today:
                date_label = "Today"
            elif dt == today - datetime.timedelta(days=1):
                date_label = "Yesterday"
            else:
                date_label = dt.strftime("%B %d, %Y")
            
            col1, col2 = st.columns([4, 1])
            with col1:
                preview = chat['title'] or chat['messages'][0]['content'][:20] + "..."
                btn = st.button(
                    f"{preview} ({date_label})", 
                    key=chat_id, 
                    use_container_width=True
                )
            with col2:
                if st.button("🗑️", key=f"del_{chat_id}"):
                    delete_chat(chat_id)
                    st.rerun()
            
            if btn:
                st.session_state.current_chat_id = chat_id

        st.divider()
        if st.button("❌ Delete All Chats", use_container_width=True):
            delete_all_chats()
            st.rerun()

        return temperature

def main_chat_interface(temperature):
    st.title("Chatbot with Ollama")
    
    # Load current chat messages
    current_chat_messages = []
    if st.session_state.current_chat_id:
        conn = get_db_connection()
        current_chat_messages = conn.execute(
            'SELECT role, content FROM messages WHERE chat_id = ? ORDER BY timestamp',
            (st.session_state.current_chat_id,)
        ).fetchall()
        conn.close()
        current_chat_messages = [dict(msg) for msg in current_chat_messages]
    
    # Display messages
    for msg in current_chat_messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])
    
    # Handle user input
    if prompt := st.chat_input("Enter your message:"):
        if not st.session_state.current_chat_id:
            st.session_state.current_chat_id = create_new_chat()
        
        save_message(st.session_state.current_chat_id, 'user', prompt)
        
        cached_response = get_cached_response(prompt)
        if cached_response:
            response = cached_response
        else:
            history = current_chat_messages.copy()
            response = ollama_generate(prompt, history, temperature)
            embedding = get_embedding(prompt)
            store_query_in_db(prompt, embedding, response)
        
        save_message(st.session_state.current_chat_id, 'assistant', response)
        
        chat_history = load_chat_history()
        if len(chat_history[st.session_state.current_chat_id]['messages']) == 1:
            update_chat_title(st.session_state.current_chat_id, prompt[:30])

        st.rerun()

# Main app
def main():
    initialize_db()
    initialize_session_state()
    temperature = chat_history_sidebar()
    main_chat_interface(temperature)

if __name__ == "__main__":
    main()