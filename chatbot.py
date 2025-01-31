import streamlit as st
import faiss
import sqlite3
import pickle
import datetime
import torch
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
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
        'model_ready': False,
        'model': None,
        'tokenizer': None
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

# Model handling
def load_model(model_name="mistralai/Mistral-7B-v0.1"):
    try:
        st.session_state.tokenizer = AutoTokenizer.from_pretrained(model_name)
        st.session_state.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        st.session_state.model_ready = True
        return True
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return False

def generate_response(prompt, history, max_length=512, temperature=0.7):
    if not st.session_state.model_ready:
        return "Model not loaded", False

    try:
        # Format conversation history
        messages = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": msg["content"]}
            for i, msg in enumerate(history)
        ]
        messages.append({"role": "user", "content": prompt})
        
        # Create instruction format
        formatted_prompt = st.session_state.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize input
        inputs = st.session_state.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            max_length=4096,
            truncation=True
        ).to("cuda")

        # Generate response
        outputs = st.session_state.model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=0.95,
            do_sample=True,
            pad_token_id=st.session_state.tokenizer.eos_token_id
        )

        # Decode response
        response = st.session_state.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):], 
            skip_special_tokens=True
        )

        return response, True

    except Exception as e:
        return f"Generation error: {str(e)}", False

# UI components
def sidebar():
    with st.sidebar:
        st.header("Model Settings")
        
        model_name = st.selectbox(
            "Select Model",
            [
                "mistralai/Mistral-7B-v0.1",
                "HuggingFaceH4/zephyr-7b-beta",
                "togethercomputer/RedPajama-INCITE-7B-Chat"
            ],
            index=0
        )
        
        if st.button("Load Model"):
            with st.spinner(f"Loading {model_name.split('/')[-1]}..."):
                if load_model(model_name):
                    st.success("Model loaded successfully!")
                else:
                    st.error("Failed to load model")

        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        max_length = st.slider("Max Response Length", 50, 2048, 512, 50)
        
        st.divider()
        st.header("Chat Management")
        
        if st.button("➕ New Chat"):
            st.session_state.current_chat_id = create_chat()
            st.rerun()
        
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
        
        return temperature, max_length

def main_interface(temperature, max_length):
    st.title("Open Source Chatbot")
    
    # Model status
    status_color = "green" if st.session_state.model_ready else "red"
    status_text = "● Model Ready" if st.session_state.model_ready else "● Model Not Loaded"
    st.markdown(f"**Status**: <span style='color:{status_color}'>{status_text}</span>", unsafe_allow_html=True)

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
            response, success = generate_response(prompt, messages, max_length, temperature)
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
    temp, length = sidebar()
    main_interface(temp, length)
