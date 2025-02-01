import sqlite3
import datetime
import pickle

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

def insert_or_update_query(query, embedding, response):
    conn = get_db_connection()
    try:
        conn.execute("""
            INSERT INTO queries (query, embedding, response)
            VALUES (?, ?, ?)
            ON CONFLICT(query) DO UPDATE SET 
                response = excluded.response,
                usage_count = usage_count + 1
        """, (query, pickle.dumps(embedding), response))
        conn.commit()
        return conn.total_changes > 0
    finally:
        conn.close()

def get_all_queries():
    conn = get_db_connection()
    try:
        return conn.execute('SELECT * FROM queries ORDER BY id').fetchall()
    finally:
        conn.close()

def create_chat(chat_id, title="New Chat"):
    conn = get_db_connection()
    try:
        conn.execute('INSERT INTO chats (id, title) VALUES (?, ?)', (chat_id, title))
        conn.commit()
    finally:
        conn.close()

def delete_chat(chat_id):
    conn = get_db_connection()
    try:
        conn.execute('DELETE FROM chats WHERE id = ?', (chat_id,))
        conn.commit()
    finally:
        conn.close()

def delete_all_chats():
    conn = get_db_connection()
    try:
        conn.execute('DELETE FROM chats')
        conn.commit()
    finally:
        conn.close()

def save_message(chat_id, role, content):
    conn = get_db_connection()
    try:
        conn.execute(
            'INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)',
            (chat_id, role, content)
        )
        conn.commit()
    finally:
        conn.close()

def load_chat_history():
    conn = get_db_connection()
    try:
        chats = conn.execute('SELECT * FROM chats ORDER BY created_at DESC').fetchall()
        messages = {}
        for chat in chats:
            chat_messages = conn.execute(
                'SELECT role, content FROM messages WHERE chat_id = ? ORDER BY timestamp',
                (chat['id'],)
            ).fetchall()
            messages[chat['id']] = [dict(msg) for msg in chat_messages]
        return {chat['id']: {'title': chat['title'], 'messages': messages[chat['id']]} for chat in chats}
    finally:
        conn.close()

def update_chat_title(chat_id, title):
    conn = get_db_connection()
    try:
        conn.execute('UPDATE chats SET title = ? WHERE id = ?', (title, chat_id))
        conn.commit()
    finally:
        conn.close()