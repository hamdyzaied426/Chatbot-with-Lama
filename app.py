import streamlit as st
import datetime
from database import (
    initialize_db,
    create_chat,
    delete_chat,
    delete_all_chats,
    save_message,
    load_chat_history,
    update_chat_title
)
from chat import (
    initialize_index_and_cache,
    get_cached_response,
    store_query,
    ollama_generate
)

def initialize_session_state():
    initialize_db()
    if 'current_chat_id' not in st.session_state:
        st.session_state.current_chat_id = None
    if 'index' not in st.session_state or 'cache' not in st.session_state:
        st.session_state.index, st.session_state.cache = initialize_index_and_cache()

def chat_history_sidebar():
    with st.sidebar:
        st.header("Chat History")
        if st.button("➕ New Chat", use_container_width=True):
            chat_id = datetime.datetime.now().isoformat()
            create_chat(chat_id)
            st.session_state.current_chat_id = chat_id
        
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
            
            col1, col2 = st.columns([4, 1])
            with col1:
                preview = chat['title'] or chat['messages'][0]['content'][:20] + "..."
                if st.button(
                    f"{preview} ({'Today' if dt == today else 'Yesterday' if dt == today - datetime.timedelta(days=1) else dt.strftime('%B %d, %Y')})",
                    key=chat_id,
                    use_container_width=True
                ):
                    st.session_state.current_chat_id = chat_id
            with col2:
                if st.button("🗑️", key=f"del_{chat_id}"):
                    delete_chat(chat_id)
                    st.rerun()

        st.divider()
        if st.button("❌ Delete All Chats", use_container_width=True):
            delete_all_chats()
            st.rerun()

        return temperature

def main_chat_interface(temperature):
    st.title("Chatbot with Ollama")
    
    current_chat_messages = []
    if st.session_state.current_chat_id:
        chat_history = load_chat_history()
        current_chat = chat_history.get(st.session_state.current_chat_id, {})
        current_chat_messages = current_chat.get('messages', [])
    
    for msg in current_chat_messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])
    
    if prompt := st.chat_input("Enter your message:"):
        if not st.session_state.current_chat_id:
            chat_id = datetime.datetime.now().isoformat()
            create_chat(chat_id)
            st.session_state.current_chat_id = chat_id
        
        save_message(st.session_state.current_chat_id, 'user', prompt)
        
        try:
            cached_response = get_cached_response(prompt, st.session_state.index, st.session_state.cache)
            if cached_response:
                response = cached_response
            else:
                response = ollama_generate(prompt, current_chat_messages, temperature)
                store_query(prompt, response, st.session_state.index, st.session_state.cache)
            
            save_message(st.session_state.current_chat_id, 'assistant', response)
            
            if len(current_chat_messages) == 0:
                update_chat_title(st.session_state.current_chat_id, prompt[:30])
            
            st.rerun()
        except Exception as e:
            st.error(f"Error: {str(e)}")

def main():
    initialize_session_state()
    temperature = chat_history_sidebar()
    main_chat_interface(temperature)

if __name__ == "__main__":
    main()