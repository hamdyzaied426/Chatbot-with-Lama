import faiss
import numpy as np
import requests
import pickle
from sentence_transformers import SentenceTransformer
from collections import Counter
from database import get_all_queries

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text):
    return embedder.encode(text).reshape(1, -1).astype('float32')

def initialize_index_and_cache():
    index = faiss.IndexFlatIP(384)
    cache = {}
    queries = get_all_queries()
    for idx, q in enumerate(queries):
        embedding = pickle.loads(q['embedding'])
        index.add(embedding)
        cache[idx] = q['response']
    return index, cache

def get_cached_response(query, index, cache):
    try:
        query_vector = get_embedding(query)
        
        if index.ntotal > 0:
            similarities, indices = index.search(query_vector, k=5)
            valid_responses = [
                cache.get(indices[0][i]) 
                for i in range(len(similarities[0])) 
                if similarities[0][i] > 0.75
            ]
            if valid_responses:
                return Counter(valid_responses).most_common(1)[0][0]
                
        queries = get_all_queries()
        for q in queries:
            db_embedding = pickle.loads(q['embedding'])
            similarity = np.dot(query_vector, db_embedding.T)[0][0]
            if similarity > 0.6:
                index.add(db_embedding)
                new_index = index.ntotal - 1
                cache[new_index] = q['response']
                return q['response']
        
        return None
    except Exception as e:
        raise e

def store_query(query, response, index, cache):
    embedding = get_embedding(query)
    from database import insert_or_update_query
    if insert_or_update_query(query, embedding, response):
        index.add(embedding)
        cache[index.ntotal - 1] = response

def ollama_generate(prompt, history, temperature=0.9):
    try:
        formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": f"{formatted_history}\nuser: {prompt}\nassistant:",
                "stream": False,
                "options": {"temperature": temperature}
            }
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        raise e