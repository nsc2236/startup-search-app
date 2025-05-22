import streamlit as st
import numpy as np
import json
import faiss
from sentence_transformers import SentenceTransformer

# Load models and files
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_indexes():
    # Load FAISS indexes
    meeting_index = faiss.read_index("meeting.index")
    with open("meeting_metadata.json") as f:
        meeting_meta = json.load(f)

    domain_index = faiss.read_index("domain.index")
    with open("domain_metadata.json") as f:
        domain_meta = json.load(f)

    return meeting_index, meeting_meta, domain_index, domain_meta

model = load_model()
meeting_index, meeting_meta, domain_index, domain_meta = load_indexes()

# Decide which index to use based on query
def route_query(query: str):
    q = query.lower()
    if "startup" in q or "founder" in q or "raised" in q:
        return "meeting"
    elif "looking for" in q or "interest" in q or "domain" in q:
        return "domain"
    else:
        return "meeting"  # fallback

# Search function
def search_faiss(index, metadata, query, k=5):
    query_vec = model.encode([query]).astype("float32")
    D, I = index.search(query_vec, k)
    results = []
    for idx in I[0]:
        chunk = metadata[str(idx)]
        results.append(chunk)
    return results

# Streamlit UI
st.title("🔎 Startup Search Tool (FAISS)")
query = st.text_input("What do you want to find?")
top_k = st.slider("Number of results", 1, 10, 5)

if st.button("Search") and query:
    target = route_query(query)
    st.markdown(f"**Searching in:** `{target}` data")

    if target == "meeting":
        results = search_faiss(meeting_index, meeting_meta, query, top_k)
    else:
        results = search_faiss(domain_index, domain_meta, query, top_k)

    for i, r in enumerate(results, 1):
        st.markdown(f"### {i}. {r.get('startup', 'Unknown')}")
        st.markdown(f"**Source:** {r.get('source_type', 'Unknown')}")
        st.write(r['text'])
