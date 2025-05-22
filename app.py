import streamlit as st
import numpy as np
import json
import faiss
from sentence_transformers import SentenceTransformer
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io

# Load service account credentials from Streamlit Secrets
info = json.loads(st.secrets["service_account"])
credentials = service_account.Credentials.from_service_account_info(info)
drive_service = build("drive", "v3", credentials=credentials)

# Replace these with your actual Drive file IDs
FILE_IDS = {
    "meeting_index": "1Q1MMqeK4clIKFDnvsYYwVU8yaLZ6sMXK",
    "meeting_metadata": "12-4vSANUGLrwz_q9kJz_56HoUQ1JAq5g",
    "domain_index": "1-eJ46ree1EzDqyCzaabvdQ4oDi3sEEgv",
    "domain_metadata": "1sro63gTse1-XbiEPrEs5PApv37a5CHNf",
}

def download_json(file_id):
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return json.load(fh)

def download_npy(file_id):
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return np.load(fh)

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_indexes():
    # Load and build FAISS indexes
    meeting_embeddings = download_npy(FILE_IDS["meeting_index"])
    domain_embeddings = download_npy(FILE_IDS["domain_index"])

    meeting_index = faiss.IndexFlatL2(meeting_embeddings.shape[1])
    meeting_index.add(meeting_embeddings)

    domain_index = faiss.IndexFlatL2(domain_embeddings.shape[1])
    domain_index.add(domain_embeddings)

    # Load metadata
    meeting_metadata = download_json(FILE_IDS["meeting_metadata"])
    domain_metadata = download_json(FILE_IDS["domain_metadata"])

    return meeting_index, meeting_metadata, domain_index, domain_metadata

model = load_model()
meeting_index, meeting_metadata, domain_index, domain_metadata = load_indexes()

# Routing logic
def route_query(query: str):
    q = query.lower()
    if "startup" in q or "founder" in q or "raised" in q:
        return "meeting"
    elif "looking for" in q or "interest" in q or "domain" in q:
        return "domain"
    else:
        return "meeting"

# Search function
def search_faiss(index, metadata, query, k=5):
    query_vec = model.encode([query]).astype("float32")
    D, I = index.search(query_vec, k)
    return [metadata[str(i)] for i in I[0]]

# Streamlit UI
st.set_page_config(page_title="🔍 Startup Search", layout="centered")
st.title("🔍 Startup Search Tool")

query = st.text_input("What do you want to find?")
top_k = st.slider("Number of results", 1, 10, 5)

if st.button("Search") and query:
    target = route_query(query)
    st.markdown(f"**Searching in:** `{target}` data")

    if target == "meeting":
        results = search_faiss(meeting_index, meeting_metadata, query, top_k)
    else:
        results = search_faiss(domain_index, domain_metadata, query, top_k)

    for i, r in enumerate(results, 1):
        st.markdown(f"### {i}. {r.get('startup', 'Unknown')}")
        st.markdown(f"**Source:** {r.get('source_type', 'Unknown')}")
        st.write(r['text'])
