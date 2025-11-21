import os
import sys
import logging
import streamlit as st
import nest_asyncio # <--- ¡LA LÍNEA QUE FALTABA ANTES ESTÁ AQUÍ!
from datetime import datetime

# --- PARCHES ---
nest_asyncio.apply()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
    PromptTemplate
)
from llama_index.core.node_parser import SentenceSplitter 
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Asistente Janus (VUI)", page_icon="🗝️", layout="centered")

# --- ¡CORRECCIÓN CRÍTICA! INICIALIZAR ESTADO AQUÍ ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- API KEYS ---
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("Error: Falta la clave API de OpenAI.")
    st.stop() 

pdf_folder_path = "./ARCHIVOS/"
persist_dir = "./storage"

# --- MOTOR RAG ---
@st.cache_resource
def get_query_engine():
    
    # 1. Cerebro (GPT-4o-mini)
    llm = OpenAI(model="gpt-4o-mini", temperature=0.2)
    
    # 2. Traductor (Embeddings Pro)
    embed_model = OpenAIEmbedding(model="text-embedding-3-large")

    Settings.llm = llm
    Settings.embed_model = embed_model
    
    print("--- INICIANDO MOTOR JANUS ---")
    reader = SimpleDirectoryReader(input_dir=pdf_folder_path, recursive=True)
    documents = reader.load_data()
    
    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
    nodes = node_parser.get_nodes_from_documents(documents)
    
    index = VectorStoreIndex(nodes, show_progress=True)
    
    # 3. Personalidad de Janus
    template_str = (
        "You are Janus, the Official Investment Assistant for the Single Investment Window (VUI) of Colombia.\n"
        "Your role is to act as a STRATEGIC FACILITATOR.\n"
        "---------------------\n"
        "Context Information (Legal Guides & Manuals):\n{context_str}\n"
        "---------------------\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. LANGUAGE DETECTION (MANDATORY): Detect the language of the user's 'Query' below. You MUST answer in that EXACT SAME LANGUAGE.\n"
        "   - If Query is in English -> Answer in English.\n"
        "   - If Query is in Spanish -> Answer in Spanish.\n"
        "2. VUE RULE: If the user asks about creating a company or S.A.S., refer them to the VUE (Ventanilla Única Empresarial). Do NOT mention VUCE.\n"
        "3. CONTENT: Prioritize practical steps ('HOW') over legal theory ('WHAT'). Use the provided context.\n"
