import streamlit as st
import nest_asyncio
import os
import sys
import logging
from datetime import datetime
from fpdf import FPDF

# --- APLICAR PARCHE INMEDIATAMENTE ---
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

# --- API KEYS ---
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("Error Crítico: Falta la clave API de OpenAI en los Secrets.")
    st.stop()

# Rutas
pdf_folder_path = "./ARCHIVOS/"
persist_dir = "./storage"

# --- MOTOR RAG ---
@st.cache_resource
def get_query_engine():
    
    # 1. Configuración del Sistema (Cerebro + Instrucciones Maestras)
    system_instruction = (
        "You are Janus, the Official Investment Assistant for the Single Investment Window (VUI) of Colombia. "
        "Your role is to act as a STRATEGIC FACILITATOR.\n"
        "CRITICAL RULES:\n"
        "1. LANGUAGE (MANDATORY): Detect the language of the user's question and answer in that EXACT SAME LANGUAGE. "
        "If the user asks in English, answer in English.\n"
        "2. VUE RULE: If asked about creating a company (S.A.S.), refer to VUE (Ventanilla Única Empresarial). Do NOT mention VUCE.\n"
        "3. CONTENT: Prioritize practical steps ('HOW') over legal theory ('WHAT').\n"
        "4. FORMAT: Use Markdown (bolding, lists)."
    )

    llm = OpenAI(
        model="gpt-4o-mini", 
        temperature=0.1,
        system_prompt=system_instruction
    )
    
    embed_model = OpenAIEmbedding(model="text-embedding-3-large")

    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # 2. Carga
    reader = SimpleDirectoryReader(input_dir=pdf_folder_path, recursive=True)
    documents = reader.load_data()
    
    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
    nodes = node_parser.get_nodes_from_documents(documents)
    
    index = VectorStoreIndex(nodes, show_progress=True)
    
    # 3. Motor
    query_engine = index.as_query_engine(similarity_top_k=5) 
    return query_engine

# --- INTERFAZ ---
st.title("Asistente Janus")
st.caption("Tu guía para la Ventanilla Única de Inversión (VUI).")

tab_chat, tab_faq = st.tabs(["Consultar a Janus 💬", "Preguntas Frecuentes 💡"])

try:
    query_engine = get_query_engine()
except Exception as e:
    st.error(f"Error al cargar el motor: {e}")
    st.stop()

# --- Pestaña 1: Chat ---
with tab_chat:
    st.header("Haz tu consulta")
    st.markdown("¡Hola! Soy Janus. Estoy aquí para guiarte en tu Inversión Directa en Colombia.")

    with st.form("query_form"):
        prompt = st.text_area("Escribe tu consulta aquí:", height=100)
        submitted = st.form_submit_button("Enviar Consulta a Janus")

    if submitted and prompt:
        with st.spinner("Janus está analizando..."):
            try:
                respuesta = query_engine.query(prompt)
                response_text = str(respuesta)
                
                with st.expander("Ver Respuesta de Janus", expanded=True):
                    st.markdown(response_text)
                    
                    # Descarga simple TXT
                    ahora = datetime.now()
                    nombre = f"Janus.Answer.{ahora.strftime('%Y%m%d.%H%M')}.txt"
                    contenido = f"PREGUNTA:\n{prompt}\n\nRESPUESTA:\n{response_text}"
                    
                    st.download_button("📥 Guardar Respuesta (TXT)", data=contenido, file_name=nombre, mime="text/plain")
            except Exception as e:
                st.error(f"Error: {e}")

# --- Pestaña 2: FAQs ---
with tab_faq:
    st.header("Preguntas Frecuentes")
    faq_1 = "¿Qué incentivos fiscales hay para energías renovables no convencionales?"
    faq_2 = "¿Cuál es la estructura de sociedad recomendada (S.A.S.) y capital mínimo?"
    faq_3 = "¿Existen restricciones para repatriar utilidades al exterior?"
    
    def run_faq(question):
        with st.spinner("Consultando..."):
            resp = query_engine.query(question)
            txt_resp = str(resp)
            with st.expander("Respuesta", expanded=True):
                st.markdown(txt_resp)

    if st.button(faq_1): run_faq(faq_1)
    if st.button(faq_2): run_faq(faq_2)
    if st.button(faq_3): run_faq(faq_3)
