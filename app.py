import os
import sys
import logging
import streamlit as st
import nest_asyncio
from datetime import datetime

# --- 1. PARCHES DE SISTEMA ---
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

# --- 2. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Asistente Janus (VUI)",
    page_icon="🗝️",
    layout="centered"
)

# --- 3. API KEYS ---
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("Error Crítico: Falta la clave API de OpenAI en los Secrets.")
    st.stop()

pdf_folder_path = "./ARCHIVOS/"
persist_dir = "./storage"

# --- 4. MOTOR RAG ---
@st.cache_resource
def get_query_engine():
    
    # Configuración del Modelo
    # Usamos temperatura baja (0.1) para máxima fidelidad a las instrucciones
    llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
    
    # Configuración del Traductor (Embeddings)
    embed_model = OpenAIEmbedding(model="text-embedding-3-large")

    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # Carga de Documentos
    reader = SimpleDirectoryReader(input_dir=pdf_folder_path, recursive=True)
    documents = reader.load_data()
    
    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
    nodes = node_parser.get_nodes_from_documents(documents)
    
    index = VectorStoreIndex(nodes, show_progress=True)
    
    # --- INSTRUCCIONES MAESTRAS (PROMPT TEMPLATE) ---
    # Usamos triple comilla (""") para evitar errores de sintaxis.
    # Estas instrucciones están en INGLÉS para garantizar el comportamiento multilingüe.
    
    qa_template_str = """You are Janus, the Official Investment Assistant for the Single Investment Window (VUI) of Colombia.
Your role is to act as a STRATEGIC FACILITATOR.

---------------------
CONTEXT INFORMATION (Legal Guides, Manuals, Projects):
{context_str}
---------------------

CRITICAL INSTRUCTIONS FOR ANSWERING:
1. LANGUAGE (MANDATORY): Detect the language of the user's query below. You MUST answer in that EXACT SAME LANGUAGE.
   - Query in English -> Answer in English.
   - Query in French -> Answer in French.
   - Query in Spanish -> Answer in Spanish.
2. VUE RULE: If the query is about creating a company (S.A.S.) or commercial registration, refer ONLY to the VUE (Ventanilla Única Empresarial). Do NOT mention VUCE.
3. CONTENT: Prioritize practical steps ('HOW') over legal theory ('WHAT').
4. OPPORTUNITIES: If asked about projects, summarize the available Project Fiches.
5. FORMAT: Use Markdown (bolding, lists) for readability.

Query: {query_str}

Answer (in the query's language):"""

    janus_template = PromptTemplate(qa_template_str)
    
    # Inyectamos el template al motor
    query_engine = index.as_query_engine(
        similarity_top_k=5, 
        text_qa_template=janus_template
    ) 
    return query_engine

# --- 5. INTERFAZ DE USUARIO ---
st.title("Asistente Janus")
st.caption("Tu guía para la Ventanilla Única de Inversión (VUI).")

tab_chat, tab_faq = st.tabs(["Consultar a Janus 💬", "Preguntas Frecuentes 💡"])

# Carga del motor
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
            with st.expander(f"Respuesta: {question}", expanded=True):
                st.markdown(txt_resp)
                
                ahora = datetime.now()
                nombre = f"Janus.FAQ.{ahora.strftime('%Y%m%d.%H%M')}.txt"
                contenido = f"PREGUNTA:\n{question}\n\nRESPUESTA:\n{txt_resp}"
                st.download_button("📥 Descargar TXT", data=contenido, file_name=nombre, mime="text/plain")

    if st.button(faq_1): run_faq(faq_1)
    if st.button(faq_2): run_faq(faq_2)
    if st.button(faq_3): run_faq(faq_3)
