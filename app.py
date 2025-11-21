import os
import sys
import logging
import streamlit as st
import nest_asyncio
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
    PromptTemplate # <-- NUEVO: Importamos esto para controlar el molde
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
    st.error("Error: Falta la clave API de OpenAI.")
    st.stop() 

pdf_folder_path = "./ARCHIVOS/"
persist_dir = "./storage"

# --- MOTOR RAG ---
@st.cache_resource
def get_query_engine():
    
    # 1. Cerebro
    llm = OpenAI(model="gpt-4o-mini", temperature=0.1) # Bajamos temperatura para ser más estrictos
    
    # 2. Traductor
    embed_model = OpenAIEmbedding(model="text-embedding-3-large")

    Settings.llm = llm
    Settings.embed_model = embed_model
    
    print("--- INICIANDO MOTOR JANUS ---")
    reader = SimpleDirectoryReader(input_dir=pdf_folder_path, recursive=True)
    documents = reader.load_data()
    
    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
    nodes = node_parser.get_nodes_from_documents(documents)
    
    index = VectorStoreIndex(nodes, show_progress=True)
    
    # 3. Personalidad de Janus (VERSIÓN AGRESIVA DE IDIOMA)
    template_str = (
        "You are Janus, an expert investment assistant for Colombia.\n"
        "---------------------\n"
        "Context Information (Legal Guides & Manuals):\n{context_str}\n"
        "---------------------\n"
        "INSTRUCTIONS:\n"
        "1. Analyze the user's query below.\n"
        "2. If the query is in English, you MUST answer in English.\n"
        "3. If the query is in Spanish, answer in Spanish.\n"
        "4. Do NOT translate the query, just answer it in the same language.\n"
        "5. Use the context provided to answer. If the context is in Spanish, translate the information to the user's language.\n"
        "6. VUE RULE: Refer to VUE for company creation. Do not mention VUCE.\n\n"
        "User Query: {query_str}\n\n"
        "Answer (in the same language as the query):"
    )
    
    qa_template = PromptTemplate(template_str)
    
    qa_template = PromptTemplate(template_str)

    # 4. Pasamos el template al motor
    query_engine = index.as_query_engine(
        similarity_top_k=5, 
        text_qa_template=qa_template # <-- Aquí inyectamos la regla
    ) 
    return query_engine

# --- INTERFAZ ---
st.title("Asistente Janus")
st.caption("Tu guía para la Ventanilla Única de Inversión (VUI).")

tab_chat, tab_faq = st.tabs(["Consultar a Janus 💬", "Preguntas Frecuentes 💡"])

try:
    query_engine = get_query_engine()
except Exception as e:
    st.error(f"Error: {e}")
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
                    
                    # Lógica de fecha y archivo
                    ahora = datetime.now()
                    fecha_hora_texto = ahora.strftime("%Y-%m-%d %H:%M:%S")
                    fecha_hora_archivo = ahora.strftime("%Y%m%d.%H%M")
                    nombre_archivo = f"Janus.Answer.{fecha_hora_archivo}.txt"

                    contenido_txt = f"""================================================================================
REPORTE DE CONSULTA - ASISTENTE VUI JANUS
FECHA Y HORA: {fecha_hora_texto}
================================================================================

PREGUNTA:
{prompt}

--------------------------------------------------------------------------------

RESPUESTA:
{response_text}

================================================================================
Generado por Inteligencia Artificial - Ventanilla Única de Inversión
"""
                    st.download_button(
                        label="📥 Guardar Respuesta (TXT)",
                        data=contenido_txt,
                        file_name=nombre_archivo,
                        mime="text/plain"
                    )
            except Exception as e:
                st.error(f"Error: {e}")

# --- Pestaña 2: FAQs ---
with tab_faq:
    st.header("Preguntas Frecuentes")
    
    faq_1 = "¿Qué incentivos fiscales hay para energías renovables no convencionales?"
    faq_2 = "¿Cuál es la estructura de sociedad recomendada (S.A.S.) y capital mínimo?"
    faq_3 = "¿Existen restricciones para repatriar utilidades al exterior?"
    faq_4 = "¿Qué permisos ambientales o licencias se necesitan para operar?"
    faq_5 = "¿Qué garantías de estabilidad jurídica ofrece Colombia?"

    def run_faq(question):
        with st.spinner("Consultando..."):
            resp = query_engine.query(question)
            txt_resp = str(resp)
            
            with st.expander("Respuesta", expanded=True):
                st.markdown(txt_resp)
                
                ahora = datetime.now()
                fecha_hora_texto = ahora.strftime("%Y-%m-%d %H:%M:%S")
                fecha_hora_archivo = ahora.strftime("%Y%m%d.%H%M")
                nombre_archivo = f"Janus.FAQ.{fecha_hora_archivo}.txt"

                contenido_txt_faq = f"""================================================================================
REPORTE FAQ - ASISTENTE VUI JANUS
FECHA Y HORA: {fecha_hora_texto}
================================================================================

PREGUNTA:
{question}

--------------------------------------------------------------------------------

RESPUESTA:
{txt_resp}

================================================================================
Generado por Inteligencia Artificial - Ventanilla Única de Inversión
"""
                st.download_button("📥 Descargar TXT", data=contenido_txt_faq, file_name=nombre_archivo, mime="text/plain")

    if st.button(faq_1): run_faq(faq_1)
    if st.button(faq_2): run_faq(faq_2)
    if st.button(faq_3): run_faq(faq_3)
    if st.button(faq_4): run_faq(faq_4)
    if st.button(faq_5): run_faq(faq_5)

