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

# --- 3. VALIDACIÓN DE CLAVES ---
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("Error Crítico: Falta la clave API de OpenAI en los 'Secrets' de Streamlit.")
    st.stop() 

# Rutas de archivos
pdf_folder_path = "./ARCHIVOS/"
persist_dir = "./storage"

# --- 4. MOTOR DE INTELIGENCIA (RAG) ---
@st.cache_resource
def get_query_engine():
    
    # --- A. PERSONALIDAD MAESTRA (SYSTEM PROMPT) ---
    # Esta instrucción va directo al "cerebro" del modelo.
    # Al estar en inglés, elimina el sesgo hacia el español.
    janus_system_prompt = (
        "You are Janus, the Official Investment Assistant for the Single Investment Window (VUI) of Colombia. "
        "Your role is to act as a STRATEGIC FACILITATOR.\n"
        "CRITICAL RULES:\n"
        "1. LANGUAGE (MANDATORY): You MUST answer in the EXACT SAME LANGUAGE as the user's question. "
        "If the user asks in English, answer in English. If in Spanish, answer in Spanish.\n"
        "2. CONTENT: Use the provided context to answer. Prioritize practical steps ('HOW') over legal theory.\n"
        "3. VUE RULE: For company creation (S.A.S.), refer to VUE (Ventanilla Única Empresarial). Do NOT mention VUCE.\n"
        "4. FORMAT: Use Markdown (bolding, lists) for readability."
    )

    # --- B. Configuración del Modelo ---
    llm = OpenAI(
        model="gpt-4o-mini", 
        temperature=0.1, 
        system_prompt=janus_system_prompt # <-- ¡ESTA ES LA CLAVE QUE FALTABA!
    )
    
    embed_model = OpenAIEmbedding(model="text-embedding-3-large")

    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # --- C. Carga de Documentos ---
    print("--- INICIANDO MOTOR JANUS ---")
    reader = SimpleDirectoryReader(input_dir=pdf_folder_path, recursive=True)
    documents = reader.load_data()
    
    # --- D. Indexación ---
    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
    nodes = node_parser.get_nodes_from_documents(documents)
    
    index = VectorStoreIndex(nodes, show_progress=True)
    
    # --- E. Motor de Consulta ---
    query_engine = index.as_query_engine(similarity_top_k=5) 
    
    return query_engine

# --- 5. INTERFAZ DE USUARIO ---
st.title("Asistente Janus")
st.caption("Tu guía para la Ventanilla Única de Inversión (VUI).")

# Pestañas
tab_chat, tab_faq = st.tabs(["Consultar a Janus 💬", "Preguntas Frecuentes 💡"])

# Carga del motor
try:
    query_engine = get_query_engine()
except Exception as e:
    st.error(f"Error al iniciar el motor: {e}")
    st.stop()

# --- PESTAÑA 1: CHAT PRINCIPAL ---
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
                    
                    # Generar nombre de archivo con fecha
                    ahora = datetime.now()
                    nombre_archivo = f"Janus.Answer.{ahora.strftime('%Y%m%d.%H%M')}.txt"
                    fecha_texto = ahora.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Contenido del TXT
                    contenido_txt = f"""================================================================================
REPORTE DE CONSULTA - ASISTENTE VUI JANUS
FECHA: {fecha_texto}
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
                st.error(f"Error generando respuesta: {e}")

# --- PESTAÑA 2: PREGUNTAS FRECUENTES ---
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
            
            with st.expander(f"Respuesta: {question}", expanded=True):
                st.markdown(txt_resp)
                
                ahora = datetime.now()
                nombre_archivo = f"Janus.FAQ.{ahora.strftime('%Y%m%d.%H%M')}.txt"
                
                contenido = f"PREGUNTA:\n{question}\n\nRESPUESTA:\n{txt_resp}"
                st.download_button("📥 Descargar TXT", data=contenido, file_name=nombre_archivo, mime="text/plain")

    if st.button(faq_1): run_faq(faq_1)
    if st.button(faq_2): run_faq(faq_2)
    if st.button(faq_3): run_faq(faq_3)
    if st.button(faq_4): run_faq(faq_4)
    if st.button(faq_5): run_faq(faq_5)
