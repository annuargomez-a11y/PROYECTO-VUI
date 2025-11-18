import os
import sys
import logging
import streamlit as st
import nest_asyncio
import re 
from fpdf import FPDF # <-- Se mantiene, aunque ya no se usa la funcion
from io import BytesIO

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

# --- API KEYS ---
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("Error: Falta la clave API de OpenAI.")
    st.stop() 

pdf_folder_path = "./ARCHIVOS/"
persist_dir = "./storage"

# --- FUNCIÓN DE LIMPIEZA (La dejamos simple) ---
# Se mantiene, aunque ya no la usamos para el PDF, pero es buena práctica tenerla
def clean_text_for_pdf(text):
    return text

# --- FUNCIÓN PDF (Se queda vacía) ---
def create_pdf(text):
    return None

# --- MOTOR RAG ---
@st.cache_resource
def get_query_engine():
    # El motor principal usará un template que PIDE TABLAS para la PANTALLA (esto funciona bien)
    llm = OpenAI(model="gpt-4o-mini", temperature=0.2)
    embed_model = OpenAIEmbedding(model="text-embedding-3-large")
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    print("--- INICIANDO MOTOR ---")
    reader = SimpleDirectoryReader(input_dir=pdf_folder_path, recursive=True)
    documents = reader.load_data()
    
    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
    nodes = node_parser.get_nodes_from_documents(documents)
    
    index = VectorStoreIndex(nodes, show_progress=True)
    
    # Template Único: Pedimos tablas (para la pantalla)
    template_str_markdown = (
        "Eres Janus, un experto asesor de inversión extranjera en Colombia. Responde usando formato Markdown.\n"
        "---------------------\n"
        "Contexto:\n{context_str}\n"
        "---------------------\n"
        "Instrucciones:\n"
        "1. Responde en el idioma de la pregunta. 2. Usa TABLAS DE MARKDOWN para cualquier comparación o listado de ventajas/desventajas. 3. Sé detallado y profesional.\n"
        "Pregunta: {query_str}\n\n"
        "Respuesta:"
    )
    qa_template_markdown = PromptTemplate(template_str_markdown)
    
    # Creamos un solo motor de consulta
    query_engine = index.as_query_engine(similarity_top_k=5, text_qa_template=qa_template_markdown)
    return query_engine

# --- INTERFAZ ---
st.title("Asistente Janus")
st.caption("Tu guía para la Ventanilla Única de Inversión (VUI).")

tab_chat, tab_faq = st.tabs(["Consultar a Janus 💬", "Preguntas Frecuentes 💡"])

# --- Ejecución del Motor ---
try:
    query_engine = get_query_engine()
except Exception as e:
    st.error(f"Error al cargar el motor: {e}")
    st.stop()


# --- Pestaña 1: Consultar a Janus ---
with tab_chat:
    st.header("Haz tu consulta")
    st.markdown("¡Hola! Soy Janus, tu asistente virtual. ¡Estoy aquí para guiarte en tu Inversión Directa en Colombia!")

    with st.form("query_form"):
        prompt = st.text_area("Escribe tu consulta aquí:", height=100)
        submitted = st.form_submit_button("Enviar Consulta a Janus")

    if submitted:
        if not prompt:
            st.warning("Por favor, escribe una pregunta.")
        else:
            with st.spinner("Janus está consultando la Guía Legal..."):
                try:
                    # Llama al motor (el único que queda)
                    respuesta = query_engine.query(prompt)
                    response_text = str(respuesta)
                    
                    with st.expander("Ver Respuesta de Janus", expanded=True):
                        st.markdown(response_text) # Muestra el markdown bonito
                        
                        # --- ¡ROLLBACK A TXT! ---
                        st.download_button(
                            label="📄 Descargar Respuesta (TXT)",
                            data=response_text,
                            file_name="Informe_Janus.txt",
                            mime="text/plain"
                        )
                except Exception as e:
                    st.error(f"Error: {e}")

# --- Pestaña 2: Preguntas Frecuentes ---
with tab_faq:
    st.header("Preguntas Frecuentes")
    
    faq_1 = "¿Cuál es la estructura de sociedad recomendada (S.A.S.) y capital mínimo?"
    # ... (y el resto de botones de FAQ) ...
    
    if st.button(faq_1):
         with st.spinner("Generando..."):
            resp_markdown = query_engine.query(faq_1)
            txt_resp = str(resp_markdown)
            
            with st.expander("Respuesta", expanded=True):
                st.markdown(txt_resp)
                
                # ¡BOTÓN DE DESCARGA TXT!
                st.download_button("📥 Descargar TXT", data=txt_resp, file_name="FAQ_Janus.txt", mime="text/plain")
