import streamlit as st
import nest_asyncio
import os
import sys
import logging
from datetime import datetime

# --- 1. PARCHES DE SISTEMA (OBLIGATORIO AL INICIO) ---
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

# Rutas
pdf_folder_path = "./ARCHIVOS/"
persist_dir = "./storage"

# --- 4. MOTOR RAG ---
@st.cache_resource
def get_query_engine():
    
    # Configuración del Modelo (Cerebro)
    llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
    
    # Configuración del Traductor
    embed_model = OpenAIEmbedding(model="text-embedding-3-large")

    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # Carga de Documentos
    # Usamos un truco para evitar re-cargar si no es necesario, pero asegurando lectura
    if not os.path.exists(persist_dir):
        reader = SimpleDirectoryReader(input_dir=pdf_folder_path, recursive=True)
        documents = reader.load_data()
        
        node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
        nodes = node_parser.get_nodes_from_documents(documents)
        
        index = VectorStoreIndex(nodes, show_progress=True)
        index.storage_context.persist(persist_dir)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
    
    # --- LA SOLUCIÓN: TEXT QA TEMPLATE ---
    # Este molde reemplaza al default de LlamaIndex.
    # Obliga al modelo a mirar el idioma de la pregunta {query_str} justo antes de responder.
    qa_prompt_str = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, answer the query.\n"
        "CRITICAL RULE: Answer in the SAME LANGUAGE as the query below.\n"
        "If the query is in English, answer in English.\n"
        "If the query is in Spanish, answer in Spanish.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    qa_template = PromptTemplate(qa_prompt_str)

    # Inyectamos el template específico
    query_engine = index.as_query_engine(
        similarity_top_k=5,
        text_qa_template=qa_template
    ) 
    return query_engine

# --- 5. INTERFAZ ---
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
            with st.expander("Respuesta", expanded=True):
                st.markdown(txt_resp)

    if st.button(faq_1): run_faq(faq_1)
    if st.button(faq_2): run_faq(faq_2)
    if st.button(faq_3): run_faq(faq_3)
