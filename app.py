import os
import sys
import logging
import streamlit as st
import nest_asyncio

# --- PARCHES ---
nest_asyncio.apply()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.core.node_parser import SentenceSplitter 

# --- CAMBIO DE CEREBRO: AHORA USAMOS OPENAI PARA TODO ---
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Asistente Janus (VUI)",
    page_icon="🗝️",
    layout="centered" 
)

# --- CONFIGURACIÓN DE API (SOLO OPENAI) ---
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("Error: Falta la clave API de OpenAI en los Secrets.")
    st.stop() 

pdf_folder_path = "./ARCHIVOS/"
persist_dir = "./storage"

# --- FUNCIÓN DEL MOTOR RAG (100% OPENAI) ---
@st.cache_resource
def get_query_engine():
    
    # 1. CEREBRO (LLM): Usamos GPT-4o-mini (Rápido, barato y muy inteligente)
    # 1. CEREBRO (LLM): Usamos GPT-4o-mini con PERSONALIDAD
    llm = OpenAI(
        model="gpt-4o-mini", 
        temperature=0.2, # Un poquito más creativo para que fluya mejor
        system_prompt="""
        Eres Janus, un experto asesor de inversión extranjera en Colombia.
        Tu trabajo es ayudar a inversionistas a entender la normativa basándote en los documentos proporcionados.
        
        Tus respuestas deben ser:
        1. Completas y detalladas (evita respuestas monosílabas).
        2. Explicativas: Si un concepto es complejo, desglósalo.
        3. Profesionales pero amables.
        
        Si la respuesta es "no hay monto mínimo", explica por qué y qué implica eso para el inversionista (flexibilidad).
        """
    )
    
    # 2. TRADUCTOR (Embedding): Usamos el modelo preciso
    embed_model = OpenAIEmbedding(model="text-embedding-3-large")

    Settings.llm = llm
    Settings.embed_model = embed_model
    
    print("--- INICIANDO MOTOR (FULL OPENAI) ---")
    
    reader = SimpleDirectoryReader(input_dir=pdf_folder_path, recursive=True)
    documents = reader.load_data()
    print(f"Documentos cargados: {len(documents)}")
    
    # Corte Inteligente
    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
    nodes = node_parser.get_nodes_from_documents(documents)
    
    print("Indexando documentos con OpenAI...")
    index = VectorStoreIndex(nodes, show_progress=True)
    
    print("¡Índice creado!")
    
    # Buscamos los 5 trozos más relevantes
    query_engine = index.as_query_engine(similarity_top_k=5) 
    return query_engine

# --- INTERFAZ DE USUARIO ---

st.title("Asistente Janus")
st.caption("Tu guía para la Ventanilla Única de Inversión (VUI).")

tab_chat, tab_faq = st.tabs(["Consultar a Janus 💬", "Preguntas Frecuentes 💡"])

# --- Pestaña 1: El Chat ---
with tab_chat:
    st.header("Haz tu consulta")
    st.markdown("¡Hola! Soy Janus. Estoy conectado al motor GPT-4o para darte respuestas precisas y estables.")

    try:
        query_engine = get_query_engine()
    except Exception as e:
        st.error(f"Error al cargar el motor: {e}")
        st.stop()

    with st.form("query_form"):
        prompt = st.text_area("Escribe tu consulta aquí:", height=150)
        submitted = st.form_submit_button("Enviar Consulta a Janus")

    if submitted:
        if not prompt:
            st.warning("Por favor, escribe una pregunta.")
        else:
            with st.spinner("Analizando con GPT-4o..."):
                try:
                    respuesta = query_engine.query(prompt)
                    response_text = str(respuesta)
                    
                    with st.expander("Ver Respuesta de Janus", expanded=True):
                        st.markdown(response_text)
                        st.download_button("📥 Guardar Respuesta", data=response_text, file_name="respuesta_janus.txt")
                    
                except Exception as e:
                    st.error(f"Error: {e}")

# --- Pestaña 2: FAQs (Completa) ---
with tab_faq:
    st.header("Preguntas Frecuentes")
    st.markdown("Haz clic en una pregunta para investigar.")
    
    faq_1 = "¿Qué incentivos fiscales hay para energías renovables no convencionales?"
    faq_2 = "¿Cuál es la estructura de sociedad recomendada (S.A.S.) y el capital mínimo?"
    faq_3 = "¿Existen restricciones para repatriar utilidades al exterior?"
    faq_4 = "¿Qué permisos ambientales o licencias se necesitan para operar?"
    faq_5 = "¿Qué garantías de estabilidad jurídica ofrece Colombia?"

    if st.button(faq_1):
        with st.spinner("Analizando..."):
            st.markdown(str(query_engine.query(faq_1)))

    if st.button(faq_2):
        with st.spinner("Analizando..."):
            st.markdown(str(query_engine.query(faq_2)))

    if st.button(faq_3):
        with st.spinner("Analizando..."):
            st.markdown(str(query_engine.query(faq_3)))
            
    if st.button(faq_4):
        with st.spinner("Analizando..."):
            st.markdown(str(query_engine.query(faq_4)))
            
    if st.button(faq_5):
        with st.spinner("Analizando..."):
            st.markdown(str(query_engine.query(faq_5)))                



