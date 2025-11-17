import os
import sys
import logging
import streamlit as st
import nest_asyncio

# --- PARCHES CRÍTICOS ---
nest_asyncio.apply()
# Ya no necesitamos forzar la CPU porque OpenAI corre en la nube, 
# pero lo dejamos por seguridad para otras librerías.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.core.node_parser import SentenceSplitter 
from llama_index.llms.google_genai import GoogleGenAI

# --- ¡CAMBIO CLAVE! IMPORTAMOS OPENAI ---
from llama_index.embeddings.openai import OpenAIEmbedding

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Asistente Janus (VUI)", page_icon="🗝️", layout="centered")

# --- GESTIÓN DE CLAVES API ---
# 1. Google (Para el "Cerebro" que responde)
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Falta la clave de Google.")
    st.stop()

# 2. OpenAI (Para el "Traductor" que lee con precisión)
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("Falta la clave de OpenAI. Configúrala en los Secrets.")
    st.stop()

pdf_folder_path = "./ARCHIVOS/"
persist_dir = "./storage" 

# --- FUNCIÓN DEL MOTOR RAG (HÍBRIDO: GOOGLE + OPENAI) ---
@st.cache_resource
def get_query_engine():
    
    # 1. CEREBRO: Google Gemini (Sigue siendo gratis y bueno respondiendo)
    llm = GoogleGenAI(model="models/gemini-pro-latest")
    
    # 2. TRADUCTOR: OpenAI (Aquí es donde invertimos los centavos para CALIDAD)
    # Usamos "text-embedding-3-large", el modelo más preciso.
    embed_model = OpenAIEmbedding(model="text-embedding-3-large")

    Settings.llm = llm
    Settings.embed_model = embed_model
    
    print("--- INICIANDO PROCESO HÍBRIDO ---")
    
    reader = SimpleDirectoryReader(input_dir=pdf_folder_path, recursive=True)
    documents = reader.load_data()
    print(f"Documentos cargados: {len(documents)}")
    
    # Corte Inteligente
    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
    nodes = node_parser.get_nodes_from_documents(documents)
    
    # Creación del Índice (Aquí es donde OpenAI procesa tus PDFs)
    print("Enviando textos a OpenAI para indexar...")
    index = VectorStoreIndex(nodes, show_progress=True)
    
    print("¡Índice OpenAI creado en memoria!")
    
    # Buscamos los 5 trozos más relevantes (Mayor contexto)
    query_engine = index.as_query_engine(similarity_top_k=5) 
    return query_engine

# --- INTERFAZ DE USUARIO ---

st.title("Asistente Janus 2.0 🧠")
st.caption("Potenciado por Gemini (Razonamiento) + OpenAI (Búsqueda de Alta Precisión).")

tab_chat, tab_faq = st.tabs(["Consultar a Janus 💬", "Preguntas Frecuentes 💡"])

with tab_chat:
    st.header("Haz tu consulta")
    
    try:
        query_engine = get_query_engine()
    except Exception as e:
        st.error(f"Error al cargar el motor: {e}")
        st.stop()

    with st.form("query_form"):
        prompt = st.text_area("Escribe tu consulta aquí:", height=150)
        submitted = st.form_submit_button("Enviar Consulta a Janus")

    if submitted and prompt:
        with st.spinner("Analizando con precisión quirúrgica..."):
            try:
                respuesta = query_engine.query(prompt)
                response_text = str(respuesta)
                with st.expander("Ver Respuesta", expanded=True):
                    st.markdown(response_text)
                    st.download_button("📥 Guardar", data=response_text, file_name="respuesta_janus.txt")
            except Exception as e:
                st.error(f"Error: {e}")

# --- Pestaña 2: Preguntas Frecuentes (COMPLETA) ---
with tab_faq:
    st.header("Preguntas Frecuentes")
    st.markdown("Haz clic en una pregunta para que Janus analice los documentos con precisión.")
    
    # 1. Definir las preguntas
    faq_1 = "¿Qué incentivos fiscales hay para energías renovables no convencionales?"
    faq_2 = "¿Cuál es la estructura de sociedad recomendada (S.A.S.) y el capital mínimo?"
    faq_3 = "¿Existen restricciones para repatriar utilidades al exterior?"
    faq_4 = "¿Qué permisos ambientales o licencias se necesitan para operar?"
    faq_5 = "¿Qué garantías de estabilidad jurídica ofrece Colombia?"

    # 2. Crear los botones
    if st.button(faq_1):
        with st.spinner("Analizando incentivos..."):
            try:
                respuesta = query_engine.query(faq_1)
                with st.expander("Ver Respuesta", expanded=True):
                    st.markdown(str(respuesta))
            except Exception as e:
                st.error(f"Error: {e}")

    if st.button(faq_2):
        with st.spinner("Analizando estructura societaria..."):
            try:
                respuesta = query_engine.query(faq_2)
                with st.expander("Ver Respuesta", expanded=True):
                    st.markdown(str(respuesta))
            except Exception as e:
                st.error(f"Error: {e}")

    if st.button(faq_3):
        with st.spinner("Analizando régimen cambiario..."):
            try:
                respuesta = query_engine.query(faq_3)
                with st.expander("Ver Respuesta", expanded=True):
                    st.markdown(str(respuesta))
            except Exception as e:
                st.error(f"Error: {e}")

    if st.button(faq_4):
        with st.spinner("Buscando licencias..."):
            try:
                respuesta = query_engine.query(faq_4)
                with st.expander("Ver Respuesta", expanded=True):
                    st.markdown(str(respuesta))
            except Exception as e:
                st.error(f"Error: {e}")

    if st.button(faq_5):
        with st.spinner("Verificando garantías legales..."):
            try:
                respuesta = query_engine.query(faq_5)
                with st.expander("Ver Respuesta", expanded=True):
                    st.markdown(str(respuesta))
            except Exception as e:
                st.error(f"Error: {e}")
                

