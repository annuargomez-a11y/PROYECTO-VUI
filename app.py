import os
import sys
import logging
import streamlit as st
import nest_asyncio

# --- PARCHES CRÍTICOS (¡No tocar!) ---
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

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Asistente Janus (VUI)",
    page_icon="🗝️",
    layout="centered" 
)

# --- CONFIGURACIÓN DE API ---
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Error: Falta la clave API de Google. Configúrala en los 'Secrets' de Streamlit.")
    st.stop() 

pdf_folder_path = "./ARCHIVOS/"
persist_dir = "./storage" # (Streamlit Cloud reconstruye esto, así que no es persistente)

# --- FUNCIÓN DEL MOTOR RAG (¡ACTUALIZADA!) ---
@st.cache_resource
def get_query_engine():
    """
    Carga o crea el índice vectorial y devuelve un motor de consulta.
    """
    
    # Configura el "Cerebro" (LLM - Google)
    llm = GoogleGenAI(model="models/gemini-pro-latest")
    
    # --- ¡VOLVEMOS AL "TRADUCTOR" LIGERO! ---
    # Este modelo SÍ cabe en la memoria gratuita de Streamlit.
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 
        device="cpu" 
    )

    Settings.llm = llm
    Settings.embed_model = embed_model
    
    print("Creando índice desde cero (ejecución en la nube)...")
    
    reader = SimpleDirectoryReader(input_dir=pdf_folder_path, recursive=True)
    documents = reader.load_data()
    print(f"Se cargaron {len(documents)} documentos.")
    
    # Usamos el "Corte Inteligente"
    print("Analizando y cortando los documentos en párrafos inteligentes...")
    node_parser = SentenceSplitter(
        chunk_size=1024,
        chunk_overlap=100
    )
    nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)
    print(f"Se crearon {len(nodes)} trozos (nodos) de texto inteligente.")
    
    print("Creando índice (esto puede tardar unos minutos)...")
    index = VectorStoreIndex(
        nodes, 
        show_progress=True, 
        embed_batch_size=100 # Lo mantenemos en lotes
    )
    
    print("¡Índice creado exitosamente en memoria!")
    query_engine = index.as_query_engine(similarity_top_k=3) 
    print("¡Sistema listo para responder!")
    return query_engine

# --- INTERFAZ DE USUARIO "ASISTENTE JANUS" ---

# --- 1. Cabecera (Sin cambios) ---
st.title("Asistente Janus")
st.caption("Tu guía para la Ventanilla Única de Inversión (VUI).")

# --- 2. Pestañas de Funciones (Sin cambios) ---
tab_chat, tab_acerca_de = st.tabs(["Conversar con Janus 💬", "Acerca de este Prototipo ℹ️"])

# --- Pestaña 1: El Chat (¡ACTUALIZADA!) ---
with tab_chat:
    
    # Inicializa el saludo de Janus
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "¡Hola! Soy Janus, tu asistente virtual. ¡Estoy aquí para guiarte en tu Inversión Directa en Colombia!"}
        ]

    # --- ¡INTERFAZ CORREGIDA! ---
    # Creamos un contenedor con altura fija para el historial
    chat_container = st.container(height=500) # Puedes ajustar el 500

    # Muestra los mensajes antiguos DENTRO del contenedor
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Carga el motor de consulta
    try:
        query_engine = get_query_engine()
    except Exception as e:
        st.error(f"Error al cargar el motor del asistente: {e}")
        st.stop()

    # Caja de chat (Queda FUERA del contenedor, fija al fondo de la pestaña)
    if prompt := st.chat_input("Pregúntale a Janus sobre la Guía Legal..."):
        
        # Añade el prompt al historial de estado
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Muestra el prompt del usuario DENTRO del contenedor
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

        # Genera y muestra la respuesta DENTRO del contenedor
        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Consultando la Guía Legal y contactando a Gemini..."):
                    try:
                        respuesta = query_engine.query(prompt)
                        response_text = str(respuesta)
                    except Exception as e:
                        response_text = f"Error al contactar a Gemini: {e}. Por favor, espera unos segundos e inténtalo de nuevo."
                
                st.markdown(response_text)
        
        # Añade la respuesta al historial de estado
        st.session_state.messages.append({"role": "assistant", "content": response_text})

# --- Pestaña 2: Información (Sin cambios) ---
with tab_acerca_de:
    st.header("Sobre este Prototipo")
    # ... (El resto del código de la pestaña 2) ...
    st.markdown("""
    Este es un prototipo RAG (Generación Aumentada por RecuperACIÓN)
    con "Corte Inteligente" (Smart Chunking).
    
    **Tecnologías utilizadas:**
    * **Interfaz:** Streamlit
    * **Orquestador RAG:** LlamaIndex
    * **Cerebro (LLM):** Google Gemini (`gemini-pro-latest`)
    * **Traductor (Embedding):** `paraphrase-multilingual-MiniLM-L12-v2` (Local/CPU)
* **Base de Conocimiento:** 14 PDFs de la Guía Legal 2025.
    """)
    st.warning("El arranque inicial de esta aplicación tarda 2-3 minutos mientras se crea el índice de los PDFs.")
