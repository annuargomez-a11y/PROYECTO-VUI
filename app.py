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

# --- FUNCIÓN DEL MOTOR RAG (Sin cambios) ---
@st.cache_resource
def get_query_engine():
    """
    Carga o crea el índice vectorial y devuelve un motor de consulta.
    """
    
    # Configura el "Cerebro" (LLM - Google)
    llm = GoogleGenAI(model="models/gemini-pro-latest")
    
    # Volvemos al "Traductor" ligero que SÍ cabe en la memoria.
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
        embed_batch_size=100
    )
    
    print("¡Índice creado exitosamente en memoria!")
    query_engine = index.as_query_engine(similarity_top_k=3) 
    print("¡Sistema listo para responder!")
    return query_engine

# --- INTERFAZ DE USUARIO "ASISTENTE JANUS" (¡DISEÑO FORMULARIO!) ---

# --- 1. Cabecera (Sin cambios) ---
st.title("Asistente Janus")
st.caption("Tu guía para la Ventanilla Única de Inversión (VUI).")

# --- 2. Pestañas de Funciones (Sin cambios) ---
tab_chat, tab_acerca_de = st.tabs(["Consultar a Janus 💬", "Acerca de este Prototipo ℹ️"])

# --- Pestaña 1: El Chat (¡AHORA ES UN FORMULARIO!) ---
with tab_chat:
    
    st.header("Haz tu consulta")
    st.markdown("¡Hola! Soy Janus. Escribe tu pregunta sobre la Guía Legal 2025 y te ayudaré a encontrar la respuesta.")

    # Carga el motor de consulta
    try:
        query_engine = get_query_engine()
    except Exception as e:
        st.error(f"Error al cargar el motor del asistente: {e}")
        st.stop()

    # --- ¡CAMBIO DE INTERFAZ! ---
    # Usamos un Formulario para agrupar la entrada y el botón
    with st.form("query_form"):
        # 1. La caja de entrada (ya no es st.chat_input)
        prompt = st.text_area("Pregúntale a Janus:", height=150)
        
        # 2. El botón de envío
        submitted = st.form_submit_button("Enviar Consulta")

    # 3. La caja de respuesta (aparece solo si se envía)
    if submitted:
        if not prompt:
            st.warning("Por favor, escribe una pregunta.")
        else:
            with st.spinner("Consultando la Guía Legal y contactando a Gemini..."):
                try:
                    respuesta = query_engine.query(prompt)
                    response_text = str(respuesta)
                    
                    st.subheader("Respuesta de Janus:")
                    st.success(response_text) # st.success pone un fondo verde
                    
                except Exception as e:
                    response_text = f"Error al contactar a Gemini: {e}. Por favor, espera unos segundos e inténtalo de nuevo."
                    st.error(response_text)

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
