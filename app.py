import os
import sys
import logging
import streamlit as st  # <-- ¡NUEVO! Importamos Streamlit
import nest_asyncio     # <-- Importamos el parche de Colab

# Aplicamos el parche de "asyncio" al inicio
nest_asyncio.apply()

# Forzamos el modo CPU para evitar 100% el error de DLL/GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- CONFIGURACIÓN ---
# Usamos los "Secretos" de Streamlit para la clave API
# (NO pongas tu clave directamente en el código)
# Verificamos si la clave existe en los secretos de Streamlit
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    # Si no estamos en Streamlit Cloud, mostramos un error
    st.error("Error: Falta la clave API de Google. Configúrala en los 'Secrets' de Streamlit.")
    st.stop() # Detiene la ejecución si no hay clave

pdf_folder_path = "./ARCHIVOS/"
persist_dir = "./storage"

# --- LÓGICA RAG (CACHEADA) ---

# ¡¡¡ESTA ES LA CLAVE!!!
# @st.cache_resource le dice a Streamlit que "guarde" esta función.
# No recargará el índice cada vez que el usuario haga una pregunta.
@st.cache_resource
def get_query_engine():
    """
    Carga o crea el índice vectorial y devuelve un motor de consulta.
    """
    
    # Configura el "Cerebro" (LLM - Google)
    llm = GoogleGenAI(model="models/gemini-pro-latest")
    
    # ESTA ES LA LÍNEA NUEVA (MULTILINGÜE)
    embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 
    device="cpu" 
)

    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # En Streamlit Cloud, el "storage" no es persistente.
    # Así que siempre crearemos el índice al iniciar la app.
    # (Para 14 PDFs, esto tarda 1-2 minutos y está bien para un prototipo).
    
    print("No se encontró índice local o estamos en la nube. Creando uno nuevo...")
    reader = SimpleDirectoryReader(input_dir=pdf_folder_path, recursive=True)
    documents = reader.load_data()
    print(f"Se cargaron {len(documents)} documentos. Creando índice...")
    
    index = VectorStoreIndex.from_documents(
        documents, 
        show_progress=True, 
        embed_batch_size=100
    )
    
    print("¡Índice creado exitosamente en memoria!")
    
    query_engine = index.as_query_engine(similarity_top_k=5)
    print("¡Sistema listo para responder!")
    return query_engine

# --- INTERFAZ DE USUARIO DE STREAMLIT ---

st.title("🤖 Asistente Virtual VUI (Prototipo)")
st.caption("Respondo preguntas basándome en los 14 PDFs de la Guía Legal 2025.")

# Inicializa el historial de chat (para que recuerde la conversación)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Muestra los mensajes antiguos
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Carga el motor de consulta (esto usará el caché)
try:
    query_engine = get_query_engine()
except Exception as e:
    st.error(f"Error al cargar el índice: {e}")
    st.stop()


# Obtiene la nueva pregunta del usuario
if prompt := st.chat_input("¿Qué quieres saber sobre invertir en Colombia?"):
    
    # Añade la pregunta del usuario al historial y la muestra
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Genera la respuesta del asistente
    with st.chat_message("assistant"):
        with st.spinner("Buscando en los 14 PDFs y contactando a Gemini..."):
            try:
                respuesta = query_engine.query(prompt)
                response_text = str(respuesta)
            except Exception as e:
                # Captura errores de API (como el 503)
                response_text = f"Error al contactar a Gemini: {e}"
        
        st.markdown(response_text)
    
    # Añade la respuesta del asistente al historial

    st.session_state.messages.append({"role": "assistant", "content": response_text})
