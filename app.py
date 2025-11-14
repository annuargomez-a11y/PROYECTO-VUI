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

# --- INTERFAZ DE USUARIO "ASISTENTE JANUS" ---

# --- 1. Cabecera (Sin cambios) ---
st.title("Asistente Janus")
st.caption("Tu guía para la Ventanilla Única de Inversión (VUI).")

# --- ¡NUEVO! Carga el motor ANTES de las pestañas ---
# (Así ambas pestañas pueden usarlo)
try:
    query_engine = get_query_engine()
except Exception as e:
    st.error(f"Error al cargar el motor del asistente: {e}")
    st.stop()

# --- 2. Pestañas de Funciones (¡MODIFICADAS!) ---
tab_chat, tab_faq = st.tabs(["Conversar con Janus 💬", "Preguntas Frecuentes 💡"])

# --- Pestaña 1: El Chat (¡SIMPLIFICADA!) ---
with tab_chat:
    
    # Inicializa el saludo de Janus
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "¡Hola! Soy Janus, tu asistente virtual. ¡Estoy aquí para guiarte en tu Inversión Directa en Colombia!"}
        ]

    # ¡INTERFAZ CORREGIDA! (Contenedor con altura)
    chat_container = st.container(height=500) 

    # Muestra los mensajes antiguos DENTRO del contenedor
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Caja de chat (Fija al fondo de la pestaña)
    if prompt := st.chat_input("Pregúntale a Janus sobre la Guía Legal..."):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Muestra el prompt del usuario
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

        # Genera y muestra la respuesta
        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Consultando la Guía Legal y contactando a Gemini..."):
                    try:
                        respuesta = query_engine.query(prompt)
                        response_text = str(respuesta)
                    except Exception as e:
                        response_text = f"Error al contactar a Gemini: {e}. Por favor, espera unos segundos e inténtalo de nuevo."
                
                st.markdown(response_text)
        
        st.session_state.messages.append({"role": "assistant", "content": response_text})

# --- Pestaña 2: Información (¡NUEVA!) ---
with tab_faq:
    st.header("Preguntas Frecuentes (FAQs)")
    st.markdown("Haz clic en una pregunta para que Janus la investigue por ti. La respuesta aparecerá en la pestaña 'Conversar con Janus'.")
    st.divider()

    # --- Definimos las 5 preguntas clave ---
    faq_1 = "¿Qué incentivos fiscales o tributarios específicos ofrece el gobierno para la Inversión Extranjera Directa en energías renovables no convencionales?"
    faq_2 = "¿Cuál es la estructura de sociedad más recomendada para una subsidiaria extranjera en Colombia (como una S.A.S.), y cuáles son los requisitos de capital mínimo para constituirla?"
    faq_3 = "¿Existen restricciones cambiarias o requisitos de registro ante el Banco de la República para traer la inversión inicial y repatriar las utilidades (dividendos)?"
    faq_4 = "¿Qué permisos o licencias clave (ambientales, regulatorias de la CREG, o de conexión) se necesitan para construir y operar un parque de generación de energía renovable?"
    faq_5 = "¿Qué protecciones legales o tratados internacionales (como Acuerdos de Estabilidad Jurídica) ofrece Colombia para proteger mi inversión?"

    # --- Lógica de Botones ---
    
    def handle_faq_click(question_text):
        """Función para manejar el clic en un botón de FAQ."""
        with st.spinner("Janus está consultando la Guía..."):
            try:
                respuesta = query_engine.query(question_text)
                # Añade la Q&A al historial del chat principal
                st.session_state.messages.append({"role": "user", "content": question_text})
                st.session_state.messages.append({"role": "assistant", "content": str(respuesta)})
                st.success("¡Respuesta lista! Revisa la pestaña 'Conversar con Janus' 💬")
            except Exception as e:
                st.error(f"Error al contactar a Gemini: {e}")

    # --- Muestra los botones ---
    if st.button(faq_1):
        handle_faq_click(faq_1)
        
    if st.button(faq_2):
        handle_faq_click(faq_2)

    if st.button(faq_3):
        handle_faq_click(faq_3)
        
    if st.button(faq_4):
        handle_faq_click(faq_4)
        
    if st.button(faq_5):
        handle_faq_click(faq_5)
