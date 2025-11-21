import streamlit as st
import nest_asyncio
import os
from datetime import datetime
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext, 
    load_index_from_storage, 
    Settings
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# --- 1. CONFIGURACIÓN INICIAL ---
nest_asyncio.apply()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

st.set_page_config(page_title="Asistente Janus (VUI)", page_icon="🗝️", layout="centered")

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("Error Crítico: Falta la clave API de OpenAI en los Secrets.")
    st.stop()

pdf_folder_path = "./ARCHIVOS/"
persist_dir = "./storage"

# --- 2. MOTOR DE TRADUCCIÓN (LA SOLUCIÓN REAL) ---
def force_translation(text, user_query):
    """
    Toma la respuesta técnica (que sale en español) y la traduce
    al idioma en que el usuario hizo la pregunta.
    """
    client = OpenAI(model="gpt-4o-mini", temperature=0)
    
    # Esta instrucción es directa y no tiene el "ruido" de los PDFs
    prompt = (
        f"User Query: '{user_query}'\n"
        f"Original Answer: '{text}'\n\n"
        "INSTRUCTION: Detect the language of the 'User Query'. "
        "Translate the 'Original Answer' into that EXACT language. "
        "Maintain the Markdown formatting (bolding, lists). "
        "If the query is already in Spanish, just return the Original Answer as is."
    )
    
    # Llama al modelo solo para traducir
    response = client.complete(prompt)
    return response.text

# --- 3. MOTOR RAG (CEREBRO TÉCNICO) ---
@st.cache_resource
def get_query_engine():
    # Configuramos el modelo para que sea un experto técnico en español
    llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
    embed_model = OpenAIEmbedding(model="text-embedding-3-large")

    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # Carga de documentos
    if not os.path.exists(persist_dir):
        reader = SimpleDirectoryReader(input_dir=pdf_folder_path, recursive=True)
        documents = reader.load_data()
        node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
        nodes = node_parser.get_nodes_from_documents(documents)
        index = VectorStoreIndex(nodes, show_progress=True)
        # index.storage_context.persist(persist_dir) # Opcional: guardar índice
    else:
        # Si tienes persistencia activada, descomenta esto:
        # storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        # index = load_index_from_storage(storage_context)
        
        # Por ahora, reconstruimos para asegurar frescura sin errores de caché
        reader = SimpleDirectoryReader(input_dir=pdf_folder_path, recursive=True)
        documents = reader.load_data()
        index = VectorStoreIndex.from_documents(documents)

    # Prompt del sistema enfocado en CALIDAD de respuesta (en Español)
    system_prompt = (
        "Eres Janus, el Asistente Oficial de la VUI Colombia. "
        "Tu rol es FACILITADOR ESTRATÉGICO. "
        "REGLAS: "
        "1. Si preguntan por crear empresa/S.A.S, refiere a la VUE (Ventanilla Única Empresarial), nunca VUCE. "
        "2. Prioriza pasos prácticos ('CÓMO'). "
        "3. Si preguntan por proyectos, usa las Fichas. "
        "4. Genera respuestas completas y detalladas en Markdown."
    )
    
    # Inyectamos el prompt al LLM
    llm.system_prompt = system_prompt
    
    return index.as_query_engine(similarity_top_k=5)

# --- 4. INTERFAZ ---
st.title("Asistente Janus")
st.caption("Tu guía para la Ventanilla Única de Inversión (VUI).")

tab_chat, tab_faq = st.tabs(["Consultar a Janus 💬", "Preguntas Frecuentes 💡"])

try:
    query_engine = get_query_engine()
except Exception as e:
    st.error(f"Error al cargar el motor: {e}")
    st.stop()

# Pestaña Chat
with tab_chat:
    st.header("Haz tu consulta")
    st.markdown("¡Hola! Soy Janus. Estoy aquí para guiarte en tu Inversión Directa en Colombia.")

    with st.form("query_form"):
        prompt = st.text_area("Escribe tu consulta aquí (Cualquier idioma):", height=100)
        submitted = st.form_submit_button("Enviar Consulta")

    if submitted and prompt:
        with st.spinner("Janus está analizando y traduciendo..."):
            try:
                # PASO 1: Obtener respuesta técnica (Saldrá en Español por los PDFs)
                respuesta_raw = query_engine.query(prompt)
                
                # PASO 2: Traducir (Aquí forzamos el inglés)
                response_final = force_translation(str(respuesta_raw), prompt)
                
                with st.expander("Ver Respuesta de Janus", expanded=True):
                    st.markdown(response_final)
                    
                    # Descarga
                    ahora = datetime.now()
                    nombre_file = f"Janus.Answer.{ahora.strftime('%Y%m%d.%H%M')}.txt"
                    contenido = f"PREGUNTA:\n{prompt}\n\nRESPUESTA:\n{response_final}"
                    st.download_button("📥 Guardar Respuesta (TXT)", data=contenido, file_name=nombre_file, mime="text/plain")
                    
            except Exception as e:
                st.error(f"Error: {e}")

# Pestaña FAQs
with tab_faq:
    st.header("Preguntas Frecuentes")
    faq_1 = "¿Qué incentivos fiscales hay para energías renovables?"
    faq_2 = "¿Cuál es la estructura de sociedad recomendada (S.A.S.) y capital mínimo?"
    faq_3 = "¿Existen restricciones para repatriar utilidades al exterior?"

    def run_faq(q):
        with st.spinner("Consultando..."):
            resp = query_engine.query(q)
            st.markdown(str(resp)) # FAQs siempre en español

    if st.button(faq_1): run_faq(faq_1)
    if st.button(faq_2): run_faq(faq_2)
    if st.button(faq_3): run_faq(faq_3)
