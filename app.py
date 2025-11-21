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

# --- 3. API KEYS ---
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("Error Crítico: Falta la clave API de OpenAI en los Secrets.")
    st.stop()

pdf_folder_path = "./ARCHIVOS/"
persist_dir = "./storage"

# --- 4. MOTOR RAG ---
@st.cache_resource
def get_query_engine():
    
    # Cerebro (GPT-4o-mini)
    llm = OpenAI(model="gpt-4o-mini", temperature=0)
    embed_model = OpenAIEmbedding(model="text-embedding-3-large")

    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # Carga
    reader = SimpleDirectoryReader(input_dir=pdf_folder_path, recursive=True)
    documents = reader.load_data()
    
    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
    nodes = node_parser.get_nodes_from_documents(documents)
    
    index = VectorStoreIndex(nodes, show_progress=True)
    
    # Template enfocado en CONTENIDO (No nos preocupamos por el idioma aquí)
    # Dejamos que responda en su idioma natural (Español) para asegurar precisión técnica.
    template_str = (
        "Eres Janus, el Asistente Oficial de la VUI Colombia.\n"
        "Rol: FACILITADOR ESTRATÉGICO.\n"
        "---------------------\n"
        "Contexto:\n{context_str}\n"
        "---------------------\n"
        "Instrucciones:\n"
        "1. REGLA VUE: Para crear empresas, refiere a VUE (Ventanilla Única Empresarial), NO VUCE.\n"
        "2. CONTENIDO: Prioriza pasos prácticos ('CÓMO').\n"
        "3. FORMATO: Usa Markdown (negritas, listas).\n"
        "Pregunta: {query_str}\n"
        "Respuesta (en Español):"
    )
    
    qa_template = PromptTemplate(template_str)
    
    query_engine = index.as_query_engine(
        similarity_top_k=5,
        text_qa_template=qa_template
    ) 
    return query_engine

# --- 5. FUNCIÓN DE TRADUCCIÓN (LA SOLUCIÓN DEFINITIVA) ---
def translate_response(original_response, user_query):
    """
    Toma la respuesta (que seguramente está en Español) y la traduce 
    al idioma de la pregunta del usuario usando una llamada pura al LLM.
    """
    # Si la pregunta ya está en español, no gastamos tiempo traduciendo
    # (Esta es una detección simple, el LLM lo hará mejor)
    
    client = OpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt_traduccion = (
        f"User Query: '{user_query}'\n"
        f"Original Answer: '{original_response}'\n\n"
        "TASK: Analyze the language of the 'User Query'. "
        "Translate the 'Original Answer' into that EXACT same language. "
        "Maintain all Markdown formatting (bolding, lists). "
        "If the query is already in Spanish, just return the Original Answer as is.\n"
        "Translated Answer:"
    )
    
    return client.complete(prompt_traduccion).text

# --- 6. INTERFAZ DE USUARIO ---
st.title("Asistente Janus")
st.caption("Tu guía para la Ventanilla Única de Inversión (VUI).")

tab_chat, tab_faq = st.tabs(["Consultar a Janus 💬", "Preguntas Frecuentes 💡"])

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
        prompt = st.text_area("Escribe tu consulta aquí (Cualquier idioma):", height=100)
        submitted = st.form_submit_button("Enviar Consulta a Janus")

    if submitted and prompt:
        with st.spinner("Janus está analizando y traduciendo..."):
            try:
                # 1. Obtener respuesta técnica (En Español)
                respuesta_raw = query_engine.query(prompt)
                
                # 2. Traducir al idioma del usuario (El paso que asegura el inglés)
                response_text = translate_response(str(respuesta_raw), prompt)
                
                with st.expander("Ver Respuesta de Janus", expanded=True):
                    st.markdown(response_text)
                    
                    # Descarga
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
            txt_resp = str(resp) # Las FAQs están en español, así que no necesitan traducción
            with st.expander("Respuesta", expanded=True):
                st.markdown(txt_resp)
                st.download_button("📥 Descargar TXT", data=f"P:{question}\nR:{txt_resp}", file_name="FAQ.txt")

    if st.button(faq_1): run_faq(faq_1)
    if st.button(faq_2): run_faq(faq_2)
    if st.button(faq_3): run_faq(faq_3)
