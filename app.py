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
    Settings,
    PromptTemplate # <-- ¡NUEVA IMPORTACIÓN CLAVE!
)
from llama_index.core.node_parser import SentenceSplitter 
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Asistente Janus (VUI)",
    page_icon="🗝️",
    layout="centered" 
)

# --- API KEYS ---
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("Error: Falta la clave API de OpenAI en los Secrets.")
    st.stop() 

pdf_folder_path = "./ARCHIVOS/"
persist_dir = "./storage"

# --- MOTOR RAG CON PERSONALIDAD FORZADA ---
@st.cache_resource
def get_query_engine():
    
    # 1. Modelos
    llm = OpenAI(model="gpt-4o-mini", temperature=0.2)
    embed_model = OpenAIEmbedding(model="text-embedding-3-large")

    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # 2. Carga / Indexación
    print("--- INICIANDO MOTOR JANUS ---")
    reader = SimpleDirectoryReader(input_dir=pdf_folder_path, recursive=True)
    documents = reader.load_data()
    
    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
    nodes = node_parser.get_nodes_from_documents(documents)
    
    print("Indexando...")
    index = VectorStoreIndex(nodes, show_progress=True)
    print("¡Índice creado!")
    
    # 3. DEFINIR LA PERSONALIDAD (EL PROMPT TEMPLATE)
    # Este es el "guion" exacto que Janus debe seguir
    # 3. DEFINIR LA PERSONALIDAD (EL PROMPT TEMPLATE)
    template_str = (
        "Eres Janus, un experto y amable asesor de inversión extranjera en Colombia (VUI).\n"
        "Tu misión es guiar a los inversionistas con respuestas claras, completas y estratégicas.\n"
        "---------------------\n"
        "Contexto de la Guía Legal:\n"
        "{context_str}\n"
        "---------------------\n"
        "Instrucciones:\n"
        "1. Responde la pregunta del usuario basándote EXCLUSIVAMENTE en el contexto anterior.\n"
        "2. Si la respuesta es técnica o breve, NO te detengas ahí. EXPLICA qué significa eso para el inversionista.\n"
        "3. Usa un tono profesional, cercano y estructurado.\n"
        "4. Si el contexto no tiene la información, dilo honestamente.\n"
        "5. IDIOMA: Detecta el idioma de la pregunta del usuario y responde EN ESE MISMO IDIOMA (Ej: Si preguntan en Inglés, responde en Inglés).\n\n"
        "Pregunta del Inversionista: {query_str}\n\n"
        "Respuesta de Janus:"
    )
    
    qa_template = PromptTemplate(template_str)

    # 4. Crear el motor inyectando el Template
    query_engine = index.as_query_engine(
        similarity_top_k=5, 
        text_qa_template=qa_template # <-- ¡AQUÍ APLICAMOS LA PERSONALIDAD!
    )
    
    return query_engine

# --- INTERFAZ ---

st.title("Asistente Janus")
st.caption("Tu guía para la Ventanilla Única de Inversión (VUI).")

tab_chat, tab_faq = st.tabs(["Consultar a Janus 💬", "Preguntas Frecuentes 💡"])

with tab_chat:
    st.header("Haz tu consulta")
    st.markdown("¡Hola! Soy Janus. Estoy conectado a GPT-4o para darte asesoría detallada.")

    try:
        query_engine = get_query_engine()
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

    with st.form("query_form"):
        prompt = st.text_area("Escribe tu consulta aquí:", height=150)
        submitted = st.form_submit_button("Enviar Consulta a Janus")

    if submitted:
        if not prompt:
            st.warning("Escribe una pregunta.")
        else:
            with st.spinner("Janus está analizando la Guía Legal..."):
                try:
                    respuesta = query_engine.query(prompt)
                    response_text = str(respuesta)
                    
                    with st.expander("Ver Respuesta de Janus", expanded=True):
                        st.markdown(response_text)
                        st.download_button("📥 Guardar Respuesta", data=response_text, file_name="respuesta_janus.txt")
                except Exception as e:
                    st.error(f"Error: {e}")

with tab_faq:
    st.header("Preguntas Frecuentes")
    faq_1 = "¿Qué incentivos fiscales hay para energías renovables?"
    faq_2 = "¿Cuál es la estructura de sociedad recomendada (S.A.S.) y el capital mínimo?"
    faq_3 = "¿Existen restricciones para repatriar utilidades?"
    faq_4 = "¿Qué permisos ambientales se necesitan?"
    faq_5 = "¿Qué garantías de estabilidad jurídica existen?"

    def run_faq(question):
        with st.spinner("Consultando..."):
            resp = query_engine.query(question)
            st.markdown(str(resp))

    if st.button(faq_1): run_faq(faq_1)
    if st.button(faq_2): run_faq(faq_2)
    if st.button(faq_3): run_faq(faq_3)
    if st.button(faq_4): run_faq(faq_4)
    if st.button(faq_5): run_faq(faq_5)

