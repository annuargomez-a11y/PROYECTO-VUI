import os
import sys
import logging
import streamlit as st
import nest_asyncio
from datetime import datetime # <-- NUEVO: Para manejar fechas

# --- PARCHES CRÍTICOS ---
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

# --- MOTOR RAG ---
@st.cache_resource
def get_query_engine():
    llm = OpenAI(model="gpt-4o-mini", temperature=0.2)
    embed_model = OpenAIEmbedding(model="text-embedding-3-large")

    Settings.llm = llm
    Settings.embed_model = embed_model
    
    print("--- INICIANDO MOTOR JANUS ---")
    reader = SimpleDirectoryReader(input_dir=pdf_folder_path, recursive=True)
    documents = reader.load_data()
    
    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
    nodes = node_parser.get_nodes_from_documents(documents)
    
    index = VectorStoreIndex(nodes, show_progress=True)
    
    template_str = (
        "Eres Janus, el Asistente Oficial de la Ventanilla Única de Inversión (VUI) de Colombia.\n"
        "Tu rol es actuar como un FACILITADOR ESTRATÉGICO.\n"
        "---------------------\n"
        "Contexto Normativo:\n{context_str}\n"
        "---------------------\n"
        "Instrucciones:\n"
        "1. Prioriza el FLUJO DEL PROCESO y los REQUISITOS. Evita instrucciones triviales de interfaz (como 'haz clic aquí') a menos que el usuario pida ayuda técnica específica."
        "2. Usa formato Markdown (negritas, listas, tablas) para que se vea bien en pantalla.\n"
        "3. Si la respuesta es breve, explica las implicaciones para el inversionista.\n"
        "4. Responde siempre en el mismo idioma de la pregunta.\n"
        "Pregunta: {query_str}\n\n"
        "Respuesta:"
    )
    
    qa_template = PromptTemplate(template_str)
    
    query_engine = index.as_query_engine(
        similarity_top_k=5, 
        text_qa_template=qa_template
    ) 
    return query_engine

# --- INTERFAZ ---
st.title("Asistente Janus")
st.caption("Tu guía para la Ventanilla Única de Inversión (VUI).")

tab_chat, tab_faq = st.tabs(["Consultar a Janus 💬", "Preguntas Frecuentes 💡"])

try:
    query_engine = get_query_engine()
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# --- Pestaña 1: Chat ---
with tab_chat:
    st.header("Haz tu consulta")
    st.markdown("¡Hola! Soy Janus. Estoy aquí para guiarte en tu Inversión Directa en Colombia.")

    with st.form("query_form"):
        prompt = st.text_area("Escribe tu consulta aquí:", height=100)
        submitted = st.form_submit_button("Enviar Consulta a Janus")

    if submitted and prompt:
        with st.spinner("Janus está analizando la normativa..."):
            try:
                respuesta = query_engine.query(prompt)
                response_text = str(respuesta)
                
                with st.expander("Ver Respuesta de Janus", expanded=True):
                    st.markdown(response_text)
                    
                    # --- LÓGICA DE FECHA Y HORA ---
                    ahora = datetime.now()
                    fecha_hora_texto = ahora.strftime("%Y-%m-%d %H:%M:%S") # Formato legible
                    fecha_hora_archivo = ahora.strftime("%Y%m%d.%H%M")    # Formato aammdd.hhmm
                    
                    nombre_archivo = f"Janus.Answer.{fecha_hora_archivo}.txt"

                    # --- FORMATO MEJORADO PARA EL ARCHIVO TXT ---
                    contenido_txt = f"""================================================================================
REPORTE DE CONSULTA - ASISTENTE VUI JANUS
FECHA Y HORA: {fecha_hora_texto}
================================================================================

PREGUNTA DEL INVERSIONISTA:
{prompt}

--------------------------------------------------------------------------------

RESPUESTA DE JANUS:
{response_text}

================================================================================
Generado por Inteligencia Artificial - Ventanilla Única de Inversión
"""
                    # Descarga
                    st.download_button(
                        label="📥 Guardar Respuesta (TXT)",
                        data=contenido_txt,
                        file_name=nombre_archivo,
                        mime="text/plain"
                    )
            except Exception as e:
                st.error(f"Error: {e}")

# --- Pestaña 2: FAQs ---
with tab_faq:
    st.header("Preguntas Frecuentes")
    
    faq_1 = "¿Qué incentivos fiscales hay para energías renovables no convencionales?"
    faq_2 = "¿Cuál es la estructura de sociedad recomendada (S.A.S.) y capital mínimo?"
    faq_3 = "¿Existen restricciones para repatriar utilidades al exterior?"
    faq_4 = "¿Qué permisos ambientales o licencias se necesitan para operar?"
    faq_5 = "¿Qué garantías de estabilidad jurídica ofrece Colombia?"

    def run_faq(question):
        with st.spinner("Consultando..."):
            resp = query_engine.query(question)
            txt_resp = str(resp)
            
            with st.expander("Respuesta", expanded=True):
                st.markdown(txt_resp)
                
                # --- LÓGICA DE FECHA Y HORA (También para FAQs) ---
                ahora = datetime.now()
                fecha_hora_texto = ahora.strftime("%Y-%m-%d %H:%M:%S")
                fecha_hora_archivo = ahora.strftime("%Y%m%d.%H%M")
                nombre_archivo = f"Janus.FAQ.{fecha_hora_archivo}.txt"

                contenido_txt_faq = f"""================================================================================
REPORTE DE PREGUNTA FRECUENTE - ASISTENTE VUI JANUS
FECHA Y HORA: {fecha_hora_texto}
================================================================================

PREGUNTA SELECCIONADA:
{question}

--------------------------------------------------------------------------------

RESPUESTA DE JANUS:
{txt_resp}

================================================================================
Generado por Inteligencia Artificial - Ventanilla Única de Inversión
"""
                st.download_button("📥 Descargar TXT", data=contenido_txt_faq, file_name=nombre_archivo, mime="text/plain")

    if st.button(faq_1): run_faq(faq_1)
    if st.button(faq_2): run_faq(faq_2)
    if st.button(faq_3): run_faq(faq_3)
    if st.button(faq_4): run_faq(faq_4)
    if st.button(faq_5): run_faq(faq_5)

# 3. Personalidad de Janus (CON REGLA DE BLOQUEO VUCE)
    template_str = (
        "Eres Janus, el Asistente Oficial de la Ventanilla Única de Inversión (VUI) de Colombia.\n"
        "Tu rol es actuar como un FACILITADOR ESTRATÉGICO.\n"
        "---------------------\n"
        "Contexto Normativo:\n{context_str}\n"
        "---------------------\n"
        "Instrucciones DE OBLIGATORIO CUMPLIMIENTO:\n"
        "1. REGLA DE ORO: Si la pregunta es sobre 'Crear Empresa', 'Constitución de Sociedad' o 'S.A.S.', la ÚNICA plataforma válida es la VUE (Ventanilla Única Empresarial).\n"
        "2. PROHIBICIÓN: En procesos de creación de empresa, ESTÁ PROHIBIDO mencionar la VUCE (Ventanilla Única de Comercio Exterior). La VUCE es solo para importaciones.\n"
        "3. Prioriza el 'CÓMO' (pasos prácticos del manual de la VUE) sobre el 'QUÉ' (teoría legal).\n"
        "4. Usa formato Markdown (negritas, listas).\n"
        "5. Responde en el idioma de la pregunta.\n"
        "Pregunta: {query_str}\n\n"
        "Respuesta:"
    )

