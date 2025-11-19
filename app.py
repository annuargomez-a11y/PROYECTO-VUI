import os
import sys
import logging
import streamlit as st
import nest_asyncio
import re 
from fpdf import FPDF # <-- Se mantiene, aunque ya no se usa la funcion
from io import BytesIO

# --- PARCHES ---
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
st.set_page_config(page_title="Asistente Janus (VUI)", page_icon="🗝️", layout="centered")

# --- API KEYS ---
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("Error: Falta la clave API de OpenAI.")
    st.stop() 

pdf_folder_path = "./ARCHIVOS/"
persist_dir = "./storage"

# --- FUNCIÓN DE LIMPIEZA (La dejamos simple) ---
# Se mantiene, aunque ya no la usamos para el PDF, pero es buena práctica tenerla
def clean_text_for_pdf(text):
    return text

# --- FUNCIÓN PDF (Se queda vacía) ---
def create_pdf(text):
    return None

# --- MOTOR RAG ---
@st.cache_resource
def get_query_engine():
    # El motor principal usará un template que PIDE TABLAS para la PANTALLA (esto funciona bien)
    llm = OpenAI(model="gpt-4o-mini", temperature=0.2)
    embed_model = OpenAIEmbedding(model="text-embedding-3-large")
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    print("--- INICIANDO MOTOR ---")
    reader = SimpleDirectoryReader(input_dir=pdf_folder_path, recursive=True)
    documents = reader.load_data()
    
    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
    nodes = node_parser.get_nodes_from_documents(documents)
    
    index = VectorStoreIndex(nodes, show_progress=True)
    
    # Template para la PANTALLA (Personalidad: FACILITADOR ESTRATÉGICO)
    template_str_markdown = (
        "Eres Janus, el Asistente Oficial de la Ventanilla Única de Inversión (VUI) de Colombia.\n"
        "Tu rol no es solo citar leyes, sino actuar como un FACILITADOR ESTRATÉGICO para el inversionista.\n"
        "---------------------\n"
        "Contexto Normativo:\n{context_str}\n"
        "---------------------\n"
        "Tus Instrucciones de Comportamiento:\n"
        "1. ENFOQUE EN EL 'CÓMO': Prioriza explicar los pasos, requisitos prácticos y procesos sobre la teoría legal pura.\n"
        "2. TONO: Profesional, cercano y resolutivo. Usa un lenguaje claro de negocios, evitando la jerga legal innecesaria.\n"
        "3. ESTRUCTURA: Usa Markdown. Si hay pasos, usa listas numeradas. Si hay opciones, usa viñetas o tablas.\n"
        "4. TRANSPARENCIA: Si el documento no explica el procedimiento exacto, indícalo y sugiere contactar a la entidad responsable.\n"
        "5. IDIOMA: Responde siempre en el mismo idioma de la pregunta.\n\n"
        "Pregunta del Inversionista: {query_str}\n\n"
        "Respuesta de Janus:"
    )
    qa_template_markdown = PromptTemplate(template_str_markdown)
    
    # Creamos un solo motor de consulta
    query_engine = index.as_query_engine(similarity_top_k=5, text_qa_template=qa_template_markdown)
    return query_engine

# --- INTERFAZ ---
st.title("Asistente Janus")
st.caption("Tu guía para la Ventanilla Única de Inversión (VUI).")

tab_chat, tab_faq = st.tabs(["Consultar a Janus 💬", "Preguntas Frecuentes 💡"])

# --- Ejecución del Motor ---
try:
    query_engine = get_query_engine()
except Exception as e:
    st.error(f"Error al cargar el motor: {e}")
    st.stop()


# --- Pestaña 1: Consultar a Janus ---
with tab_chat:
    st.header("Haz tu consulta")
    st.markdown("¡Hola! Soy Janus, tu asistente virtual. ¡Estoy aquí para guiarte en tu Inversión Directa en Colombia!")

    with st.form("query_form"):
        prompt = st.text_area("Escribe tu consulta aquí:", height=100)
        submitted = st.form_submit_button("Enviar Consulta a Janus")

    if submitted:
        if not prompt:
            st.warning("Por favor, escribe una pregunta.")
        else:
            with st.spinner("Janus está consultando la Guía Legal..."):
                try:
                    # Llama al motor (el único que queda)
                    respuesta = query_engine.query(prompt)
                    response_text = str(respuesta)
                    
                    with st.expander("Ver Respuesta de Janus", expanded=True):
                        st.markdown(response_text) # Muestra el markdown bonito
                        
                        # --- ¡ROLLBACK A TXT! ---
                        st.download_button(
                            label="📄 Descargar Respuesta (TXT)",
                            data=response_text,
                            file_name="Informe_Janus.txt",
                            mime="text/plain"
                        )
                except Exception as e:
                    st.error(f"Error: {e}")

# --- Pestaña 2: Preguntas Frecuentes (¡COMPLETA!) ---
with tab_faq:
    st.header("Preguntas Frecuentes")
    st.markdown("Haz clic en una pregunta para que Janus la investigue por ti.")
    st.divider()

    # Definimos las 5 preguntas clave
    faq_1 = "¿Qué incentivos fiscales o tributarios específicos ofrece el gobierno para la Inversión Extranjera Directa en energías renovables no convencionales?"
    faq_2 = "¿Cuál es la estructura de sociedad más recomendada para una subsidiaria extranjera en Colombia (como una S.A.S.), y cuáles son los requisitos de capital mínimo para constituirla?"
    faq_3 = "¿Existen restricciones cambiarias o requisitos de registro ante el Banco de la República para traer la inversión inicial y repatriar las utilidades (dividendos)?"
    faq_4 = "¿Qué permisos o licencias clave (ambientales, regulatorias de la CREG, o de conexión) se necesitan para construir y operar un parque de generación de energía renovable?"
    faq_5 = "¿Qué protecciones legales o tratados internacionales (como Acuerdos de Estabilidad Jurídica) ofrece Colombia para proteger mi inversión?"

    # --- Lógica de Botones ---
    
    def run_faq(question):
        """Función que ejecuta la consulta y maneja la respuesta en la pestaña de FAQ."""
        with st.spinner("Generando informe..."):
            try:
                # El motor de PDF usa un template para generar LISTAS limpias
                resp_markdown = query_engine_markdown.query(question)
                resp_pdf = query_engine_pdf.query(question) 
                txt_resp_markdown = str(resp_markdown)
                txt_resp_pdf = str(resp_pdf)
                
                with st.expander(f"Respuesta a: {question}", expanded=True):
                    st.markdown(txt_resp_markdown)
                    
                    # Generación del PDF
                    pdf_data = create_pdf(txt_resp_pdf)
                    if pdf_data:
                        st.download_button("📥 Descargar PDF", data=pdf_data, file_name=f"FAQ_{question[:30]}.pdf", mime="application/pdf")
            except Exception as e:
                st.error(f"Error: {e}")

    # --- Mostrar los 5 botones ---
    if st.button(faq_1): run_faq(faq_1)
    if st.button(faq_2): run_faq(faq_2)
    if st.button(faq_3): run_faq(faq_3)
    if st.button(faq_4): run_faq(faq_4)
    if st.button(faq_5): run_faq(faq_5)


