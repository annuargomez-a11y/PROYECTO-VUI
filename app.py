import os
import sys
import logging
import streamlit as st
import nest_asyncio
import re 
from fpdf import FPDF
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
    st.error("Error: Falta la clave API de OpenAI en los Secrets.")
    st.stop() 

pdf_folder_path = "./ARCHIVOS/"
persist_dir = "./storage"

# --- FUNCIÓN DE LIMPIEZA (SOLO PARA PDF - Mantiene el texto limpio) ---
def clean_text_for_pdf(text):
    """Convierte el texto markdown complejo a texto plano para impresión."""
    replacements = {
        '”': '"', '“': '"', '‘': "'", '’': "'",
        '–': '-', '—': '-', '…': '...',
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
        'ñ': 'n', 'Ñ': 'N'
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    
    # Quita negritas, cursivas, y estructura de tabla ([|] y [-])
    text = text.replace('**', '').replace('*', '') 
    text = text.replace('|', ' | ').replace('---', '')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- FUNCIÓN PARA CREAR PDF ---
def create_pdf(text):
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, 'INFORME DE ASESORIA - VUI COLOMBIA', 0, 1, 'C')
            self.ln(5)
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, 'Generado por Asistente Janus', 0, 0, 'C')

    # Usamos la función de limpieza agresiva para asegurar que no haya caracteres rotos
    clean_content = clean_text_for_pdf(text)
    
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=11)
    
    try:
        # Multi_cell ajusta el texto al ancho.
        pdf.multi_cell(0, 6, clean_content.encode('latin-1', 'replace').decode('latin-1'))
        return pdf.output(dest='S').encode('latin-1', 'replace')
    except Exception as e:
        print(f"Error PDF: {e}")
        return None

# --- MOTOR RAG ---
@st.cache_resource
def get_query_engine():
    # El motor principal usará un template que PIDE TABLAS
    llm = OpenAI(model="gpt-4o-mini", temperature=0.2)
    embed_model = OpenAIEmbedding(model="text-embedding-3-large")
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # Template para la PANTALLA (Pide tablas para un look ejecutivo)
    template_str_markdown = (
        "Eres Janus, un experto asesor de inversión extranjera en Colombia. Responde usando formato Markdown.\n"
        "---------------------\n"
        "Contexto:\n{context_str}\n"
        "---------------------\n"
        "Instrucciones:\n"
        "1. Responde en el idioma de la pregunta. 2. Usa TABLAS DE MARKDOWN para cualquier comparación o listado de ventajas/desventajas. 3. Sé detallado y profesional.\n"
        "Pregunta: {query_str}\n\n"
        "Respuesta:"
    )
    
    # Template para el PDF (Pide LISTAS simples para impresión)
    template_str_pdf = (
        "Eres un experto asesor de inversión. NO uses formato Markdown ni tablas. Responde la siguiente pregunta en forma de lista o párrafos muy claros, idealmente con encabezados en mayúsculas.\n"
        "Contexto: {context_str}\n"
        "Pregunta: {query_str}\n\n"
        "Respuesta en texto plano:"
    )
    
    qa_template_markdown = PromptTemplate(template_str_markdown)
    qa_template_pdf = PromptTemplate(template_str_pdf)
    
    print("--- INICIANDO MOTOR ---")
    reader = SimpleDirectoryReader(input_dir=pdf_folder_path, recursive=True)
    documents = reader.load_data()
    
    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
    nodes = node_parser.get_nodes_from_documents(documents)
    
    index = VectorStoreIndex(nodes, show_progress=True)
    
    # Creamos dos Query Engines con diferentes templates
    query_engine_markdown = index.as_query_engine(similarity_top_k=5, text_qa_template=qa_template_markdown)
    query_engine_pdf = index.as_query_engine(similarity_top_k=5, text_qa_template=qa_template_pdf) 

    return query_engine_markdown, query_engine_pdf

# --- INTERFAZ ---
st.title("Asistente Janus")
st.caption("Tu guía para la Ventanilla Única de Inversión (VUI).")

tab_chat, tab_faq = st.tabs(["Consultar a Janus 💬", "Preguntas Frecuentes 💡"])

# --- Ejecución del Motor ---
try:
    # Desempacamos los dos motores
    query_engine_markdown, query_engine_pdf = get_query_engine()
except Exception as e:
    st.error(f"Error al cargar el motor: {e}")
    st.stop()


# --- Pestaña 1: El Chat (¡CON EL DISEÑO FORMULARIO QUE TE GUSTÓ!) ---
with tab_chat:
    st.header("Haz tu consulta")
    st.markdown("¡Hola! Soy Janus, tu asistente virtual. ¡Estoy aquí para guiarte en tu Inversión Directa en Colombia!")

    with st.form("query_form"):
        prompt = st.text_area("Escribe tu consulta aquí:", height=100)
        submitted = st.form_submit_button("Enviar Consulta a Janus")

    if submitted and prompt:
        if not prompt:
            st.warning("Por favor, escribe una pregunta.")
        else:
            with st.spinner("Janus está redactando el informe..."):
                try:
                    # Llama al motor de PANTALLA (Markdown)
                    respuesta_markdown = query_engine_markdown.query(prompt)
                    response_text_markdown = str(respuesta_markdown)
                    
                    # Llama al motor de PDF (Lista simple)
                    respuesta_pdf = query_engine_pdf.query(prompt)
                    response_text_pdf = str(respuesta_pdf)
                    
                    with st.expander("Ver Respuesta de Janus", expanded=True):
                        st.markdown(response_text_markdown) # Muestra el markdown bonito
                        
                        pdf_bytes = create_pdf(response_text_pdf) # Usa la respuesta limpia para el PDF
                        
                        if pdf_bytes:
                            st.download_button(
                                label="📄 Descargar Informe PDF",
                                data=pdf_bytes,
                                file_name="Informe_Janus.pdf",
                                mime="application/pdf"
                            )
                except Exception as e:
                    st.error(f"Error: {e}")

# --- Pestaña 2: Preguntas Frecuentes ---
with tab_faq:
    st.header("Preguntas Frecuentes")
    st.markdown("Haz clic en una pregunta para investigarla.")
    
    faq_1 = "¿Cuál es la estructura de sociedad recomendada (S.A.S.) y capital mínimo?"
    # ... (Aquí irían los botones del 1 al 5 como antes) ...

    # Ejemplo de botón (usando el motor de PDF para la descarga)
    if st.button(faq_1):
         with st.spinner("Generando..."):
            resp_markdown = query_engine_markdown.query(faq_1)
            resp_pdf = query_engine_pdf.query(faq_1) # Query extra para el PDF
            
            with st.expander("Respuesta", expanded=True):
                st.markdown(str(resp_markdown))
                pdf_data = create_pdf(str(resp_pdf))
                if pdf_data:
                    st.download_button("📥 Descargar PDF", data=pdf_data, file_name="FAQ_Janus.pdf", mime="application/pdf")
