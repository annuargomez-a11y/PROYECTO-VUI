import os
import sys
import logging
import streamlit as st
import nest_asyncio
import re  # <-- Para limpieza de texto
from fpdf import FPDF

# --- PARCHES ---
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
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Asistente Janus (VUI)", page_icon="🗝️", layout="centered")

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("Error: Falta la clave API de OpenAI.")
    st.stop() 

pdf_folder_path = "./ARCHIVOS/"
persist_dir = "./storage"

# --- FUNCIÓN DE LIMPIEZA DE TEXTO ---
def clean_markdown(text):
    """Elimina los símbolos de Markdown para que el PDF se vea limpio."""
    # Quitar negritas (**texto**)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    # Quitar cursivas (*texto*)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    # Quitar estructura de tablas markdown (| y -)
    text = text.replace('|', ' ').replace('---', '')
    # Quitar viñetas extrañas
    text = text.replace('###', '').replace('##', '')
    return text

# --- FUNCIÓN PDF ROBUSTA ---
def create_pdf(text):
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, 'Informe de Asesoria - VUI Colombia', 0, 1, 'C')
            self.ln(5)
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, 'Generado por Janus AI', 0, 0, 'C')

    # Limpiamos el texto antes de imprimirlo
    clean_text = clean_markdown(text)
    
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=11)
    
    # Solución para caracteres especiales (tildes, ñ)
    # Reemplazamos caracteres problemáticos comunes en Latin-1
    clean_text = clean_text.replace('”', '"').replace('“', '"').replace('–', '-')
    
    try:
        # Codificar a latin-1 ignorando errores para evitar colapso
        clean_text = clean_text.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 8, clean_text)
        return pdf.output(dest='S').encode('latin-1')
    except Exception as e:
        return None

# --- MOTOR RAG ---
@st.cache_resource
def get_query_engine():
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
    # Prompt simple para evitar formatos complejos que rompan el PDF básico
    query_engine = index.as_query_engine(similarity_top_k=5) 
    return query_engine

# --- INTERFAZ ---
st.title("Asistente Janus")
st.caption("Tu guía para la Ventanilla Única de Inversión (VUI).")

tab_chat, tab_faq = st.tabs(["Consultar a Janus 💬", "Preguntas Frecuentes 💡"])

with tab_chat:
    st.header("Haz tu consulta")
    try:
        query_engine = get_query_engine()
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

    with st.form("query_form"):
        prompt = st.text_area("Escribe tu consulta aquí:", height=100)
        submitted = st.form_submit_button("Enviar Consulta a Janus")

    if submitted and prompt:
        with st.spinner("Janus está redactando..."):
            try:
                respuesta = query_engine.query(prompt)
                response_text = str(respuesta)
                
                with st.expander("Ver Respuesta de Janus", expanded=True):
                    st.markdown(response_text) # En pantalla se ve con formato bonito
                    
                    # Para el PDF usamos la versión limpia
                    pdf_bytes = create_pdf(response_text)
                    
                    if pdf_bytes:
                        st.download_button(
                            label="📄 Descargar Informe PDF",
                            data=pdf_bytes,
                            file_name="Informe_Janus.pdf",
                            mime="application/pdf"
                        )
            except Exception as e:
                st.error(f"Error: {e}")

with tab_faq:
    st.header("Preguntas Frecuentes")
    faq_1 = "¿Cuál es la estructura de sociedad recomendada (S.A.S.) y capital mínimo?"
    if st.button(faq_1):
         with st.spinner("Generando..."):
            resp = query_engine.query(faq_1)
            txt_resp = str(resp)
            with st.expander("Respuesta", expanded=True):
                st.markdown(txt_resp)
                pdf_data = create_pdf(txt_resp)
                if pdf_data:
                    st.download_button("📥 Descargar PDF", data=pdf_data, file_name="FAQ_Janus.pdf", mime="application/pdf")
