import os
import sys
import logging
import streamlit as st
import nest_asyncio
from datetime import datetime
from fpdf import FPDF

# --- 1. PARCHES ---
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

# --- 2. CONFIGURACIÓN ---
st.set_page_config(
    page_title="Asistente Janus (VUI)",
    page_icon="🗝️",
    layout="centered"
)

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("Error Crítico: Falta la clave API de OpenAI.")
    st.stop()

pdf_folder_path = "./ARCHIVOS/"
persist_dir = "./storage"

# --- 3. PDF ---
def clean_text_for_pdf(text):
    replacements = {
        '”': '"', '“': '"', '‘': "'", '’': "'", '–': '-', '—': '-', '…': '...',
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
        'ñ': 'n', 'Ñ': 'N'
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    text = text.replace('**', '').replace('*', '')
    text = text.replace('|', ' - ').replace('---', '')
    return text

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

    clean_content = clean_text_for_pdf(text)
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=11)
    try:
        pdf.multi_cell(0, 6, clean_content.encode('latin-1', 'replace').decode('latin-1'))
        return pdf.output(dest='S').encode('latin-1', 'replace')
    except Exception:
        return None

# --- 4. MOTOR RAG ---
@st.cache_resource
def get_query_engine():
    
    # SYSTEM PROMPT (EN INGLÉS PARA EVITAR SESGOS)
    system_instruction = (
        "You are Janus, the Official Investment Assistant for the Single Investment Window (VUI) of Colombia. "
        "Your role is to act as a STRATEGIC FACILITATOR.\n"
        "CRITICAL RULES:\n"
        "1. LANGUAGE (MANDATORY): Detect the language of the user's question and answer in that EXACT SAME LANGUAGE. "
        "If the user asks in English, answer in English.\n"
        "2. VUE RULE: If asked about creating a company (S.A.S.), refer to VUE (Ventanilla Única Empresarial). Do NOT mention VUCE.\n"
        "3. CONTENT: Prioritize practical steps ('HOW') over legal theory ('WHAT').\n"
        "4. FORMAT: Use Markdown (bolding, lists)."
    )

    llm = OpenAI(
        model="gpt-4o-mini", 
        temperature=0.1,
        system_prompt=system_instruction
    )
    
    embed_model = OpenAIEmbedding(model="text-embedding-3-large")

    Settings.llm = llm
    Settings.embed_model = embed_model
    
    reader = SimpleDirectoryReader(input_dir=pdf_folder_path, recursive=True)
    documents = reader.load_data()
    
    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
    nodes = node_parser.get_nodes_from_documents(documents)
    
    index = VectorStoreIndex(nodes, show_progress=True)
    
    query_engine = index.as_query_engine(similarity_top_k=5) 
    return query_engine

# --- 5. INTERFAZ ---
st.title("Asistente Janus")
st.caption("Tu guía para la Ventanilla Única de Inversión (VUI).")

tab_chat, tab_faq = st.tabs(["Consultar a Janus 💬", "Preguntas Frecuentes 💡"])

try:
    query_engine = get_query_engine()
except Exception as e:
    st.error(f"Error al cargar motor: {e}")
    st.stop()

# Pestaña 1
with tab_chat:
    st.header("Haz tu consulta")
    st.markdown("¡Hola! Soy Janus. Estoy aquí para guiarte en tu Inversión Directa en Colombia.")

    with st.form("query_form"):
        prompt = st.text_area("Escribe tu consulta aquí:", height=100)
        submitted = st.form_submit_button("Enviar Consulta a Janus")

    if submitted and prompt:
        with st.spinner("Janus está analizando..."):
            try:
                respuesta = query_engine.query(prompt)
                response_text = str(respuesta)
                
                with st.expander("Ver Respuesta de Janus", expanded=True):
                    st.markdown(response_text)
                    
                    # PDF
                    ahora = datetime.now()
                    nombre_pdf = f"Janus.Answer.{ahora.strftime('%Y%m%d.%H%M')}.pdf"
                    texto_pdf = f"PREGUNTA:\n{prompt}\n\nRESPUESTA:\n{response_text}"
                    pdf_bytes = create_pdf(texto_pdf)
                    
                    if pdf_bytes:
                        st.download_button("📥 Guardar PDF", data=pdf_bytes, file_name=nombre_pdf, mime="application/pdf")
                    
                    # TXT
                    nombre_txt = f"Janus.Answer.{ahora.strftime('%Y%m%d.%H%M')}.txt"
                    st.download_button("📥 Guardar TXT", data=texto_pdf, file_name=nombre_txt, mime="text/plain")
            except Exception as e:
                st.error(f"Error: {e}")

# Pestaña 2
with tab_faq:
    st.header("Preguntas Frecuentes")
    
    faq_1 = "¿Qué incentivos fiscales hay para energías renovables no convencionales?"
    faq_2 = "¿Cuál es la estructura de sociedad recomendada (S.A.S.) y capital mínimo?"
    faq_3 = "¿Existen restricciones para repatriar utilidades al exterior?"

    def run_faq(question):
        with st.spinner("Consultando..."):
            resp = query_engine.query(question)
            txt_resp = str(resp)
            with st.expander("Respuesta", expanded=True):
                st.markdown(txt_resp)

    if st.button(faq_1): run_faq(faq_1)
    if st.button(faq_2): run_faq(faq_2)
    if st.button(faq_3): run_faq(faq_3)
