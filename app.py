import os
import sys
import logging
import streamlit as st
import nest_asyncio
import re 
from fpdf import FPDF

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

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Asistente Janus (VUI)", page_icon="🗝️", layout="centered")

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("Error: Falta la clave API de OpenAI.")
    st.stop() 

pdf_folder_path = "./ARCHIVOS/"
persist_dir = "./storage"

# --- FUNCIÓN DE LIMPIEZA (ELIMINADOR DE TABLAS) ---
def clean_text_for_pdf(text):
    """Convierte el formato Markdown complejo en texto plano simple."""
    
    # 1. Reemplazos de caracteres especiales (tildes, etc)
    replacements = {
        '”': '"', '“': '"', '‘': "'", '’': "'",
        '–': '-', '—': '-', '…': '...',
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
        'ñ': 'n', 'Ñ': 'N'
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    
    # 2. DESTRUIR LA TABLA (Esto es lo que falló antes)
    # Reemplazamos las barras de tabla por guiones o espacios
    text = text.replace('|', ' - ') 
    
    # Quitamos las líneas horizontales de la tabla markdown
    text = re.sub(r'[-]{3,}', '', text)
    
    # Quitamos negritas y cursivas
    text = text.replace('**', '').replace('*', '')
    
    # Quitamos espacios extra múltiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# --- FUNCIÓN PDF ---
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
        # Multi_cell ajusta el texto al ancho. 
        # Al haber limpiado la tabla, se verá como párrafos seguidos.
        pdf.multi_cell(0, 6, clean_content)
        return pdf.output(dest='S').encode('latin-1', 'replace')
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
    
    # 3. DEFINIR LA PERSONALIDAD (EL PROMPT TEMPLATE)
    template_str = (
        "Eres Janus, un experto y amable asesor de inversión extranjera en Colombia.\n"
        "Tu misión es guiar a los inversionistas con respuestas claras, completas y estratégicas.\n"
        "---------------------\n"
        "Contexto:\n{context_str}\n"
        "---------------------\n"
        "Instrucciones:\n"
        "1. Responde en Español. Sé detallado y profesional.\n"
        "2. FORMATO CRÍTICO: Cuando respondas una consulta comparativa (como 'cuadro comparativo' o 'diferencias'), NO uses tablas de Markdown. Estructura tu respuesta usando listas con encabezados en **Negrita** para mantener la legibilidad en documentos PDF.\n"
        "3. Si la respuesta es técnica o breve, EXPLICA sus implicaciones.\n"
        "4. Si el contexto no tiene la información, dilo honestamente.\n"
        "Pregunta: {query_str}\n\n"
        "Respuesta:"
    )
    
    qa_template = PromptTemplate(template_str)
    
    query_engine = index.as_query_engine(similarity_top_k=5, text_qa_template=qa_template) 
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
        with st.spinner("Janus está redactando el informe..."):
            try:
                respuesta = query_engine.query(prompt)
                response_text = str(respuesta)
                
                with st.expander("Ver Respuesta de Janus", expanded=True):
                    st.markdown(response_text) 
                    
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

