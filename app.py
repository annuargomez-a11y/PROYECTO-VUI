import os
import sys
import logging
import streamlit as st
import nest_asyncio
from fpdf import FPDF  # <-- ¡NUEVA IMPORTACIÓN! (La impresora)

# --- PARCHES CRÍTICOS ---
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

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Asistente Janus (VUI)",
    page_icon="🗝️",
    layout="centered" 
)

# --- CONFIGURACIÓN DE API ---
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("Error: Falta la clave API de OpenAI en los Secrets.")
    st.stop() 

pdf_folder_path = "./ARCHIVOS/"
persist_dir = "./storage"

# --- FUNCIÓN PARA CREAR PDF (NUEVA) ---
def create_pdf(text):
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, 'Informe de Janus - VUI Colombia', 0, 1, 'C')
            self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, 'Pagina ' + str(self.page_no()), 0, 0, 'C')

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Limpieza básica de caracteres para evitar errores de codificación en FPDF
    # FPDF estándar a veces pelea con tildes si no se usa una fuente unicode externa,
    # así que usamos latin-1 para compatibilidad básica.
    text = text.encode('latin-1', 'replace').decode('latin-1')
    
    pdf.multi_cell(0, 10, text)
    
    # Retornamos el PDF como bytes
    return pdf.output(dest='S').encode('latin-1')

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
    query_engine = index.as_query_engine(similarity_top_k=5) 
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
            st.warning("Por favor, escribe una pregunta.")
        else:
            with st.spinner("Janus está redactando el informe..."):
                try:
                    respuesta = query_engine.query(prompt)
                    response_text = str(respuesta)
                    
                    with st.expander("Ver Respuesta de Janus", expanded=True):
                        st.markdown(response_text)
                        
                        # --- GENERACIÓN DEL PDF ---
                        pdf_bytes = create_pdf(response_text)
                        
                        # --- BOTÓN DE DESCARGA PDF ---
                        st.download_button(
                            label="📄 Descargar Informe en PDF",
                            data=pdf_bytes,
                            file_name="Informe_Janus_VUI.pdf",
                            mime="application/pdf"
                        )
                    
                except Exception as e:
                    st.error(f"Error: {e}")

with tab_faq:
    st.header("Preguntas Frecuentes")
    
    faq_1 = "¿Qué incentivos fiscales hay para energías renovables?"
    faq_2 = "¿Cuál es la estructura de sociedad recomendada (S.A.S.) y capital mínimo?"
    faq_3 = "¿Existen restricciones para repatriar utilidades?"
    faq_4 = "¿Qué permisos ambientales se necesitan?"
    faq_5 = "¿Qué garantías de estabilidad jurídica existen?"

    def run_faq(question):
        with st.spinner("Generando informe..."):
            resp = query_engine.query(question)
            txt_resp = str(resp)
            
            with st.expander("Respuesta", expanded=True):
                st.markdown(txt_resp)
                
                # Generar PDF también para las FAQs
                pdf_data = create_pdf(txt_resp)
                
                st.download_button(
                    label="📄 Descargar PDF",
                    data=pdf_data,
                    file_name="FAQ_Janus.pdf",
                    mime="application/pdf"
                )

    if st.button(faq_1): run_faq(faq_1)
    if st.button(faq_2): run_faq(faq_2)
    if st.button(faq_3): run_faq(faq_3)
    if st.button(faq_4): run_faq(faq_4)
    if st.button(faq_5): run_faq(faq_5)
