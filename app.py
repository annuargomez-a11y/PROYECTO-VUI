import os
import sys
import logging
import streamlit as st
import nest_asyncio
import markdown # <-- NUEVO: Para entender el formato
from xhtml2pdf import pisa # <-- NUEVO: La impresora avanzada
from io import BytesIO

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

# --- FUNCIÓN PARA CREAR PDF "BONITO" (HTML a PDF) ---
def create_pdf(markdown_text):
    # 1. Convertir Markdown a HTML (incluyendo tablas)
    html_content = markdown.markdown(markdown_text, extensions=['tables'])
    
    # 2. Crear una plantilla HTML con estilo CSS profesional
    full_html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Helvetica, sans-serif; font-size: 11pt; line-height: 1.5; color: #333; }}
            h1, h2, h3 {{ color: #2c3e50; margin-top: 20px; }}
            
            /* Estilo para las tablas */
            table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; font-weight: bold; color: #333; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            
            /* Encabezado y Pie */
            .header {{ text-align: center; margin-bottom: 30px; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; }}
            .footer {{ font-size: 9pt; text-align: center; margin-top: 50px; color: #777; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>Informe de Asesoría - VUI Colombia</h2>
            <p>Generado por Asistente Janus (IA)</p>
        </div>
        
        {html_content}
        
        <div class="footer">
            <p>Este documento es generado por inteligencia artificial basado en la Guía Legal de Inversión 2025.</p>
        </div>
    </body>
    </html>
    """
    
    # 3. Generar el PDF usando xhtml2pdf
    result_file = BytesIO()
    pisa_status = pisa.CreatePDF(full_html, dest=result_file)
    
    if pisa_status.err:
        return None
    return result_file.getvalue()

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
    
    # TEMPLATE DE PERSONALIDAD (Para asegurar buenas respuestas)
    template_str = (
        "Eres Janus, un experto y amable asesor de inversión extranjera en Colombia (VUI).\n"
        "Usa formato Markdown para estructurar tu respuesta (negritas, listas, y TABLAS si te piden comparar).\n"
        "---------------------\n"
        "Contexto:\n{context_str}\n"
        "---------------------\n"
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
                        
                        # Generar PDF Pro
                        pdf_bytes = create_pdf(response_text)
                        if pdf_bytes:
                            st.download_button(
                                label="📄 Descargar Informe PDF (Formato Profesional)",
                                data=pdf_bytes,
                                file_name="Informe_Janus.pdf",
                                mime="application/pdf"
                            )
                        else:
                            st.error("Error al generar el PDF.")
                    
                except Exception as e:
                    st.error(f"Error: {e}")

with tab_faq:
    st.header("Preguntas Frecuentes")
    # ... (Puedes dejar las FAQs igual que antes o copiarlas aquí si las necesitas)
    faq_1 = "¿Cuál es la estructura de sociedad recomendada (S.A.S.) y capital mínimo?"
    if st.button(faq_1):
         with st.spinner("Generando..."):
            resp = query_engine.query(faq_1)
            txt_resp = str(resp)
            with st.expander("Respuesta", expanded=True):
                st.markdown(txt_resp)
                pdf_data = create_pdf(txt_resp)
                st.download_button("📥 Descargar PDF", data=pdf_data, file_name="FAQ_Janus.pdf", mime="application/pdf")
