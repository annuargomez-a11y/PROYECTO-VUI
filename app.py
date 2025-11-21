import os
import sys
import logging
import streamlit as st
import nest_asyncio # <--- ¡ESTA ES LA LÍNEA QUE FALTA!
from datetime import datetime
from fpdf import FPDF

# --- 1. PARCHES DE SISTEMA ---
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

# --- 2. CONFIGURACIÓN DE LA APP ---
st.set_page_config(
    page_title="Asistente Janus (VUI)",
    page_icon="🗝️",
    layout="centered"
)

# --- 3. VALIDACIÓN DE CLAVES ---
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("Error Crítico: Falta la clave API de OpenAI en los 'Secrets' de Streamlit.")
    st.stop()

# Rutas de archivos
pdf_folder_path = "./ARCHIVOS/"
persist_dir = "./storage"

# --- 4. FUNCIONES DE PDF (Formato Texto Limpio) ---
def clean_text_for_pdf(text):
    """Limpia el formato Markdown para que el PDF no se rompa."""
    # Reemplazos de caracteres especiales latinos
    replacements = {
        '”': '"', '“': '"', '‘': "'", '’': "'", '–': '-', '—': '-', '…': '...',
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
        'ñ': 'n', 'Ñ': 'N'
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    
    # Eliminar sintaxis Markdown (negritas, tablas)
    text = text.replace('**', '').replace('*', '') 
    text = text.replace('|', ' - ').replace('---', '')
    text = re.sub(r'[-]{3,}', '', text)
    return text

def create_pdf(text):
    """Genera un archivo PDF simple tipo reporte."""
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
        # Codificación latin-1 para compatibilidad con FPDF
        pdf.multi_cell(0, 6, clean_content.encode('latin-1', 'replace').decode('latin-1'))
        return pdf.output(dest='S').encode('latin-1', 'replace')
    except Exception as e:
        return None

# --- 5. MOTOR DE INTELIGENCIA (RAG) ---
@st.cache_resource
def get_query_engine():
    
    # --- A. PERSONALIDAD MAESTRA (SYSTEM PROMPT) ---
    # Esta es la instrucción que controla el comportamiento "Políglota" y "Facilitador".
    # Al estar en inglés y como System Message, tiene máxima prioridad.
    janus_system_prompt = (
        "You are Janus, the Official Investment Assistant for the Single Investment Window (VUI) of Colombia. "
        "Your role is to act as a STRATEGIC FACILITATOR to help investors navigate Colombian regulations.\n\n"
        "CRITICAL RULES YOU MUST FOLLOW:\n"
        "1. LANGUAGE (MANDATORY): You must detect the language of the user's question and answer in that EXACT SAME LANGUAGE. "
        "If the user asks in English, answer in English. If in French, answer in French.\n"
        "2. VUE RULE: If the user asks about creating a company (S.A.S.) or commercial registration, refer them to the VUE (Ventanilla Única Empresarial). "
        "Do NOT mention VUCE (which is only for foreign trade).\n"
        "3. CONTENT STYLE: Prioritize practical steps ('HOW') over legal theory ('WHAT'). Use the provided context to give specific details.\n"
        "4.

