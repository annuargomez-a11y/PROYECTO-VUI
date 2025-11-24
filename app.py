import streamlit as st
import nest_asyncio
import os
import sys
from datetime import datetime

# --- 1. PARCHES DE SISTEMA ---
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

# --- 2. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Asistente Janus (VUI)", page_icon="🗝️", layout="centered")

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("Error Crítico: Falta la clave API de OpenAI en los Secrets.")
    st.stop()

pdf_folder_path = "./ARCHIVOS/"
persist_dir = "./storage"

# --- 3. FUNCIÓN DE TRADUCCIÓN MEJORADA (Versión "Fuerza Bruta") ---
def translate_response(text, user_query):
    client = OpenAI(model="gpt-4o-mini", temperature=0)
    
    # Le damos una identidad de traductor experto para evitar que opine
    prompt_traduccion = (
        f"You are a professional translator for an Investment Agency.\n"
        f"USER QUERY: '{user_query}'\n"
        f"ORIGINAL CONTENT: '{text}'\n\n"
        "TASKS:\n"
        "1. Identify the language of the USER QUERY (e.g., English, French, German).\n"
        "2. IGNORE the language of the ORIGINAL CONTENT.\n"
        "3. Translate the ORIGINAL CONTENT into the language of the USER QUERY.\n"
        "4. If the User Query explicitly asks for a specific language (e.g. 'in English'), PRIORITIZE that instruction over detection.\n"
        "5. Maintain all Markdown formatting (bolding, bullets).\n"
        "6. Output ONLY the translation, no introductory text."
    )
    return client.complete(prompt_traduccion).text

# --- 4. MOTOR RAG (CON LA AUDITORÍA INTEGRADA) ---
@st.cache_resource
def get_query_engine():
    # Configuración del Modelo (AJUSTE 1: Temperatura 0.2)
    llm = OpenAI(model="gpt-4o-mini", temperature=0.2)
    embed_model = OpenAIEmbedding(model="text-embedding-3-large")

    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # Carga
    reader = SimpleDirectoryReader(input_dir=pdf_folder_path, recursive=True)
    documents = reader.load_data()
    
    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
    nodes = node_parser.get_nodes_from_documents(documents)
    
    index = VectorStoreIndex(nodes, show_progress=True)
    
    # --- AJUSTE 2: SYSTEM PROMPT DE AUDITORÍA IED ---
    system_prompt = (
        "Eres 'Janus', el Asesor Estratégico y Oficial de Cumplimiento de la VUI Colombia.\n"
        "Tu misión es facilitar negocios en Transición Energética con seguridad jurídica.\n\n"
        "REGLAS CRÍTICAS DE NEGOCIO:\n"
        "1. [TRIBUTARIO] Renta: El beneficio del 50% (Ley 1715) tiene vigencia de **15 AÑOS** y se puede arrastrar.\n"
        "2. [FINANCIERO] Depreciación: La tasa es **FLEXIBLE** (hasta 33.33% anual), decisión del inversor.\n"
        "3. [CAMBIARIO] Deuda: 🚨 ALERTA ROJA. Exige registro (Formulario 4) ANTES del desembolso.\n"
        "4. [SOCIAL] Guajira: Advierte tiempos largos en Consultas Previas. Sugiere holgura.\n"
        "5. [OPERATIVO] Retrasos UPME: SI ES CULPA DEL TRANSPORTADOR, TU PRIORIDAD ES EL AHORRO. NO recomiendes pagar garantías de entrada. TIENES QUE sugerir imperativamente invocar la 'Causal Eximente' (Art. 17, CREG 075) para modificar la FPO sin costo.\n"
        "6. [ESTRATÉGICO] Zonas Francas: Si es Offshore, mantienen 20% renta SIN exportar.\n"
        "7. [JURÍDICO] Estabilidad: No hay contratos. Remite a BITs.\n\n"
        "CIERRE: '¿Le gustaría agendar cita con un especialista de la Dirección de Inversión?'"
    )
    
    llm.system_prompt = system_prompt
    
    return index.as_query_engine(similarity_top_k=5)

# --- 5. INTERFAZ DE USUARIO ---
st.title("Asistente Janus")
st.caption("Tu guía para la Ventanilla Única de Inversión (VUI).")

tab_chat, tab_faq = st.tabs(["Consultar a Janus 💬", "Preguntas Frecuentes 💡"])

try:
    query_engine = get_query_engine()
except Exception as e:
    st.error(f"Error al cargar el motor: {e}")
    st.stop()

# Pestaña 1: Chat
with tab_chat:
    st.header("Haz tu consulta")
    st.markdown("¡Hola! Soy Janus. Estoy aquí para guiarte en tu Inversión Directa en Colombia.")

    with st.form("query_form"):
        prompt = st.text_area("Escribe tu consulta aquí (Cualquier idioma):", height=100)
        submitted = st.form_submit_button("Enviar Consulta")

    if submitted and prompt:
        with st.spinner("Janus está analizando..."):
            try:
                # 1. Respuesta Técnica
                respuesta_raw = query_engine.query(prompt)
                
                # 2. Traducción
                response_final = translate_response(str(respuesta_raw), prompt)
                
                with st.expander("Ver Respuesta de Janus", expanded=True):
                    st.markdown(response_final)
                    
                    # Descarga
                    ahora = datetime.now()
                    nombre = f"Janus.Answer.{ahora.strftime('%Y%m%d.%H%M')}.txt"
                    contenido = f"PREGUNTA:\n{prompt}\n\nRESPUESTA:\n{response_final}"
                    st.download_button("📥 Guardar Respuesta (TXT)", data=contenido, file_name=nombre, mime="text/plain")
            except Exception as e:
                st.error(f"Error: {e}")

# Pestaña 2: FAQs
with tab_faq:
    st.header("Preguntas Frecuentes")
    
    faq_1 = "¿Qué incentivos fiscales hay para energías renovables no convencionales?"
    faq_2 = "¿Cuál es la estructura de sociedad recomendada (S.A.S.) y capital mínimo?"
    faq_3 = "¿Existen restricciones para repatriar utilidades al exterior?"
    faq_4 = "¿Qué permisos ambientales o licencias se necesitan para operar?"
    faq_5 = "¿Qué garantías de estabilidad jurídica ofrece Colombia?"

    def run_faq(q):
        with st.spinner("Consultando..."):
            resp = query_engine.query(q)
            st.markdown(str(resp))

    if st.button(faq_1): run_faq(faq_1)
    if st.button(faq_2): run_faq(faq_2)
    if st.button(faq_3): run_faq(faq_3)
    if st.button(faq_4): run_faq(faq_4)
    if st.button(faq_5): run_faq(faq_5)
