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

# --- 3. FUNCIÓN DE TRADUCCIÓN (Mantenemos esta joya) ---
def translate_response(text, user_query):
    client = OpenAI(model="gpt-4o-mini", temperature=0)
    prompt_traduccion = (
        f"User Query: '{user_query}'\n"
        f"Original Answer: '{text}'\n\n"
        "INSTRUCTION: \n"
        "1. Detect the language of the 'User Query'.\n"
        "2. Translate the 'Original Answer' into that EXACT language.\n"
        "3. Do NOT add introductions. Maintain Markdown.\n"
        "4. If query is Spanish, return text as is.\n"
        "Translation:"
    )
    return client.complete(prompt_traduccion).text

# --- 4. MOTOR RAG ---
@st.cache_resource
def get_query_engine():
    # Configuración del Modelo
    llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
    embed_model = OpenAIEmbedding(model="text-embedding-3-large")

    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # Carga
    reader = SimpleDirectoryReader(input_dir=pdf_folder_path, recursive=True)
    documents = reader.load_data()
    
    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
    nodes = node_parser.get_nodes_from_documents(documents)
    
    index = VectorStoreIndex(nodes, show_progress=True)
    
    # --- NUEVO SYSTEM PROMPT (Con las reglas del "Primo") ---
    system_prompt = (
        "Eres Janus, el Asistente Oficial de la VUI Colombia. Tu rol es FACILITADOR ESTRATÉGICO.\n\n"
        "REGLAS DE NEGOCIO CRÍTICAS:\n"
        "1. GEOGRAFÍA (Energía): Si el usuario NO especifica 'Costa Afuera' (Offshore), ASUME proyecto en Tierra Firme. "
        "NO menciones 'Ocupación Temporal' ni cronogramas de la DIMAR. Guía hacia Licencia Ambiental (ANLA/CAR).\n"
        "2. IDENTIDAD INSTITUCIONAL: Menciona siempre a la entidad (VUI, UPME, DIAN), NO al software. "
        "Ejemplo: Di 'Gestiona en la plataforma de la UPME', NUNCA digas 'Regístrate en Bizagi'.\n"
        "3. REGLA VUE: Para crear empresas, refiere a VUE, nunca VUCE.\n"
        "4. PRIORIDAD: Pasos prácticos ('CÓMO') sobre teoría.\n"
        "5. CIERRE COMERCIAL: Al final, pregunta siempre: '¿Te gustaría que te contacte con un especialista de la Dirección de Inversión?'"
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
                # 1. Respuesta Técnica (Español + Reglas de Negocio)
                respuesta_raw = query_engine.query(prompt)
                
                # 2. Traducción (Si aplica)
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

# --- 3. FUNCIÓN DE TRADUCCIÓN (Sin cambios, funciona perfecto) ---
def translate_response(text, user_query):
    client = OpenAI(model="gpt-4o-mini", temperature=0)
    prompt_traduccion = (
        f"User Query: '{user_query}'\n"
        f"Original Answer: '{text}'\n\n"
        "INSTRUCTION: \n"
        "1. Detect the language of the 'User Query'.\n"
        "2. Translate the 'Original Answer' into that EXACT language.\n"
        "3. Do NOT add introductions. Maintain Markdown.\n"
        "4. If query is Spanish, return text as is.\n"
        "Translation:"
    )
    return client.complete(prompt_traduccion).text

# --- 4. MOTOR RAG (AQUÍ ESTÁN LOS CAMBIOS DE LA AUDITORÍA) ---
@st.cache_resource
def get_query_engine():
    # Configuración del Modelo
    # CAMBIO 1: Ajuste fino de temperatura a 0.2 para balancear rigor y fluidez
    llm = OpenAI(model="gpt-4o-mini", temperature=0.2) 
    embed_model = OpenAIEmbedding(model="text-embedding-3-large")

    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # Carga de Documentos
    reader = SimpleDirectoryReader(input_dir=pdf_folder_path, recursive=True)
    documents = reader.load_data()
    
    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
    nodes = node_parser.get_nodes_from_documents(documents)
    
    index = VectorStoreIndex(nodes, show_progress=True)
    
    # --- CAMBIO 2: NUEVO SYSTEM PROMPT (REGLAS DE AUDITORÍA IED) ---
    system_prompt = (
        "Eres 'Janus', el Asesor Estratégico y Oficial de Cumplimiento Virtual de la Ventanilla Única de Inversión (VUI) de Colombia.\n"
        "Tu misión es orientar a inversionistas en Transición Energética. NO eres un buscador, eres un facilitador de negocios.\n\n"
        "DIRECTRICES DE COMPORTAMIENTO:\n"
        "1. Seguridad Jurídica: Basa tus respuestas ÚNICAMENTE en el contexto recuperado (RAG). Si no está en los documentos, indícalo.\n"
        "2. Anticipación: Advierte riesgos proactivamente (Tributarios, Sociales, Cambiarios).\n\n"
        "REGLAS DE LÓGICA DE NEGOCIO (CRÍTICAS):\n"
        "1. [TRIBUTARIO] Renta: Aclara SIEMPRE que el beneficio del 50% (Ley 1715) tiene vigencia de **15 AÑOS** y se puede arrastrar si no hay renta líquida inicial.\n"
        "2. [FINANCIERO] Depreciación: Aclara que la tasa es **FLEXIBLE** (hasta 33.33% anual), el inversionista decide cuánto usar.\n"
        "3. [CAMBIARIO] Deuda Externa: 🚨 ALERTA ROJA. Si mencionan préstamos, exige registro (Formulario 4) ANTES o SIMULTÁNEAMENTE al desembolso. Menciona sanciones severas.\n"
        "4. [SOCIAL] Guajira: Advierte con tacto que esta región presenta tiempos de Consultas Previas más largos. Sugiere holgura en cronograma.\n"
        "5. [OPERATIVO] Retrasos UPME: Sugiere invocar 'Causal Eximente' (CREG 075) para modificar la FPO sin costo, antes de pagar garantías extra.\n"
        "6. [REGULATORIO] Hidrógeno: El certificado comercial NO es requisito para beneficios tributarios de Ley 1715.\n"
        "7. [ESTRATÉGICO] Zonas Francas: Si es Costa Afuera (Offshore), aclara que mantienen 20% renta SIN requisito de exportación.\n"
        "8. [JURÍDICO] Estabilidad: No existen contratos de estabilidad. Remite a Tratados Bilaterales (BITs).\n\n"
        "CIERRE OBLIGATORIO: Al final, pregunta siempre: '¿Le gustaría que agende una cita prioritaria con un especialista de la Dirección de Inversión para revisar su caso?'"
    )
    
    llm.system_prompt = system_prompt
    
    return index.as_query_engine(similarity_top_k=5)

# --- 5. INTERFAZ DE USUARIO (Sin cambios estructurales) ---
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
        with st.spinner("Janus está analizando normativa y riesgos..."):
            try:
                # 1. Respuesta Técnica (Español + Reglas de Negocio)
                respuesta_raw = query_engine.query(prompt)
                
                # 2. Traducción (Si aplica)
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
        with st.spinner("Consultando base legal..."):
            resp = query_engine.query(q)
            st.markdown(str(resp))

    if st.button(faq_1): run_faq(faq_1)
    if st.button(faq_2): run_faq(faq_2)
    if st.button(faq_3): run_faq(faq_3)
    if st.button(faq_4): run_faq(faq_4)
    if st.button(faq_5): run_faq(faq_5)
