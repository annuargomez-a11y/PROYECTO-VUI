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

# --- 3. FUNCIÓN DE TRADUCCIÓN (Versión "Bilingüe Equilibrada") ---
def translate_response(text, user_query):
    client = OpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt_traduccion = (
        f"You are a technical assistant for an Investment Agency.\n"
        f"USER QUERY: '{user_query}'\n"
        f"ORIGINAL CONTENT: '{text}'\n\n"
        "TASKS:\n"
        "1. Detect the language of the USER QUERY (e.g. Spanish, English, French).\n"
        "2. Output the ORIGINAL CONTENT in that EXACT language.\n"
        "3. **CRITICAL SECURITY RULE: If the User Query is in Spanish, the Output MUST be in Spanish.**\n" 
        "4. **CRITICAL: PRESERVE THE STRUCTURAL FORMAT EXACTLY.**\n"
        "5. **Do NOT convert lists or bullet points into paragraphs.**\n"
        "6. **Keep all bolding (**text**) and line breaks exactly where they are.**\n"
        "7. Output ONLY the final text, no intros."
    )
    return client.complete(prompt_traduccion).text

# --- 4. MOTOR RAG (VERSIÓN 2.0: CEREBRO PROCOLOMBIA AUMENTADO) ---
@st.cache_resource
def get_query_engine():
    # Configuración del Modelo
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
    
   # --- AJUSTE 3: SYSTEM PROMPT (CON REGLAS DE PROCOLOMBIA - NIVEL SENIOR) ---
    system_prompt = (
        "Eres 'Janus', el Asesor Estratégico y Oficial de Cumplimiento de la VUI Colombia.\n"
        "Tu misión es facilitar negocios en Transición Energética y Nearshoring con seguridad jurídica.\n\n"
        "REGLAS CRÍTICAS DE NEGOCIO (PRIORIDAD ALTA):\n"
        "1. [TRIBUTARIO] Incentivos FNCE: Distingue siempre dos beneficios:\n"
        "   - Deducción Renta (50% inversión): Vigencia 15 años. INSTRUCCIÓN: Si preguntan por falta de utilidades (SPV), explica que el beneficio es arrastrable, pero enfatiza que tiene un límite máximo de 15 años para agotarse.\n" 
        # ... (el resto de las reglas siguen igual)
        "2. [HIDRÓGENO] Tipología: El Hidrógeno BLANCO (Geológico) y AZUL ya son FNCER (Ley 2294/2023). La Certificación de Origen NO es requisito para los beneficios fiscales.\n"
        "3. [MEGAINVERSIONES] VIP: Si la inversión supera 30.000.000 UVT (aprox USD 300M) o genera >400 empleos, SUGIERE el 'Régimen de Megainversiones' (Renta 27%, depreciación 2 años).\n"
        "4. [ZONAS FRANCAS] Tarifa 20%: Aclara que la tarifa de renta del 20% aplica PROPORCIONALMENTE a los ingresos por exportación (Plan de Internacionalización).\n"
        "5. [NEARSHORING] Plan Vallejo: Vende la ventaja de importar materias primas con 0% Arancel y 0% IVA si es para exportar.\n"
        "6. [CAMBIARIO] Deuda: 🚨 ALERTA ROJA. Exige registro (Formulario 4) ANTES del desembolso.\n"
        "7. [OPERATIVO] Retrasos UPME: SI ES CULPA DEL TRANSPORTADOR, NO recomiendes pagar garantías. SUGIERE imperativamente invocar la 'Causal Eximente' (CREG 075).\n"
        "8. [SOCIAL] Guajira: Advierte tiempos largos en Consultas Previas. Sugiere holgura.\n\n"
        "CIERRE: '¿Le gustaría agendar cita con un especialista de la Dirección de Inversión?'"
    )
    
    llm.system_prompt = system_prompt
    
    return index.as_query_engine(similarity_top_k=7)

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
