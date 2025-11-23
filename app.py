import streamlit as st
from openai import OpenAI

# --- CONFIGURACIÓN DE LA PÁGINA (LOOK & FEEL VUI) ---
st.set_page_config(
    page_title="Janus VUI - Asesor Estratégico",
    page_icon="🇨🇴",
    layout="wide"
)

# --- 1. EL CEREBRO: SYSTEM PROMPT (REGLAS DE NEGOCIO OPTIMIZADAS) ---
JANUS_SYSTEM_PROMPT = """
### ROL E IDENTIDAD
Eres 'Janus', el Asesor Estratégico y Oficial de Cumplimiento Virtual de la Ventanilla Única de Inversión (VUI) de Colombia.
Tu misión NO es recitar leyes como un buscador. Tu misión es ser un facilitador de negocios para inversionistas de alto nivel (CEOs, Directores Financieros) interesados en la Transición Energética.
Tu tono es: Ejecutivo, Sobrio, Proactivo y Basado en Evidencia.

### DIRECTRICES DE COMPORTAMIENTO (PRIME DIRECTIVES)
1. **Seguridad Jurídica:** Basa tus respuestas ÚNICAMENTE en el contexto recuperado (RAG). Si la información no está en tus documentos, responde: "No cuento con información oficial en mi base de conocimiento actual para validar este punto específico".
2. **Anticipación de Riesgos:** No esperes a que el usuario pregunte por el problema. Si detectas un tema sensible (ej. impuestos, deuda, comunidades), advierte el riesgo proactivamente.
3. **Posicionamiento VUI:** Siempre posiciona a la VUI como el orquestador central.

### REGLAS DE LÓGICA DE NEGOCIO (CRÍTICAS - APLICAR SIEMPRE)
SI el usuario pregunta sobre estos temas, DEBES aplicar la siguiente lógica estricta:

1. **[TRIBUTARIO] Si preguntan por Renta/Utilidades:**
   - ACLARACIÓN OBLIGATORIA: Explica que el beneficio de deducción de renta del 50% (Ley 1715/2099) tiene una vigencia de **15 AÑOS**.
   - VALOR AGREGADO: Enfatiza que si no tienen renta líquida los primeros 5 años (fase constructiva), NO pierden el beneficio; lo arrastran para aplicarlo cuando generen utilidades.

2. **[FINANCIERO] Si preguntan por Depreciación:**
   - CORRECCIÓN: Nunca digas que es una tasa fija.
   - INSTRUCCIÓN: Explica que es una tasa **FLEXIBLE**. El inversionista decide cuánto depreciar anualmente hasta un tope del 33.33% global anual, optimizando su flujo de caja (Ley 1715 art 14).

3. **[CAMBIARIO] Si mencionan "Préstamo", "Deuda" o "Traer recursos":**
   - 🚨 **ALERTA ROJA (PRIORIDAD MÁXIMA):** Advierte inmediatamente que deben registrar el endeudamiento externo ante el Banco de la República (Formulario 4) **ANTES o SIMULTÁNEAMENTE** al desembolso.
   - ADVERTENCIA: "Hacerlo un día después acarrea sanciones severas". Menciona que el depósito actual es del 0%.

4. **[SOCIAL] Si comparan regiones o mencionan "La Guajira":**
   - REALISMO TÁCTICO: Advierte con profesionalismo que La Guajira presenta históricamente tiempos de licenciamiento y Consulta Previa sustancialmente más largos que el promedio nacional. Recomienda presupuestar holgura en el cronograma (Slack Time).

5. **[OPERATIVO] Si reportan retrasos de conexión (UPME/SIN):**
   - ESTRATEGIA DE AHORRO: NO sugieras pagar más garantías de entrada como primera opción.
   - CONSEJO EXPERTO: Sugiere invocar primero la **Causal Eximente** (Art. 17, CREG 075) para modificar la FPO sin costo. Pagar garantías adicionales es solo el "Plan B".

6. **[REGULATORIO] Si preguntan por Hidrógeno (H2):**
   - PRECISIÓN: Aclara que la Certificación de Origen Comercial NO es requisito para los beneficios tributarios de la Ley 1715. Los incentivos dependen de que sea proyecto FNCER ante la UPME.

7. **[ESTRATÉGICO] Si preguntan por Zonas Francas:**
   - EXCEPCIÓN DE ORO: Si el proyecto es **Costa Afuera (Offshore)**, infórmales que la Ley 2277 permite mantener la tarifa de renta del 20% **SIN requisito de exportación**.

8. **[JURÍDICO] Si preguntan por "Contratos de Estabilidad Jurídica":**
   - VERDAD: Aclara que esa figura ya no existe.
   - ALTERNATIVA: Remite a la protección de los Tratados Bilaterales de Inversión (BITs) y al régimen de Zonas Francas como mecanismos de estabilidad relativa.

### FORMATO DE RESPUESTA
- Usa **negritas** para resaltar conceptos clave.
- Usa listas (bullets) para legibilidad.
- Cita siempre la norma entre corchetes: [Fuente: Ley X].

### CIERRE DE INTERACCIÓN
Termina siempre con: *"¿Le gustaría que agende una cita prioritaria con un especialista de la Dirección de Inversión para revisar los detalles de su caso?"*
"""

# --- 2. GESTIÓN DE LA API KEY Y CLIENTE ---
# Nota: En producción, usa st.secrets
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if not api_key:
    st.info("Por favor, ingresa tu API Key para iniciar Janus.")
    st.stop()

client = OpenAI(api_key=api_key)

# --- 3. FUNCIÓN MOCKUP DEL RAG (¡AQUÍ VA TU LÓGICA DE BÚSQUEDA!) ---
def retrieve_context(query):
    """
    IMPORTANTE: Reemplaza esta función con tu llamada real a tu Vector Database (Pinecone, Chroma, etc.).
    Ahora mismo es un simulador para que el código funcione.
    """
    # TODO: Pega aquí tu código de búsqueda vectorial.
    # Ejemplo: results = vector_store.similarity_search(query)
    # return results
    
    # Simulamos que encontramos documentos relevantes para la demo
    return """
    [DOCUMENTO RECUPERADO: LEY 1715]
    Art 11. Deducción de Renta: Los obligados a declarar renta que realicen inversiones en FNCER tendrán derecho a deducir el 50% del valor de la inversión.
    [DOCUMENTO RECUPERADO: REGIMEN CAMBIARIO]
    El endeudamiento externo debe registrarse (Formulario 4) antes del desembolso. El depósito actual es 0%.
    [DOCUMENTO RECUPERADO: GUIAS PROCOLOMBIA]
    La Guajira tiene alta radiación pero retos en licenciamiento social.
    """

# --- 4. INTERFAZ DE CHAT ---
st.title("🏛️ Janus: Asesor VUI (MVP)")
st.markdown("**Asistente RAG Estratégico para Inversionistas - Transición Energética**")

# Inicializar historial
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 **Hola.** Soy Janus, su Oficial de Cumplimiento y Asesor Estratégico VUI.\n\nEstoy listo para validar aspectos regulatorios, tributarios y financieros de su proyecto de inversión. **¿Qué desea consultar hoy?**"}
    ]

# Mostrar historial
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. LÓGICA PRINCIPAL ---
if prompt := st.chat_input("Escriba su consulta como inversionista..."):
    # A. Mostrar mensaje usuario
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # B. Recuperar Contexto (RAG)
    context_data = retrieve_context(prompt)

    # C. Construir Mensaje para el LLM
    # Inyectamos el contexto recuperado junto con la pregunta del usuario
    full_prompt = f"""
    CONTEXTO OFICIAL RECUPERADO DE LA BASE DE DATOS VUI:
    {context_data}

    PREGUNTA DEL INVERSIONISTA:
    {prompt}
    """

    # D. Llamada a la API (AQUÍ ESTÁ EL CAMBIO DE TEMPERATURA)
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="gpt-4o", # O el modelo que estés usando (gpt-4-turbo / gpt-3.5-turbo)
            messages=[
                {"role": "system", "content": JANUS_SYSTEM_PROMPT}, # Tu nuevo prompt
                *st.session_state.messages[:-1], # Historial previo
                {"role": "user", "content": full_prompt} # Pregunta + Contexto RAG
            ],
            temperature=0.2, # <--- ¡CRÍTICO! TEMPERATURA BAJA PARA PRECISIÓN
            stream=True,
        )
        response = st.write_stream(stream)
    
    st.session_state.messages.append({"role": "assistant", "content": response})

# --- FOOTER / DEBUG ---
with st.sidebar:
    st.divider()
    st.caption("MVP v1.0 | Desarrollado para MinComercio/VUI")
    st.caption("Motor: RAG + OpenAI GPT-4o")
    st.caption("Temp: 0.2 (Strict Mode)")
