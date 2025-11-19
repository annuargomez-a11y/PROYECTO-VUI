````mermaid`
graph LR
    User((Usuario)) -->|Navegador Web| Streamlit[Streamlit Cloud<br/>(Hosting de la App)]
    GitHub[GitHub<br/>(Código y PDFs)] -->|Despliegue Automático| Streamlit
    Streamlit <-->|Consulta y Respuesta| OpenAI[OpenAI API<br/>(Cerebro y Traductor)]
    
    style Streamlit fill:#ff4b4b,stroke:#333,stroke-width:2px,color:white
    style OpenAI fill:#74aa9c,stroke:#333,stroke-width:2px,color:white
    style GitHub fill:#f0f0f0,stroke:#333,stroke-width:2px,color:black
    ````mermaid`

    
