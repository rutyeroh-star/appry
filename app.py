import streamlit as st
import pandas as pd
from openai import OpenAI

# Configuración de la página
st.set_page_config(page_title="Intérprete alvaDesc", page_icon="🧬")

st.title("🧬 Intérprete de Descriptores Moleculares")
st.write("""
Esta aplicación permite analizar archivos CSV generados por software de quimioinformática (como alvaDesc o RDKit). 
Utiliza Inteligencia Artificial para interpretar el perfil farmacocinético y fisicoquímico de las moléculas.
""")

# 1. Cuadro de texto seguro para ingresar la API Key (en el menú lateral)
st.sidebar.header("🔑 Configuración")
api_key = st.sidebar.text_input("Ingresa la API Key de ChatGPT:", type="password")
st.sidebar.caption("La llave proporcionada por el profesor no se guarda, solo se usa durante esta sesión.")

# 2. Zona para subir el archivo CSV
archivo_csv = st.file_uploader("Sube tu archivo de descriptores (.csv)", type=["csv"])

if archivo_csv is not None:
    # Leer el archivo haciendo que Pandas adivine automáticamente el separador
    df = pd.read_csv(archivo_csv, sep=None, engine='python')
    
    st.write("### 📋 Vista previa de tus datos:")
    st.dataframe(df.head()) # Muestra las primeras filas para confirmar que cargó bien
    
    # 3. Botón para iniciar el análisis
    if st.button("🧠 Analizar e Interpretar Datos"):
        if not api_key:
            st.error("⚠️ Por favor, ingresa la API Key en el menú lateral izquierdo primero.")
        else:
            # Inicializar el cliente de OpenAI con la llave ingresada
            client = OpenAI(api_key=api_key)
            
            # --- SOLUCIÓN DEFINITIVA AL LÍMITE DE MEMORIA ---
            # 1. Tomamos máximo las primeras 15 columnas y 3 filas
            columnas_maximas = min(15, len(df.columns))
            df_reducido = df.iloc[:3, :columnas_maximas]
            
            # 2. Convertimos a texto
            datos_texto = df_reducido.to_csv(index=False)
            
            # 3. Cortamos el texto por la fuerza a 1500 caracteres (aprox 400 tokens)
            # Esto garantiza que NUNCA vuelva a exceder el límite de ChatGPT.
            datos_texto = datos_texto[:1500]
            
            # Instrucción detallada para la IA
            prompt_quimico = f"""
            Actúa como un experto en química computacional y descubrimiento de fármacos.
            A continuación te presento una muestra de datos de descriptores moleculares (en formato CSV):
            
            {datos_texto}
            
            Por favor, realiza lo siguiente:
            1. Identifica los principales descriptores presentes (ej. MW, LogP, TPSA, nHDon, nHAcc).
            2. Evalúa si, en general, estas moléculas cumplen con la Regla de los 5 de Lipinski o las reglas de Veber.
            3. Haz una predicción breve sobre su posible permeabilidad y biodisponibilidad oral.
            """
            
            # Llamada a la API mientras mostramos un mensaje de carga
            with st.spinner('Analizando el espacio químico...'):
                try:
                    respuesta = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt_quimico}]
                    )
                    
                    # Mostrar el resultado
                    st.success("¡Análisis completado!")
                    st.markdown("### 📊 Interpretación Farmacológica")
                    st.write(respuesta.choices[0].message.content)
                    
                except Exception as e:
                    st.error(f"Hubo un error al conectar con ChatGPT. Revisa la API Key. Detalle: {e}")
