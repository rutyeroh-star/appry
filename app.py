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
    
    st.write("### ⚙️ Ajuste de Datos para la IA")
    st.info("Para no saturar la memoria de ChatGPT, selecciona las columnas más relevantes y la cantidad de moléculas.")
    
    # 1. Selector inteligente de columnas
    todas_columnas = df.columns.tolist()
    # Intentar sugerir automáticamente columnas importantes por su nombre
    columnas_sugeridas = [col for col in todas_columnas if any(kw in col.upper() for kw in ['MW', 'LOGP', 'TPSA', 'HACC', 'HDON', 'NAME', 'ID'])]
    if not columnas_sugeridas:
        columnas_sugeridas = todas_columnas[:10] # Si no encuentra comunes, sugiere las primeras 10
        
    columnas_seleccionadas = st.multiselect(
        "Selecciona los descriptores (columnas) a analizar:", 
        options=todas_columnas, 
        default=columnas_sugeridas
    )
    
    # 2. Control de filas (moléculas)
    num_filas = st.slider("Número de moléculas (filas) a incluir:", min_value=1, max_value=50, value=5)
    
    # 3. Botón para iniciar el análisis
    if st.button("🧠 Analizar e Interpretar Datos"):
        if not api_key:
            st.error("⚠️ Por favor, ingresa la API Key en el menú lateral izquierdo primero.")
        elif not columnas_seleccionadas:
            st.warning("⚠️ Por favor, selecciona al menos una columna de la lista.")
        else:
            # Inicializar el cliente de OpenAI con la llave ingresada
            client = OpenAI(api_key=api_key)
            
            # --- NUEVA SOLUCIÓN AMPLIADA Y DINÁMICA ---
            # Tomamos exactamente las filas y columnas que elegiste en la interfaz
            df_reducido = df.iloc[:num_filas][columnas_seleccionadas]
            
            # Convertimos a texto
            datos_texto = df_reducido.to_csv(index=False)
            
            # Ampliamos el límite seguro a 35,000 caracteres (aprox. 9,000 tokens)
            # Esto te permite mandar mucha más data sin romper ChatGPT
            if len(datos_texto) > 35000:
                st.warning("⚠️ Los datos seleccionados rozan el límite de memoria. Se analizará la mayor cantidad posible (se recortará el exceso).")
                datos_texto = datos_texto[:35000]
            
            # --- NUEVO: Verificador visual ---
            # Esto te permite ver exactamente qué texto está recibiendo la IA
            with st.expander("👀 Ver los datos exactos que se enviarán a ChatGPT"):
                st.text(datos_texto)
            
            # --- NUEVO: Prompt mejorado y más estricto ---
            prompt_quimico = f"""
            Actúa como un experto en química computacional y descubrimiento de fármacos.
            A continuación te presento los valores exactos calculados para una serie de moléculas. 
            Están en formato de tabla (CSV):
            
            ```csv
            {datos_texto}
            ```
            
            BASADO ESTRICTAMENTE EN LOS NÚMEROS DE LA TABLA ARRIBA, realiza lo siguiente:
            1. Menciona explícitamente los valores numéricos que estás leyendo de la tabla (ejemplo: "La primera molécula tiene un MW de X y un LogP de Y").
            2. Evalúa si estas moléculas cumplen con la Regla de los 5 de Lipinski o las reglas de Veber, basándote en esos valores específicos.
            3. Haz una predicción breve sobre su posible permeabilidad y biodisponibilidad oral.
            
            NOTA: Si falta algún descriptor clave en la tabla, analiza únicamente con los datos numéricos que SÍ están presentes. No respondas que no tienes datos, usa los que están en la tabla.
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
