import streamlit as st
import pandas as pd
import re
from io import StringIO
import google.generativeai as genai

# Configura tu clave de Gemini (guárdala en .streamlit/secrets.toml si es en producción)
GEMINI_KEY = "AIzaSyBWGE-ZJi-G5GGTcT5NFJOUe6dav14wkoE"  # ⚠️ Cambia esto por tu clave real
genai.configure(api_key=GEMINI_KEY)

# Prompt simplificado para ejecución básica (puedes colocar el tuyo completo si deseas)
prompt = """
Genera 200 registros sintéticos de estudiantes con 5 columnas:
Marital status, Course, Admission grade, Age at enrollment, Target.
Formato CSV, sin texto adicional, sin encabezados repetidos ni celdas vacías.
"""

def generar_datos_sinteticos():
    model = genai.GenerativeModel("gemini-2.0-flash-lite")
    chat = model.start_chat()
    respuesta = chat.send_message(prompt)
    texto_csv = limpiar_csv(respuesta.text)
    df = pd.read_csv(StringIO(texto_csv))
    return df

def limpiar_csv(texto):
    texto = re.sub(r"```csv\n|```", "", texto).strip()
    texto = re.sub(r"\n{2,}", "\n", texto)
    return texto

# INTERFAZ STREAMLIT
st.set_page_config(page_title="Generador de Datos Sintéticos", layout="centered")
st.title("🧠 Generador de Datos Sintéticos con Gemini")
st.markdown("Genera datos de estudiantes ficticios para pruebas y análisis de IA educativa.")

n_lotes = st.slider("Número de lotes a generar (cada uno tiene 200 registros)", 1, 3, 1)

if st.button("Generar datos"):
    st.info("Generando datos, por favor espera...")
    datos = []
    for i in range(n_lotes):
        try:
            df = generar_datos_sinteticos()
            datos.append(df)
            st.success(f"Lote {i+1} generado correctamente.")
        except Exception as e:
            st.error(f"Error en lote {i+1}: {e}")
    
    if datos:
        df_final = pd.concat(datos, ignore_index=True)
        st.dataframe(df_final.head(10))  # Muestra solo los primeros 10 registros
        csv = df_final.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Descargar CSV completo", data=csv, file_name="datos_sinteticos.csv", mime="text/csv")
