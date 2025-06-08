import os
import re
import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Función para cargar datos reales y prompt
def load_data():
    df_datos = pd.read_csv("data/datasetCompleto.csv", sep=",")
    with open("./data/prompt.txt", "r", encoding="utf-8") as f:
        prompt = f.read()
    return df_datos, prompt

# Configurar el modelo de Gemini
def configure_model():
    genai.configure(api_key=os.getenv("GEMINI_KEY"))
    return genai.GenerativeModel(
        "gemini-2.0-flash-lite",
        generation_config={
            "temperature": 0.5,
            "top_p": 0.8,
            "top_k": 40,
        }
    )

# Función para generar datos sintéticos
def generate_synthetic_data(df_datos, prompt, model, num_lotes):
    datos_totales = []
    for i in range(num_lotes):
        try:
            chat_session = model.start_chat(history=[])
            respuesta = chat_session.send_message(prompt.format(df_datos=df_datos))
            lote = respuesta.text
            row_count = len(lote.split('\n')) - 1  
            datos_totales.append((i + 1, lote, row_count))
        except Exception as e:
            raise Exception(f"Error en lote {i + 1}: {e}")
    return datos_totales

# Función para limpiar texto CSV
def limpiar_texto_csv(texto):
    texto = re.sub(r'```csv\n|```', '', texto)
    texto = texto.strip()
    texto = re.sub(r'\n+', '\n', texto)
    return texto

# Función para estandarizar CSV
def estandarizar_csv(texto):
    lineas = texto.split('\n')
    lineas_limpias = []
    for linea in lineas:
        linea = re.sub(r'\s*,\s*', ',', linea.strip())
        linea = re.sub(r'^,|,$', '', linea)
        lineas_limpias.append(linea)
    return '\n'.join(lineas_limpias)

# Función para validar datos sintéticos
def validate_synthetic_data(df_sintetico, df_real):
    # Eliminar filas con nombres de columnas
    idx_eliminar = df_sintetico[df_sintetico.apply(lambda row: row.astype(str).str.contains('Marital status').any(), axis=1)].index
    df_sintetico = df_sintetico.drop(idx_eliminar)

    # Eliminar valores nulos y duplicados
    df_sintetico = df_sintetico.dropna()
    df_sintetico = df_sintetico.drop_duplicates(keep='first')

    # Renombrar columna
    df_sintetico.rename(columns={'Daytime/evening attendance': 'Daytime/evening attendance\t'}, inplace=True)

    # Verificar columnas
    if list(df_real.columns) != list(df_sintetico.columns):
        raise ValueError("Las columnas de real_data y synthetic_data no coinciden")

    # Convertir tipos de datos
    tipos_datos_objetivo = {
        'Marital status': 'int64',
        'Application mode': 'int64',
        'Application order': 'int64',
        'Course': 'int64',
        'Daytime/evening attendance\t': 'int64',
        'Previous qualification': 'int64',
        'Previous qualification (grade)': 'float64',
        'Nacionality': 'int64',
        "Mother's qualification": 'int64',
        "Father's qualification": 'int64',
        "Mother's occupation": 'int64',
        "Father's occupation": 'int64',
        'Admission grade': 'float64',
        'Displaced': 'int64',
        'Educational special needs': 'int64',
        'Debtor': 'int64',
        'Tuition fees up to date': 'int64',
        'Gender': 'int64',
        'Scholarship holder': 'int64',
        'Age at enrollment': 'int64',
        'International': 'int64',
        'Curricular units 1st sem (credited)': 'int64',
        'Curricular units 1st sem (enrolled)': 'int64',
        'Curricular units 1st sem (approved)': 'int64',
        'Curricular units 1st sem (grade)': 'float64',
        'Curricular units 2nd sem (credited)': 'int64',
        'Curricular units 2nd sem (enrolled)': 'int64',
        'Curricular units 2nd sem (approved)': 'int64',
        'Curricular units 2nd sem (grade)': 'float64',
        'Unemployment rate': 'float64',
        'Inflation rate': 'float64',
        'GDP': 'float64',
        'Target': 'int64'
    }

    for columna, tipo_objetivo in tipos_datos_objetivo.items():
        if columna in df_sintetico.columns:
            try:
                if tipo_objetivo == 'int64':
                    df_sintetico[columna] = pd.to_numeric(df_sintetico[columna], errors='coerce').astype('int64')
                elif tipo_objetivo == 'float64':
                    df_sintetico[columna] = pd.to_numeric(df_sintetico[columna], errors='coerce').astype('float64')
            except Exception as e:
                raise Exception(f"Error al convertir la columna '{columna}' a '{tipo_objetivo}': {e}")
            
    return df_sintetico
