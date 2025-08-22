import streamlit as st
import pandas as pd
from io import StringIO
from src.model import load_data, configure_model, generate_synthetic_data, limpiar_texto_csv, estandarizar_csv, validate_synthetic_data

# Configuración de la página
st.set_page_config(page_title="Generador de Datos Sintéticos", layout="wide", page_icon=":bulb:")

# Estilo personalizado para mejorar la visibilidad y estética
st.markdown(
    """
    <style>
        .stApp {
            background-color: #F5F7FA;
        }
        div[data-testid="stButton"] button {
            background-color: #90C2E7;
            font-size: 16px;
            padding: 12px 20px;
            border-radius: 8px;
            border: none;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            width: 30%;
        }
        div[data-testid="stButton"] button:hover {
            background-color: #89CFF0;
            transform: scale(1.05);
        }
        h1, h2, h3 {
            font-family: 'Arial', sans-serif;
            color: #2E86C1;
        }
        p {
            color: #555;
            font-family: 'Arial', sans-serif;
            line-height: 1.6;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Título principal
with st.container():
    st.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: center; padding: 20px;">
            <img src="https://cdn-icons-png.flaticon.com/512/3845/3845897.png" alt="Birrete" style="width: 60px; margin-right: 15px;">
            <h1 style="color: #4A90E2; margin: 0;">Generador de Datos Sintéticos</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <p style='text-align: center; font-size:18px;'><br>Bienvenido al generador de datos sintéticos. Este sistema utiliza un modelo de inteligencia artificial para crear datos simulados de estudiantes universitarios, basados en un conjunto de datos reales.</p><br>
        """,
        unsafe_allow_html=True
    )

# Cargar datos reales y prompt
df_datos, prompt = load_data()

# Configurar el modelo
model = configure_model()

# Ayuda
with st.container():
    with st.expander(":question: **¿Necesitas ayuda?**"):
        st.markdown(
            """
            <div style="background-color: #F0F8FF; padding: 15px; border-radius: 5px;">
                <b style="font-size:16px; color:#2E86C1;">Guía de uso</b>
                <p style="text-align: left; font-size:15px; color:#333;">
                - Elige la cantidad de lotes de datos que deseas generar. (Cada lote genera aproximadamente 100 datos).<br>
                - Presiona <b>Generar datos sintéticos</b> para iniciar el proceso.<br>
                - Explora y filtra los datos generados en la tabla.<br>
                - Descarga los resultados en formato CSV para tus análisis.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

# Dataset original
with st.container():
    st.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: left; padding: 10px 0;">
            <img src="https://cdn-icons-png.flaticon.com/512/716/716784.png" alt="Carpeta" style="width: 35px; margin-right: 10px;">
            <h2 style="color: #2E86C1; margin: 0;">Dataset cargado automáticamente</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.dataframe(df_datos.head(), use_container_width=True)

# Estado para almacenar los datos generados
if "synthetic_data" not in st.session_state:
    st.session_state.synthetic_data = None
    st.session_state.synthetic_df = None

# Sección principal dividida en columnas
with st.container():
    st.markdown(
        """ 
        <div style="display: flex; align-items: center; justify-content: left; padding: 10px 0;">
            <img src="https://cdn-icons-png.flaticon.com/512/7743/7743674.png" alt="Configuración" style="width: 40px; margin-right: 10px;">
            <h2 style="color: #2E86C1; margin: 0;">Configuración de generación</h2>
        </div>
        """, 
        unsafe_allow_html=True
    )
    st.markdown("<p style='font-size:16px; color:#555; margin-bottom: 2px;'>Ajusta el número de lotes de datos sintéticos que deseas generar.</p>", unsafe_allow_html=True)
    num_lotes = st.slider("Selector de número de lotes", min_value=1, max_value=5, value=3, step=1, label_visibility="hidden")

    # Botón de generación
    if st.button(":rocket: **Generar datos sintéticos**"):
        with st.spinner("Procesando datos... Un momento, por favor..."):
            try:
                datos_sinteticos = generate_synthetic_data(df_datos, prompt, model, num_lotes * 20)
                if datos_sinteticos:
                    combined_data = ""
                    for batch_num, lote, _ in datos_sinteticos:
                        datos_limpios = limpiar_texto_csv(lote)
                        datos_estandar = estandarizar_csv(datos_limpios)
                        df_sintetico = pd.read_csv(StringIO(datos_estandar))
                        df_final = validate_synthetic_data(df_sintetico, df_datos)
                        batch_data = df_final.to_csv(index=False, header=(batch_num == 1))
                        combined_data += batch_data
                    st.session_state.synthetic_data = combined_data
                    try:
                        st.session_state.synthetic_df = pd.read_csv(StringIO(combined_data))
                        st.success(":white_check_mark: **¡Datos generados exitosamente!**")
                    except Exception:
                        st.warning(":warning: **No se pudo mostrar la tabla, pero puedes descargar los datos.**")
            except Exception as e:
                st.error(f":x: **Error al generar datos:** {e}")

# Mostrar la tabla y resumen si los datos están disponibles
with st.container():
    if st.session_state.synthetic_df is not None:
        st.markdown(
            """ 
            <div style="display: flex; align-items: center; justify-content: left; padding: 10px 0;">
                <img src="https://cdn-icons-png.flaticon.com/512/5984/5984404.png" alt="Vista previa" style="width: 40px; margin-right: 10px;">
                <h2 style="color: #2E86C1; margin: 0;">Vista previa de los datos generados</h2>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        num_filas = len(st.session_state.synthetic_df)
        st.info(f":package: Se generaron **{num_filas} registros** sintéticos.")
        
        st.dataframe(
            st.session_state.synthetic_df.head(20),
            use_container_width=True,
            height=400
        )

        # Contenedor para la sección de filtrado
        with st.container():
            st.markdown(
                """ 
                <div style="display: flex; align-items: center; justify-content: left; padding: 10px 0;">
                    <img src="https://cdn-icons-png.flaticon.com/512/566/566737.png" alt="Filtrar datos" style="width: 20px; margin-right: 10px;">
                    <h3 style="color: #2E86C1; margin: 0;">Filtrar datos generados sintéticos</h3>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            with st.expander("Mostrar opciones de filtrado"):
                columna_filtro = st.selectbox("Filtrar por columna", st.session_state.synthetic_df.columns)
                valor_filtro = st.text_input(f"Valor a buscar en {columna_filtro}")
                
                if valor_filtro:
                    df_filtrado = st.session_state.synthetic_df[
                        st.session_state.synthetic_df[columna_filtro].astype(str).str.contains(valor_filtro, case=False, na=False)
                    ]
                    st.dataframe(df_filtrado, use_container_width=True)

# Botón para descargar
with st.container():
    if st.session_state.synthetic_data is not None:
        st.download_button(
            label=":floppy_disk: Descargar datos como CSV",
            data=st.session_state.synthetic_data,
            file_name="datos_sinteticos_gemini.csv",
            mime="text/csv"
        )
    else:
        st.info(":arrow_up: Primero genera los datos.")

# Pie de página
with st.container():
    st.markdown("---")
    st.markdown(":technologist: *Desarrollado por Laura Valentina Caicedo y Juan José Muñoz*")

# Mensaje sobre los datos generados
with st.container():
    st.warning(":warning: *Nota: Estos datos han sido generados por un modelo de inteligencia artificial y pueden contener errores, incoherencias o información ficticia. Úselos únicamente con fines educativos o de prueba.*")
