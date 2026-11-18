"""F02: métricas en vivo calculadas con el feedback guardado (data/feedback/feedback.csv).

Página aparte, sin enlace en la app: el CSS de la app oculta el menú lateral,
así que se entra por URL: http://localhost:8501/metricas
"""
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Igual que en app.py, pero un nivel más arriba: app/pages/metricas.py -> raíz del repo
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.feedback import FEEDBACK_PATH  # noqa: E402  (la misma ruta en la que escribe la F01)

CLASES = [1, 2, 3, 4]
REFRESCO = "5s"   # cada cuánto se mira si ha llegado feedback nuevo


@st.cache_data
def cargar_feedback(mtime):
    """Lee el CSV. `mtime` (fecha de modificación) solo es la clave de la caché:
    si el fichero no ha cambiado, no se vuelve a leer; si cambia, sí."""
    df = pd.read_csv(FEEDBACK_PATH)
    df = df.drop_duplicates("id_feedback", keep="last")   # doble clic en "Enviar respuesta"
    df = df[df["nova_predicho"].isin(CLASES) & df["nova_correcto"].isin(CLASES)]
    return df.astype({"nova_predicho": int, "nova_correcto": int})


def dibujar_matriz(df):
    # Filas = NOVA real (lo que dice la persona), columnas = lo que predijo el modelo
    cm = confusion_matrix(df["nova_correcto"], df["nova_predicho"], labels=CLASES)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False,
                xticklabels=CLASES, yticklabels=CLASES, ax=ax)
    ax.set_xlabel("NOVA predicho por el modelo")
    ax.set_ylabel("NOVA real (según el usuario)")
    return fig


st.set_page_config(page_title="Métricas en vivo · Real Food Classifier", page_icon="📊",
                   layout="wide")
st.title("Métricas en vivo")
st.caption("Calculadas con las respuestas a «¿Ha acertado el modelo?» que guarda la app.")


# El fragmento se vuelve a ejecutar solo cada REFRESCO sin recargar toda la página:
# así la pantalla se actualiza cuando llega feedback nuevo
@st.fragment(run_every=REFRESCO)
def metricas_en_vivo():
    if not FEEDBACK_PATH.exists() or FEEDBACK_PATH.stat().st_size == 0:
        st.info("Todavía no hay feedback. Haz una predicción y responde «¿Ha acertado el modelo?».")
        return

    df = cargar_feedback(FEEDBACK_PATH.stat().st_mtime)

    # Filtro por modelo: útil cuando la app pase al modelo reentrenado (F03)
    versiones = sorted(df["modelo_version"].dropna().unique())
    version = st.selectbox("Modelo", ["Todos"] + versiones, key="version")
    if version != "Todos":
        df = df[df["modelo_version"] == version]
    if df.empty:
        st.info("No hay feedback para este modelo.")
        return

    y_real, y_pred = df["nova_correcto"], df["nova_predicho"]
    # Solo las clases que aparecen en el feedback: una clase sin respuestas
    # contaría como F1 = 0 y bajaría el F1 macro sin motivo (pasa con pocas respuestas)
    clases_presentes = sorted(set(y_real) | set(y_pred))

    c1, c2, c3 = st.columns(3)
    c1.metric("Respuestas", len(df))
    c2.metric("Accuracy", f"{accuracy_score(y_real, y_pred):.1%}")
    c3.metric("F1 macro", f"{f1_score(y_real, y_pred, labels=clases_presentes, average='macro', zero_division=0):.3f}")

    if len(df) < 30:
        st.warning("Hay pocas respuestas: las métricas aún no son fiables.")

    col_matriz, _ = st.columns([1, 1])
    with col_matriz:
        fig = dibujar_matriz(df)
        st.pyplot(fig)
        plt.close(fig)

    st.caption(f"Actualizado a las {datetime.now():%H:%M:%S} · se refresca cada {REFRESCO}")


metricas_en_vivo()
