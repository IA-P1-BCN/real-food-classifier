"""A01: app de Streamlit con diseño "C · Lab".

Formulario con forma de etiqueta nutricional -> predicción del grupo NOVA con probabilidades.
De momento usa un modelo provisional (placeholder); en A04 se cambia por el mejor modelo de M08.

Arrancar, desde la raíz del repo:   streamlit run app/app.py
Tema (colores y fuentes): .streamlit/config.toml · Estilos propios: app/style.css
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

# Streamlit solo añade app/ al path: añadimos la raíz para poder importar src/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.build_dataset import contar_ingredientes  # noqa: E402  (misma regla que el dataset)
from src.ingredient_markers import detect_markers, find_additives  # noqa: E402
from src.label_checks import check_label  # noqa: E402  (mismas reglas que la limpieza de E01)
from src.label_extraction import PNNS_GROUPS_2  # noqa: E402  (las 42 subcategorías)
from src.modeling import (  # noqa: E402
    CATEGORY, TARGET, build_preprocessor, get_feature_columns, load_train,
)

# ---------------------------------------------------------------- Datos fijos de la interfaz
# Nombre, descripción y color de acento de cada grupo NOVA
NOVA = {
    1: {"name": "Sin procesar", "desc": "Naturales o mínimamente procesados", "color": "#7FD49B"},
    2: {"name": "Ingrediente culinario", "desc": "Aceites, sal, azúcar, mantequilla", "color": "#F2D06B"},
    3: {"name": "Procesado", "desc": "Conservas, quesos, pan artesano", "color": "#FFA36B"},
    4: {"name": "Ultraprocesado", "desc": "Formulaciones industriales con aditivos", "color": "#FF7A6B"},
}

# Nombre en español de cada categoría (lo que ve el usuario).
# El modelo se entrenó con los nombres en inglés: el desplegable devuelve el valor en inglés
# y solo traduce el texto que se muestra (format_func)
CATEGORY_ES = {
    "Alcoholic beverages": "Bebidas alcohólicas",
    "Appetizers": "Aperitivos salados (patatas fritas, snacks, crackers)",
    "Artificially sweetened beverages": "Bebidas con edulcorantes (light, zero)",
    "Baby foods": "Alimentos infantiles",
    "Baby milks": "Leches infantiles",
    "Biscuits and cakes": "Galletas y bizcochos",
    "Bread": "Pan",
    "Breakfast cereals": "Cereales de desayuno",
    "Cereals": "Cereales, arroz y pasta",
    "Cheese": "Quesos",
    "Chocolate products": "Chocolate y derivados",
    "Dairy desserts": "Postres lácteos",
    "Dressings and sauces": "Salsas y aliños",
    "Dried fruits": "Frutas desecadas",
    "Eggs": "Huevos",
    "Fats": "Aceites y grasas",
    "Fish and seafood": "Pescado y marisco",
    "Fruit juices": "Zumos de fruta",
    "Fruit nectars": "Néctares de fruta",
    "Fruits": "Frutas",
    "Ice cream": "Helados",
    "Legumes": "Legumbres",
    "Meat": "Carne",
    "Milk and yogurt": "Leche y yogur",
    "Nuts": "Frutos secos",
    "Offals": "Casquería",
    "One-dish meals": "Platos preparados",
    "Pastries": "Bollería",
    "Pizza pies and quiches": "Pizzas, empanadas y quiches",
    "Plant-based milk substitutes": "Bebidas vegetales",
    "Potatoes": "Patatas",
    "Processed meat": "Embutidos y carne procesada",
    "Salty and fatty products": "Productos salados y grasos",
    "Sandwiches": "Sándwiches y bocadillos",
    "Soups": "Sopas, caldos y cremas",
    "Sweetened beverages": "Refrescos y bebidas azucaradas",
    "Sweets": "Dulces y golosinas",
    "Teas and herbal teas and coffees": "Té, infusiones y café",
    "Unsweetened beverages": "Bebidas sin azúcar",
    "Vegetables": "Verduras y hortalizas",
    "Waters and flavored waters": "Aguas y aguas con sabor",
    "unknown": "Otra / no lo sé",
}
# Opciones ordenadas por su nombre en español (más fácil de encontrar); "unknown" al final
CATEGORY_OPTIONS = sorted((c for c in PNNS_GROUPS_2 if c != "unknown"),
                          key=lambda c: CATEGORY_ES.get(c, c)) + ["unknown"]

# Nombre legible de cada marcador tiene_* (los que usa el modelo)
MARKER_NAMES = {
    "tiene_aromas": "Aromas",
    "tiene_edulcorantes": "Edulcorantes",
    "tiene_jarabes": "Jarabes",
    "tiene_almidon_modificado": "Almidón mod.",
    "tiene_proteina_aislada": "Proteína aislada",
    "tiene_colorantes": "Colorantes",
    "tiene_emulsionantes": "Emulsionantes",
    "tiene_potenciadores": "Potenciadores",
    "tiene_aceite_palma": "Aceite de palma",
    "tiene_conservantes": "Conservantes",
    "tiene_sal": "Sal",
    "tiene_azucar": "Azúcar",
    "tiene_aceite": "Aceite",
}

# Filas de la etiqueta: (columna del dataset, texto, unidad, ¿subfila?, ¿obligatorio?, nota)
NUTRIENTS = [
    ("energy-kcal_100g", "Energía", "kcal", False, True, "en kcal, no kJ"),  # muchas etiquetas ponen primero los kJ
    ("fat_100g", "Grasas", "g", False, True, ""),
    ("saturated-fat_100g", "de las cuales saturadas", "g", True, True, ""),
    ("carbohydrates_100g", "Hidratos de carbono", "g", False, True, ""),
    ("sugars_100g", "de los cuales azúcares", "g", True, True, ""),
    ("fiber_100g", "Fibra", "g", False, False, "opcional"),   # en la UE no es obligatoria
    ("proteins_100g", "Proteínas", "g", False, True, ""),
    ("salt_100g", "Sal", "g", False, True, ""),
    ("fruits-vegetables-legumes_100g", "Frutas, verduras y legumbres", "%", False, False, "opcional"),
]

# Nombre corto para los mensajes de error (las subfilas no se entienden solas)
SHORT_NAMES = {"saturated-fat_100g": "Grasas saturadas", "sugars_100g": "Azúcares"}


# ---------------------------------------------------------------- Modelo
@st.cache_resource  # se entrena una sola vez; las siguientes recargas lo reutilizan
def load_model():
    """Modelo PROVISIONAL: árbol de decisión pequeño, entrenado al arrancar con el train.
    Solo sirve para comprobar que la app funciona. En A04 esta función cargará
    el mejor modelo guardado con joblib (M08) y el resto de la app no cambia."""
    df = load_train()
    cols = get_feature_columns(df)
    model = Pipeline([
        ("prep", build_preprocessor(cols, log_skewed=False, scale=False)),  # árboles: sin log ni escala
        ("clf", DecisionTreeClassifier(max_depth=8, min_samples_leaf=50,
                                       class_weight="balanced", random_state=42)),
    ])
    model.fit(df[cols["all"]], df[TARGET])
    return model, cols


def build_row(values, cols):
    """Pasa lo que ha escrito el usuario a un DataFrame de una fila con las columnas del modelo.
    Lo que se deja vacío va como NaN: el pipeline lo imputa igual que en el entrenamiento."""
    row = {c: values.get(c) for c in cols["all"]}
    return pd.DataFrame([row]).astype({c: "float64" for c in cols["numeric"]})


def autodetect_from_ingredients():
    """Se ejecuta cuando cambia la lista de ingredientes (al salir del cuadro de texto).
    Marca las casillas tiene_* y rellena el número de aditivos a partir del texto.
    Escribimos en st.session_state con la misma key que cada widget: así la casilla
    aparece marcada en la siguiente recarga. El usuario puede corregirlas después."""
    text = st.session_state.ingredients_text
    for column, found in detect_markers(text).items():
        if column in st.session_state.marker_columns:   # solo los marcadores que usa el modelo
            st.session_state[column] = found
    st.session_state.additives_n = len(find_additives(text))
    st.session_state.autodetected = bool(text.strip())


def reset_form():
    """Botón "Nuevo producto": borra todo lo escrito y el resultado.
    Igual que en la autodetección, escribimos en st.session_state con la key de cada
    widget, y en la siguiente recarga cada campo aparece vacío o sin marcar.
    (Borrar las keys con del no basta: el navegador vuelve a mandar los valores antiguos)"""
    st.session_state[CATEGORY] = None                 # desplegable sin elegir
    for column, *_ in NUTRIENTS:
        st.session_state[column] = None               # nutrientes vacíos
    st.session_state.ingredients_text = ""
    st.session_state.sin_lista_ingredientes = False
    st.session_state.additives_n = 0
    for column in st.session_state.marker_columns:
        st.session_state[column] = False              # casillas sin marcar
    st.session_state.autodetected = False
    st.session_state.result = {"nova": None, "proba": None, "errors": [], "warnings": []}


def clear_ingredients():
    """Se ejecuta al activar "no tiene lista de ingredientes": vacía el texto, desmarca
    las casillas y pone 0 aditivos. En el dataset, los productos sin lista tienen
    todos los tiene_* a 0, así que el modelo nunca ha visto otra combinación."""
    if st.session_state.sin_lista_ingredientes:
        st.session_state.ingredients_text = ""
        for column in st.session_state.marker_columns:
            st.session_state[column] = False
        st.session_state.additives_n = 0
        st.session_state.autodetected = False


# ---------------------------------------------------------------- Utilidades de HTML
def html(fragmento):
    """Pinta HTML con st.markdown.
    Quitamos las sangrías porque Markdown convierte las líneas con 4 espacios
    al inicio en bloques de código. Unimos con espacio para no pegar palabras."""
    limpio = " ".join(linea.strip() for linea in fragmento.splitlines())
    st.markdown(limpio, unsafe_allow_html=True)


def label_row(text, unit, sub=False, note=""):
    """Una fila de la etiqueta: nombre | (input, lo pone quien llama) | unidad.
    Devuelve la columna del medio para meter ahí el input."""
    c_name, c_input, c_unit = st.columns([3, 1.1, 0.45], vertical_alignment="center")
    with c_name:
        extra = f" <small>{note}</small>" if note else ""
        html(f'<div class="rf-row-label{" sub" if sub else ""}">{text}{extra}</div>')
    with c_unit:
        html(f'<span class="rf-unit">{unit}</span>')
    return c_input


# ---------------------------------------------------------------- Componentes visuales
def render_header():
    """Cabecera: etiqueta superior, píldora de estado y título grande."""
    html("""
        <header class="rf-header">
            <div class="rf-topbar">
                <span class="rf-eyebrow">// Clasificador NOVA · IA School</span>
                <span class="rf-pill"><span class="rf-dot"></span>Modelo provisional</span>
            </div>
            <h1 class="rf-title">Real Food<br><span class="rf-accent">Classifier</span></h1>
            <p class="rf-subtitle">Copia los datos de la etiqueta y descubre su <strong>grado de procesamiento</strong>.</p>
        </header>
    """)


def render_result(result):
    """Columna derecha: sello del resultado (o mensaje vacío / errores) + leyenda con probabilidades."""
    html('<div class="rf-result-head"><h2>Resultado</h2></div>')
    nova, proba = result["nova"], result["proba"]

    if result["errors"]:
        html(f'<div class="rf-empty rf-error">{"<br>".join(result["errors"])}</div>')
    elif nova is None:
        html('<div class="rf-empty">Rellena la etiqueta y pulsa <strong>Predecir</strong>.</div>')
    else:
        info = NOVA[nova]
        # --nova es una variable CSS: el sello toma el color del grupo predicho
        html(f"""
            <div class="rf-result">
                <div class="rf-stamp" style="--nova: {info['color']}">
                    <span class="rf-stamp-label">NOVA</span>
                    <span class="rf-stamp-num">{nova}</span>
                    <span class="rf-stamp-name">{info['name']}</span>
                </div>
                <div class="rf-result-text">
                    <span class="rf-result-name">{info['name']}</span>
                    <span class="rf-result-desc">{info['desc']}.</span>
                    <span class="rf-result-conf">Confianza: {proba[nova]:.0%}</span>
                </div>
            </div>
        """)
        # Avisos: valores raros pero posibles (p. ej. energía que parece estar en kJ)
        for warning in result.get("warnings", []):
            html(f'<div class="rf-empty rf-warning">{warning}</div>')

    # Leyenda: cada fila lleva de fondo una barra con la probabilidad de su grupo.
    # El color al 15 % de opacidad se consigue añadiendo "26" (hex) al color
    rows = ""
    for n, info in NOVA.items():
        p = proba.get(n) if proba else None
        active = " active" if n == nova else ""
        bar = f'style="--p: {p:.1%}; --c-soft: {info["color"]}26"' if p is not None else ""
        p_text = f'<span class="rf-legend-p" style="color: {info["color"]}">{p:.0%}</span>' if p is not None else ""
        rows += f"""
            <div class="rf-legend-row{active}" {bar}>
                <span class="rf-legend-n" style="color: {info['color']}">NOVA {n}</span>
                <span class="rf-legend-name">{info['name']}</span>
                <span class="rf-legend-desc">{info['desc']}</span>
                {p_text}
            </div>
        """
    title = '<div class="rf-legend-title">Probabilidad de cada grupo</div>' if proba else ""
    html(f'<div class="rf-legend">{title}{rows}</div>')
    html('<p class="rf-footer">Datos: Open Food Facts (ODbL) · Proyecto IA School, Factoría F5</p>')


# ---------------------------------------------------------------- Página
st.set_page_config(page_title="Real Food Classifier", page_icon="🥦", layout="wide",
                   initial_sidebar_state="collapsed")
st.html(Path(__file__).parent / "style.css")   # Streamlit lo envuelve en <style> por ser .css

# Guardamos el último resultado para que no se pierda si la página se recarga
st.session_state.setdefault("result", {"nova": None, "proba": None, "errors": [], "warnings": []})

with st.spinner("Preparando el modelo (solo la primera vez)..."):
    model, cols = load_model()
# Marcadores que usa el modelo (el callback de autodetección los necesita)
st.session_state.marker_columns = [c for c in cols["binary"] if c != "sin_lista_ingredientes"]

render_header()
col_label, col_result = st.columns([1.1, 1], gap="large")

# ----- Columna izquierda: la etiqueta nutricional -----
with col_label:
    # Contenedor normal (no st.form): así la lista de ingredientes puede marcar las casillas
    # en cuanto se pega. El modelo solo predice al pulsar el botón
    with st.container(key="label_card"):
        html('<h2 class="rf-label-title">Información nutricional</h2><div class="rf-rule-thin"></div>')

        html('<p class="rf-field-label">Categoría del producto</p>')
        # index=None: nada elegido por defecto (así "unknown" nunca es el valor por defecto).
        # format_func: muestra el nombre en español, pero category sigue valiendo el inglés.
        # key: en A03 Gemini podrá proponer la categoría escribiendo en st.session_state
        category = st.selectbox("Categoría del producto", CATEGORY_OPTIONS, index=None,
                                format_func=lambda c: CATEGORY_ES.get(c, c),
                                placeholder="Elige una categoría", key=CATEGORY,
                                label_visibility="collapsed")   # el texto ya va arriba

        html("""
            <div class="rf-rule-thick"></div>
            <div class="rf-per"><span>Por 100 g / 100 ml</span><span>vacío = no aparece</span></div>
        """)

        # Una fila por nutriente. El key "rows_..." activa el estilo de filas del CSS
        values = {}
        with st.container(key="rows_nutrientes"):
            for column, text, unit, sub, required, note in NUTRIENTS:
                with label_row(text, unit, sub, note):
                    # value=None deja la casilla vacía: "no lo sé" no es lo mismo que "cero"
                    values[column] = st.number_input(
                        text, min_value=0.0, max_value=100.0 if unit == "%" else None,
                        value=None, step=0.1, format="%.2f", placeholder="-",
                        label_visibility="collapsed", key=column)

        # ----- Ingredientes -----
        with st.container(key="sec_ingredientes"):
            html('<div class="rf-rule-thick"></div><h3 class="rf-section-title">Ingredientes</h3>')
            no_list = st.toggle("El producto no tiene lista de ingredientes",
                                key="sin_lista_ingredientes", on_change=clear_ingredients)
            # on_change: al salir del cuadro (o Ctrl+Enter) se marcan las casillas solas
            ingredients = st.text_area("Lista de ingredientes", key="ingredients_text",
                                       placeholder="Pega aquí la lista de ingredientes de la etiqueta",
                                       on_change=autodetect_from_ingredients,
                                       disabled=no_list)   # sin lista: los campos se bloquean

            with st.container(key="rows_aditivos"):
                with label_row("Número de aditivos", "E-xxx"):
                    # Sin value=: el valor inicial (0) va en session_state, porque el callback
                    # de autodetección también lo escribe (Streamlit avisa si se hacen las dos cosas)
                    st.session_state.setdefault("additives_n", 0)
                    additives_n = st.number_input("Número de aditivos", min_value=0, step=1,
                                                  label_visibility="collapsed", key="additives_n",
                                                  disabled=no_list)

            html('<p class="rf-field-label top">¿Qué aparece en los ingredientes?</p>')
            if st.session_state.get("autodetected"):
                html('<p class="rf-hint">Marcado automáticamente a partir de la lista. Revísalo y corrige si hace falta. Si cambias la lista, se vuelve a marcar desde cero.</p>')
            markers = {}
            grid = st.columns(3)
            for i, column in enumerate(st.session_state.marker_columns):
                with grid[i % 3]:
                    markers[column] = st.checkbox(MARKER_NAMES.get(column, column), key=column,
                                                  disabled=no_list)

        submitted = st.button("Predecir", type="primary", key="predict")
        # on_click: el callback se ejecuta antes de recargar, así la página ya sale vacía
        st.button("Nuevo producto", key="reset", on_click=reset_form,
                  help="Borra todos los datos y el resultado")

# ----- Lógica al pulsar Predecir -----
if submitted:
    # Validaciones básicas antes de predecir
    missing = [SHORT_NAMES.get(column, text) for column, text, unit, sub, required, note in NUTRIENTS
               if required and values[column] is None]
    errors = []
    if category is None:
        errors.append("Elige una categoría.")
    if missing:
        errors.append("Faltan nutrientes obligatorios: " + ", ".join(missing) + ".")
    if not no_list and not ingredients.strip():
        errors.append("Escribe los ingredientes o marca que no tiene lista.")
    # Comprobaciones lógicas (saturadas > grasas, energía en kJ...): src/label_checks.py
    check_errors, warnings = check_label(values, category)
    errors += check_errors

    if errors:
        st.session_state.result = {"nova": None, "proba": None, "errors": errors, "warnings": []}
    else:
        # Mismas reglas que en src/build_dataset.py (paso 7)
        if no_list:
            n_ingredients, additives = 1, 0          # sin lista = un ingrediente, sin aditivos
            markers = {k: False for k in markers}    # y ningún marcador (como en el dataset)
        else:
            n_ingredients, additives = contar_ingredientes(ingredients), additives_n

        values.update({k: int(v) for k, v in markers.items()})   # True/False -> 1/0
        values.update({
            CATEGORY: category,
            "sin_lista_ingredientes": int(no_list),
            "n_ingredientes": n_ingredients or None,  # 0 (texto ilegible) -> vacío
            "additives_n": additives,
        })

        X = build_row(values, cols)
        proba = model.predict_proba(X)[0]
        proba_by_class = {int(c): float(p) for c, p in zip(model.classes_, proba)}
        nova = max(proba_by_class, key=proba_by_class.get)   # la clase más probable
        st.session_state.result = {"nova": nova, "proba": proba_by_class, "errors": [],
                                   "warnings": warnings}

# ----- Columna derecha: resultado -----
with col_result:
    render_result(st.session_state.result)
