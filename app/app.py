"""A01: app básica de Streamlit.

Formulario manual con los datos de la etiqueta -> predicción del grupo NOVA con probabilidades.
De momento usa un modelo provisional (placeholder); en A04 se cambia por el mejor modelo de M08.

Arrancar, desde la raíz del repo:   streamlit run app/app.py
(los colores del tema están en .streamlit/config.toml)
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
from src.label_extraction import PNNS_GROUPS_2  # noqa: E402  (las 42 subcategorías)
from src.modeling import (  # noqa: E402
    CATEGORY, TARGET, build_preprocessor, get_feature_columns, load_train,
)

# Nombre, descripción, color y emoji de cada grupo NOVA (semáforo: verde -> rojo)
NOVA = {
    1: {"name": "Sin procesar", "desc": "Alimentos naturales o mínimamente procesados",
        "color": "#2E7D32", "emoji": "🥦"},
    2: {"name": "Ingrediente culinario", "desc": "Aceites, sal, azúcar, mantequilla...",
        "color": "#F9A825", "emoji": "🧂"},
    3: {"name": "Procesado", "desc": "Conservas, quesos, pan artesano...",
        "color": "#EF6C00", "emoji": "🥫"},
    4: {"name": "Ultraprocesado", "desc": "Formulaciones industriales con aditivos",
        "color": "#C62828", "emoji": "🍭"},
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

# Nutrientes del formulario: (columna del dataset, etiqueta, obligatorio)
NUTRIENTS = [
    ("energy-kcal_100g", "Energía (kcal)", True),
    ("fat_100g", "Grasas (g)", True),
    ("saturated-fat_100g", "Grasas saturadas (g)", True),
    ("carbohydrates_100g", "Hidratos de carbono (g)", True),
    ("sugars_100g", "Azúcares (g)", True),
    ("fiber_100g", "Fibra (g) · opcional", False),   # en la UE no es obligatoria en la etiqueta
    ("proteins_100g", "Proteínas (g)", True),
    ("salt_100g", "Sal (g)", True),
]

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


# ---------------------------------------------------------------- Piezas visuales
def result_card(nova):
    """Tarjeta grande del color de la clase predicha."""
    info = NOVA[nova]
    st.markdown(f"""
<div class="nova-card" style="background:{info['color']}">
  <div class="big">{info['emoji']} NOVA {nova}</div>
  <div class="name">{info['name']}</div>
  <div class="desc">{info['desc']}</div>
</div>""", unsafe_allow_html=True)


def probability_bars(classes, proba):
    """Una barra por clase, con el color de su grupo NOVA."""
    html = ""
    for c, p in zip(classes, proba):
        info = NOVA[c]
        html += f"""
<div class="bar-row">
  <div class="bar-label"><span>NOVA {c} · {info['name']}</span><b>{p:.0%}</b></div>
  <div class="bar-bg"><div class="bar-fill" style="width:{p:.1%};background:{info['color']}"></div></div>
</div>"""
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------- Página
st.set_page_config(page_title="Real Food Classifier", page_icon="🥦", layout="wide")
st.html(Path(__file__).parent / "style.css")

# Barra lateral: explicación de los grupos NOVA
with st.sidebar:
    st.header("¿Qué es NOVA?")
    st.write("Clasifica los alimentos según **cuánto se han procesado**, no según sus nutrientes.")
    for n, info in NOVA.items():
        st.markdown(f"<span class='legend-chip' style='background:{info['color']}'></span>"
                    f"**NOVA {n} · {info['name']}**<br><small>{info['desc']}</small>",
                    unsafe_allow_html=True)
    st.divider()
    st.caption("Datos: Open Food Facts (ODbL) · Proyecto IA School, Factoría F5")

st.title("🥦 Real Food Classifier")
st.markdown("Copia los datos de la etiqueta y descubre su **grado de procesamiento**.")
st.warning("Modelo provisional: las predicciones todavía no son las del modelo final.", icon="🚧")

with st.spinner("Preparando el modelo (solo la primera vez)..."):
    model, cols = load_model()

form_col, result_col = st.columns([3, 2], gap="large")

# ---------------------------------------------------------------- Formulario
with form_col:
    # st.form: la app no se recalcula con cada tecla, solo al pulsar el botón
    with st.form("label_form", border=False):
        with st.container(border=True, key="card_producto"):
            st.subheader("🏷️ Producto")
            # index=None: nada elegido por defecto (así "unknown" nunca es el valor por defecto).
            # key: en A02/A03 Gemini podrá proponer la categoría escribiendo en st.session_state
            # format_func: muestra el nombre en español, pero category sigue valiendo el inglés
            category = st.selectbox("Categoría", CATEGORY_OPTIONS, index=None,
                                    format_func=lambda c: CATEGORY_ES.get(c, c),
                                    placeholder="Elige una categoría", key=CATEGORY)
        with st.container(border=True, key="card_nutrientes"):
            st.subheader("📊 Información nutricional")
            st.caption("Por 100 g o 100 ml. Deja vacío lo que no aparezca en la etiqueta.")
            values = {}
            left, right = st.columns(2)
            for i, (column, label, required) in enumerate(NUTRIENTS):
                # value=None deja la casilla vacía (en vez de 0): "no lo sé" no es lo mismo que "cero"
                with (left if i % 2 == 0 else right):
                    values[column] = st.number_input(
                        label, min_value=0.0, value=None, step=0.1, format="%.2f", key=column,
                        # Solo la energía lleva ayuda: muchas etiquetas ponen primero los kJ
                        help="Copia el valor en kcal, no en kJ" if column == "energy-kcal_100g" else None,
                    )
            values["fruits-vegetables-legumes_100g"] = st.number_input(
                "Frutas, verduras y legumbres (%) · opcional", min_value=0.0, max_value=100.0,
                value=None, step=1.0, key="fruits-vegetables-legumes_100g",
                help="Solo si la etiqueta lo indica")

        with st.container(border=True, key="card_ingredientes"):
            st.subheader("🧾 Ingredientes")
            no_list = st.toggle("El producto no tiene lista de ingredientes",
                                key="sin_lista_ingredientes")
            ingredients = st.text_area("Lista de ingredientes", key="ingredients_text",
                                       placeholder="Harina de trigo, azúcar, aceite de girasol...")
            additives_n = st.number_input("Número de aditivos (E-xxx)", min_value=0, value=0,
                                          step=1, key="additives_n")

            st.markdown("**¿Qué aparece en los ingredientes?**")
            markers = {}
            marker_columns = [c for c in cols["binary"] if c != "sin_lista_ingredientes"]
            grid = st.columns(3)
            for i, column in enumerate(marker_columns):
                with grid[i % 3]:
                    markers[column] = st.checkbox(MARKER_NAMES.get(column, column), key=column)

        submitted = st.form_submit_button("🔍 Predecir", type="primary", width="stretch")

# ---------------------------------------------------------------- Resultado
with result_col:
    st.markdown('<div class="result-anchor"></div>', unsafe_allow_html=True)  # marca para el CSS sticky
    st.subheader("Resultado")
    if not submitted:
        st.info("Rellena el formulario y pulsa **Predecir**.", icon="👈")
    else:
        # Validaciones básicas antes de predecir
        missing = [label for column, label, required in NUTRIENTS
                   if required and values[column] is None]
        errors = []
        if category is None:
            errors.append("Elige una categoría.")
        if missing:
            errors.append("Faltan nutrientes obligatorios: " + ", ".join(missing))
        if not no_list and not ingredients.strip():
            errors.append("Escribe los ingredientes o marca que no tiene lista.")

        if errors:
            for e in errors:
                st.error(e, icon="⚠️")
        else:
            # Mismas reglas que en src/build_dataset.py (paso 7)
            if no_list:
                n_ingredients, additives = 1, 0          # sin lista = un ingrediente, sin aditivos
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
            classes = [int(c) for c in model.classes_]
            predicted = classes[proba.argmax()]

            result_card(predicted)
            st.markdown("**Probabilidad de cada grupo**")
            probability_bars(classes, proba)

            with st.expander("Ver los datos que recibe el modelo"):
                st.dataframe(X.T.astype(str).rename(columns={0: "valor"}), width="stretch")