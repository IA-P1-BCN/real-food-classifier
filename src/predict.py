"""A04: carga el mejor modelo guardado (M08) y predice el grupo NOVA de un producto.

El .joblib es un dict (formato de M05): model, features, pnns_categories, label_offset.
Aquí no se imputa ni se escala: XGBoost trata los nulos y la categoría por sí mismo.
Solo se le da a la fila el mismo formato que tenía X en el entrenamiento.
"""
import joblib
import pandas as pd

from src.modeling import CATEGORY, MODELS_DIR

# Modelo final de M08 (se genera con notebooks/04_m08_compare_models.ipynb, en Docker)
MODEL_PATH = MODELS_DIR / "m08_xgboost.joblib"


def load_bundle(path=MODEL_PATH):
    return joblib.load(path)


def to_model_input(values, bundle):
    """dict {columna: valor} -> DataFrame de una fila igual que el X de entrenamiento."""
    features = bundle["features"]

    # Si falta una columna, es un error de la app: mejor fallar que predecir con NaN sin avisar
    missing = set(features) - set(values)
    if missing:
        raise ValueError(f"Faltan columnas para el modelo: {sorted(missing)}")

    # Mismo orden de columnas que en el fit (XGBoost lo comprueba)
    X = pd.DataFrame([values])[features]

    numeric = [c for c in features if c != CATEGORY]
    X[numeric] = X[numeric].astype("float64")   # None -> NaN: XGBoost lo trata como en train

    # CLAVE: categorías FIJAS. Con .astype("category") sobre una sola fila, la única
    # categoría presente recibiría el código 0 ("Alcoholic beverages") y el modelo
    # predeciría mal sin dar ningún error
    if values[CATEGORY] not in bundle["pnns_categories"]:
        raise ValueError(f"Categoría desconocida: {values[CATEGORY]!r}")
    X[CATEGORY] = pd.Categorical(X[CATEGORY], categories=bundle["pnns_categories"])
    return X


def predict_nova(values, bundle):
    """Devuelve (clases NOVA 1-4, probabilidades, X que ha recibido el modelo)."""
    X = to_model_input(values, bundle)
    proba = bundle["model"].predict_proba(X)[0]
    classes = [int(c) + bundle["label_offset"] for c in bundle["model"].classes_]
    return classes, proba, X