import uuid
from datetime import datetime, timezone

import pandas as pd

from src.modeling import ROOT

FEEDBACK_PATH = ROOT / "data" / "feedback" / "feedback.csv" 

FEATURES = [
    "pnns_groups_2", "additives_n",
    "energy-kcal_100g", "fat_100g", "saturated-fat_100g", "carbohydrates_100g",
    "sugars_100g", "fiber_100g", "proteins_100g", "salt_100g",
    "fruits-vegetables-legumes_100g", "n_ingredientes",
    "tiene_aromas", "tiene_edulcorantes", "tiene_jarabes", "tiene_almidon_modificado",
    "tiene_proteina_aislada", "tiene_colorantes", "tiene_emulsionantes",
    "tiene_potenciadores", "tiene_aceite_palma", "tiene_conservantes",
    "tiene_sal", "tiene_azucar", "tiene_aceite", "sin_lista_ingredientes",
]

META = [
    "id_feedback",      # uuid: detecta duplicados (Streamlit puede guardar dos veces lo mismo)
    "fecha",            # ISO 8601 en UTC, p. ej. 2026-11-17T10:32:00Z
    "origen",           # "foto" o "manual"
    "modelo_version",   # nombre del .joblib que hizo la predicción, p. ej. "m08_xgboost"
    "nova_predicho",    # NOVA que dijo el modelo (1-4)
    "confianza",        # probabilidad de la clase predicha (0-1)
    "es_correcto",      # el usuario dice si la predicción es correcta: "si" / "no"
    "nova_correcto",    # NOVA real según el usuario (1-4); si es_correcto = "si", igual a nova_predicho
    "product_name",     # solo para revisar a mano y para el filtro de fuga con el test
    "ingredients_text", # opcional, solo para revisar a mano
]
COLUMNS = META + FEATURES


NUTRIENTES = [c for c in FEATURES if c.endswith("_100g")]

def _validar(fila):
    """Lanza ValueError con un mensaje claro si la fila no sirve para reentrenar."""
    reglas = [
        (set(fila) <= set(COLUMNS),
         f"Columnas desconocidas: {sorted(set(fila) - set(COLUMNS))}"),
        (fila.get("nova_predicho") in (1, 2, 3, 4),
         "nova_predicho debe ser un número entero de 1 a 4"),
        (fila.get("nova_correcto") in (1, 2, 3, 4),
         "nova_correcto debe ser un número entero de 1 a 4"),
        (fila.get("es_correcto") in ("si", "no"),
         'es_correcto debe ser "si" o "no"'),
        ((fila.get("es_correcto") == "si") == (fila.get("nova_correcto") == fila.get("nova_predicho")),
         "es_correcto no coincide con nova_predicho y nova_correcto"),
        (not all(pd.isna(fila.get(c)) for c in NUTRIENTES),
         "Falta la tabla nutricional: al menos un nutriente debe tener valor"),
    ]
    for se_cumple, mensaje in reglas:
        if not se_cumple:
            raise ValueError(mensaje)


def es_valida(fila):
    try:
        _validar(fila)
    except ValueError:
        return False
    return True


def guardar_feedback(fila, path=FEEDBACK_PATH):
    """Valida una fila de feedback y la añade al CSV. Devuelve True si se guardó.
    `fila` es un dict con las columnas de COLUMNS; id_feedback y fecha se rellenan solos.
    Si la fila no es válida lanza ValueError (la app muestra el mensaje al usuario)."""
    fila = dict(fila)
    fila.setdefault("id_feedback", str(uuid.uuid4()))
    fila.setdefault("fecha", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    _validar(fila)

    path.parent.mkdir(parents=True, exist_ok=True)
    primera_vez = not path.exists()   # solo la primera vez se escribe la cabecera
    pd.DataFrame([fila]).reindex(columns=COLUMNS).to_csv(
        path, mode="a", header=primera_vez, index=False)
    return True