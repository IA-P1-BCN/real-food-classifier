"""Detecta los marcadores tiene_* y los aditivos a partir del texto de ingredientes.

En el dataset, los marcadores salen de las etiquetas en inglés que calcula Open Food Facts
(ingredients_tags, additives_tags; ver crear_variables en src/build_dataset.py).
En la app solo tenemos el texto que pega el usuario (en español), así que aquí
reproducimos esas mismas reglas con palabras clave en español y números E.

Es una aproximación: sirve para marcar las casillas automáticamente, y el usuario
puede corregirlas antes de predecir.

Uso:  from src.ingredient_markers import detect_markers, find_additives
"""
import re
import unicodedata

# Nombres frecuentes de aditivos que suelen aparecer sin su número E.
# Sin acentos y en minúsculas, porque el texto se normaliza así antes de buscar
ADDITIVE_NAMES = {
    r"metabisulfito": "E223",
    r"(?<![a-z])bisulfito": "E222",
    r"(?<![a-z])sulfito sodico": "E221",   # (?<![a-z]): que no sea parte de "metabisulfito"
    r"sorbato potasico": "E202",
    r"acido sorbico": "E200",
    r"benzoato sodico": "E211",
    r"nitrito sodico": "E250",
    r"nitrato potasico": "E252",
    r"propionato calcico": "E282",
    r"acido citrico": "E330",
    r"acido ascorbico": "E300",
    r"ascorbato sodico": "E301",
    r"acido lactico": "E270",
    r"lecitina": "E322",
    r"mono ?y ?digliceridos": "E471",
    r"goma xantana": "E415",
    r"goma guar": "E412",
    r"goma garrofin": "E410",
    r"goma arabiga": "E414",
    r"carragenan": "E407",
    r"pectina": "E440",
    r"bicarbonato sodico": "E500",
    r"difosfato": "E450",
    r"polifosfato": "E452",
    r"glutamato monosodico": "E621",
    r"inosinato": "E631",
    r"guanilato": "E627",
    r"curcumina": "E100",
    r"carmin": "E120",
    r"sucralosa": "E955",
    r"aspartamo": "E951",
    r"acesulfamo": "E950",
    r"sacarina": "E954",
    r"ciclamato": "E952",
    r"glucosidos de esteviol": "E960",
}

# Palabras clave por marcador (equivalentes en español de las reglas de build_dataset.py).
# Los que en el dataset salen de los números E se completan además con find_additives
KEYWORDS = {
    "tiene_aromas": r"\baromas?\b|aromatizantes?|\bflavou?rings?\b",
    "tiene_edulcorantes": r"\bedulcorantes?\b|sucralosa|aspartamo|acesulfamo|sacarina|ciclamato|esteviol",
    "tiene_jarabes": r"\bjarabes?\b|\bsirope\b|maltodextrina|dextrosa|\bsyrup\b",
    "tiene_almidon_modificado": r"almidon(?:es)?(?: de [a-z]+)? modificados?",
    "tiene_proteina_aislada": r"proteinas? aisladas?|aislado de proteina|hidrolizad|\bsuero\b|lactosuero|caseinato",
    "tiene_colorantes": r"\bcolorantes?\b",
    "tiene_emulsionantes": r"\bemulgentes?\b|\bemulsionantes?\b|\bestabilizantes?\b|\bespesantes?\b|\bgelificantes?\b",
    "tiene_potenciadores": r"potenciador(?:es)? del sabor|glutamato",
    "tiene_aceite_palma": r"\bpalma\b|palmiste",
    "tiene_conservantes": r"\bconserva(?:dor|nte)s?\b",
    "tiene_sal": r"\bsal\b|\bsalt\b",
    "tiene_azucar": r"\bazucar(?:es)?\b|\bsugar\b",
    "tiene_aceite": r"\baceites?\b|\boil\b",
}

# Rangos de números E, iguales a los de crear_variables (src/build_dataset.py)
E_RANGES = {
    "tiene_colorantes": r"^E1\d\d$",
    "tiene_conservantes": r"^E2(?:[0-4]\d|5[0-2]|8[0-5])$",
    "tiene_emulsionantes": r"^E4\d\d$",
    "tiene_potenciadores": r"^E6[23]\d$",
    "tiene_edulcorantes": r"^E9[56]\d$",
    "tiene_almidon_modificado": r"^E14\d\d$",
}


def normalize(text):
    """Minúsculas y sin acentos: 'AZÚCAR' y 'azucar' se buscan igual."""
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    return text.lower()


def find_additives(text):
    """Aditivos del texto como números E sin repetir, p. ej. ['E223', 'E330'].
    Reconoce 'E330', 'E-330', 'E 330', 'e330i' y los nombres de ADDITIVE_NAMES."""
    norm = normalize(text)
    codes = {f"E{num}" for num in re.findall(r"\be[ -]?(\d{3,4})[a-z]?\b", norm)}
    codes |= {code for pattern, code in ADDITIVE_NAMES.items() if re.search(pattern, norm)}
    return sorted(codes)


def detect_markers(text):
    """Devuelve {columna tiene_*: True/False} según el texto de ingredientes."""
    norm = normalize(text)
    additives = find_additives(text)
    markers = {}
    for column, pattern in KEYWORDS.items():
        found = bool(re.search(pattern, norm))
        # Si el marcador también se define por números E, basta con que aparezca uno
        if column in E_RANGES:
            found = found or any(re.match(E_RANGES[column], code) for code in additives)
        markers[column] = found
    return markers


if __name__ == "__main__":
    # Prueba rápida:  python src/ingredient_markers.py "harina, azúcar, sal, E330"
    import sys
    texto = " ".join(sys.argv[1:])
    print("Aditivos:", find_additives(texto))
    print("Marcadores:", [k for k, v in detect_markers(texto).items() if v])