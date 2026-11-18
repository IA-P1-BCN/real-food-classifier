"""A02: extrae los datos de la foto de una etiqueta con Gemini.

Uso desde la app:   from src.label_extraction import extract_label
Uso en terminal:    python src/label_extraction.py foto_tabla.jpg foto_ingredientes.jpg
"""
import io
import json
import logging
import os
import sys
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

load_dotenv()  # lee GEMINI_API_KEY (y GEMINI_MODEL, opcional) del fichero .env

DEFAULT_MODEL = "gemini-3.5-flash-lite"   # rápido (~5 s) y lee bien tablas e ingredientes
MAX_PHOTOS = 4  # fotos por producto (delante, detrás, laterales)
RETRY_CODES = [429, 500, 502, 503, 504]  # errores temporales: límite de peticiones o servidor saturado
TIMEOUT_MS = 60_000                      # máximo por intento: así nunca se queda esperando para siempre
MAX_SIDE = 2048                          # px del lado largo: suficiente para leer la letra pequeña

# Las 42 subcategorías de pnns_groups_2 del dataset limpio (incluye "unknown").
# Van como enum en el esquema: Gemini solo puede devolver uno de estos nombres exactos.
PNNS_GROUPS_2 = [
    "Alcoholic beverages", "Appetizers", "Artificially sweetened beverages", "Baby foods",
    "Baby milks", "Biscuits and cakes", "Bread", "Breakfast cereals", "Cereals", "Cheese",
    "Chocolate products", "Dairy desserts", "Dressings and sauces", "Dried fruits", "Eggs",
    "Fats", "Fish and seafood", "Fruit juices", "Fruit nectars", "Fruits", "Ice cream",
    "Legumes", "Meat", "Milk and yogurt", "Nuts", "Offals", "One-dish meals", "Pastries",
    "Pizza pies and quiches", "Plant-based milk substitutes", "Potatoes", "Processed meat",
    "Salty and fatty products", "Sandwiches", "Soups", "Sweetened beverages", "Sweets",
    "Teas and herbal teas and coffees", "Unsweetened beverages", "Vegetables",
    "Waters and flavored waters", "unknown",
]
Category = Enum("Category", {name: name for name in PNNS_GROUPS_2})

# Nombre en el esquema de Gemini -> nombre de la columna en el dataset (lo que espera el modelo)
NUTRIENT_COLUMNS = {
    "energy_kcal": "energy-kcal_100g",
    "fat": "fat_100g",
    "saturated_fat": "saturated-fat_100g",
    "carbohydrates": "carbohydrates_100g",
    "sugars": "sugars_100g",
    "fiber": "fiber_100g",
    "proteins": "proteins_100g",
    "salt": "salt_100g",
    "fruits_vegetables_legumes": "fruits-vegetables-legumes_100g",
}


# ---------------------------------------------------------------- Esquema de respuesta
# Gemini rellena exactamente esta estructura (structured output). None = no aparece en la foto.
class Nutrients(BaseModel):
    energy_kcal: float | None = Field(None, description="Energy in kcal per 100 g/ml")
    fat: float | None = Field(None, description="Fat in g per 100 g/ml")
    saturated_fat: float | None = Field(None, description="Saturated fat in g per 100 g/ml")
    carbohydrates: float | None = Field(None, description="Carbohydrates in g per 100 g/ml")
    sugars: float | None = Field(None, description="Sugars in g per 100 g/ml")
    fiber: float | None = Field(None, description="Fibre in g per 100 g/ml")
    proteins: float | None = Field(None, description="Protein in g per 100 g/ml")
    salt: float | None = Field(None, description="Salt in g per 100 g/ml")
    fruits_vegetables_legumes: float | None = Field(
        None, description="Percentage of fruits, vegetables and legumes, only if stated")


class LabelExtraction(BaseModel):
    is_food_label: bool = Field(description="False if the photo is not a food label or is unreadable")
    has_nutrition_table: bool = Field(description="True only if a nutrition table is visible")
    product_name: str | None = None
    category: Category
    nutrients: Nutrients
    ingredients_text: str | None = Field(None, description="Ingredient list, verbatim")
    additives: list[str] = Field(description="E-numbers found, e.g. ['E330', 'E471']")


PROMPT = """You read photos of food product labels (mostly from Spain).
You receive one or more photos of the SAME product (different sides of the package).
Combine the information from all photos into a single answer. Rules:
- Use only what is visible in the photos. Never guess or invent values: use null if not visible.
- Nutrients: use the "per 100 g" or "per 100 ml" column, never the per-serving column.
- If energy is only in kJ, convert to kcal (kJ / 4.184). If only sodium is given, salt = sodium * 2.5.
- has_nutrition_table: false if no photo shows a nutrition table. Then all nutrients are null.
- is_food_label: false if the photos are not a food label or are too blurry to read.
- ingredients_text: copy the full ingredient list as written. If it appears in several languages,
  copy only the Spanish one (or the first one if there is no Spanish). Use null only if no
  ingredient list is visible in any photo. If you found additives, the list was visible: copy it.
- additives: E-numbers in the ingredients, written as 'E' + number (e.g. E330). If an additive is
  named without its number (e.g. "citric acid"), add its E-number only if you are sure.
- category: the closest category to the product. Use "unknown" if you cannot tell.
  Salty crackers, crisps and salty snacks are "Appetizers"; "Biscuits and cakes" is only for
  sweet biscuits, cookies and cakes."""


class LabelExtractionError(Exception):
    """Error técnico (sin API key, fichero no válido, fallo de la API).
    La app lo captura y muestra el mensaje al usuario."""


# ---------------------------------------------------------------- Funciones
def _read_image(image):
    """Acepta una ruta o bytes (p. ej. st.file_uploader(...).getvalue()).
    Comprueba con Pillow que es una imagen y la prepara para Gemini: la endereza,
    la reduce a MAX_SIDE px y la guarda como JPG. Devuelve (bytes, mime_type)."""
    if isinstance(image, (str, Path)):
        try:
            data = Path(image).read_bytes()
        except OSError as e:
            raise LabelExtractionError(f"No se puede leer el fichero: {image}") from e
    elif isinstance(image, bytes):
        data = image
    else:
        raise LabelExtractionError("La imagen debe ser una ruta o bytes")

    try:
        with Image.open(io.BytesIO(data)) as img:
            # Las fotos de móvil guardan el giro en los metadatos (EXIF): aquí se aplica de verdad
            img = ImageOps.exif_transpose(img).convert("RGB")
            # Una foto de móvil tiene ~4000 px y pesa varios MB: tarda en subir y Gemini
            # la reduce igualmente. Reducirla aquí hace la petición mucho más rápida
            img.thumbnail((MAX_SIDE, MAX_SIDE))
            out = io.BytesIO()
            # Guardar como JPG resuelve también el MPO (las imágenes extra se descartan)
            img.save(out, format="JPEG", quality=85)
    except (UnidentifiedImageError, OSError) as e:
        raise LabelExtractionError("El fichero no es una imagen válida") from e
    return out.getvalue(), "image/jpeg"


def _to_dict(result):
    """Pasa la respuesta de Gemini a un dict plano con los nombres de columna del dataset."""
    if not result.is_food_label:
        # Si no es una etiqueta, no se devuelve ningún dato aunque Gemini haya rellenado algo:
        # A03 solo tiene que mirar los dos flags
        return {"is_food_label": False, "has_nutrition_table": False, "product_name": None,
                "pnns_groups_2": "unknown", "ingredients_text": None, "additives": [],
                **{column: None for column in NUTRIENT_COLUMNS.values()}}
    out = {
        "is_food_label": result.is_food_label,
        "has_nutrition_table": result.has_nutrition_table,
        "product_name": result.product_name,
        "pnns_groups_2": result.category.value,
        "ingredients_text": result.ingredients_text,
        "additives": sorted(set(a.upper().replace(" ", "") for a in result.additives)),
    }
    nutrients = result.nutrients.model_dump()
    # Sin tabla, los nutrientes se fuerzan a None aunque Gemini haya devuelto algo:
    # así nunca llega a la app un valor inventado
    for field, column in NUTRIENT_COLUMNS.items():
        out[column] = nutrients[field] if result.has_nutrition_table else None
    return out


def _make_client():
    """Cliente de Gemini con reintentos automáticos (los hace el propio SDK).
    Si la API devuelve un error temporal (RETRY_CODES) o falla la conexión, espera
    2 s, 4 s, 8 s... (backoff exponencial) y vuelve a intentarlo, hasta 4 intentos.
    Cada intento tiene un límite de TIMEOUT_MS: si Gemini no contesta, se corta y se reintenta."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "tu_api_key":
        raise LabelExtractionError("Falta GEMINI_API_KEY en el fichero .env")
    retry = types.HttpRetryOptions(attempts=4, initial_delay=2, http_status_codes=RETRY_CODES)
    return genai.Client(api_key=api_key,
                        http_options=types.HttpOptions(retry_options=retry, timeout=TIMEOUT_MS))


def extract_label(images, client=None, model=None):
    """Fotos de un producto -> dict con nutrientes por 100 g, ingredientes, aditivos y categoría.

    `images`: una foto o una lista de fotos del MISMO producto (ruta o bytes cada una).
    Gemini las recibe todas juntas, así la tabla y los ingredientes pueden estar en caras distintas.
    Si las fotos no son una etiqueta o no tienen tabla nutricional, lo indica con
    is_food_label / has_nutrition_table (A03 avisa al usuario). Los errores técnicos
    lanzan LabelExtractionError. `client` sirve para pasar un cliente falso en pruebas.
    """
    if not isinstance(images, (list, tuple)):     # una sola foto -> lista de una
        images = [images]
    if not images:
        raise LabelExtractionError("Sube al menos una foto")
    if len(images) > MAX_PHOTOS:
        raise LabelExtractionError(f"Máximo {MAX_PHOTOS} fotos por producto")
    photos = [types.Part.from_bytes(data=data, mime_type=mime)
              for data, mime in map(_read_image, images)]   # valida cada foto antes de llamar a la API

    try:
        if client is None:
            client = _make_client()
        response = client.models.generate_content(
            model=model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
            contents=[*photos, PROMPT],                 # todas las fotos + las instrucciones
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=LabelExtraction,   # obliga a Gemini a seguir el esquema
                # Leer una etiqueta no necesita razonar mucho: pensar poco = respuesta más rápida
                thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
            ),
        )
    # Cualquier fallo técnico sale como LabelExtractionError, nunca como excepción "cruda"
    except LabelExtractionError:              # p. ej. falta la API key: ya tiene buen mensaje
        raise
    except errors.APIError as e:              # la API contesta con error (401, 429, 503...)
        raise LabelExtractionError(f"Error de la API de Gemini ({e.code}): {e.message}") from e
    except Exception as e:                    # sin internet, timeout, configuración no válida...
        # Captura amplia a propósito: la librería de red del SDK cambia entre versiones
        # (httpx, httpx2, aiohttp), así que no dependemos de sus clases de error.
        # `from e` conserva el error original para depurar
        raise LabelExtractionError(
            f"No se pudo contactar con Gemini ({type(e).__name__}). Revisa tu conexión") from e

    result = response.parsed                       # objeto LabelExtraction ya validado
    if result is None:                             # respuesta bloqueada o cortada
        raise LabelExtractionError("Gemini no devolvió una respuesta válida. Prueba otra foto")
    return _to_dict(result)


if __name__ == "__main__":
    # Prueba rápida. Todas las fotos que pases son del MISMO producto:
    # python src/label_extraction.py foto_tabla.jpg foto_ingredientes.jpg
    # Muestra en pantalla los reintentos del SDK, para saber qué pasa mientras espera
    logging.basicConfig(format="%(message)s")
    logging.getLogger("google_genai._api_client").setLevel(logging.INFO)
    print("Enviando fotos a Gemini... (puede tardar hasta un minuto)")
    try:
        print(json.dumps(extract_label(sys.argv[1:]), indent=2, ensure_ascii=False))
    except LabelExtractionError as e:
        print(f"ERROR: {e}")
