"""A02: extrae los datos de la foto de una etiqueta con Gemini.

Uso desde la app:   from src.label_extraction import extract_label
Uso en terminal:    python src/label_extraction.py foto.jpg
"""
import io
import json
import os
import sys
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

load_dotenv()  # lee GEMINI_API_KEY (y GEMINI_MODEL, opcional) del fichero .env

DEFAULT_MODEL = "gemini-3.6-flash"

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
Extract the data into the response schema. Rules:
- Use only what is visible in the photo. Never guess or invent values: use null if not visible.
- Nutrients: use the "per 100 g" or "per 100 ml" column, never the per-serving column.
- If energy is only in kJ, convert to kcal (kJ / 4.184). If only sodium is given, salt = sodium * 2.5.
- has_nutrition_table: false if there is no nutrition table in the photo. Then all nutrients are null.
- is_food_label: false if the photo is not a food label or is too blurry to read.
- ingredients_text: copy the ingredient list as written, in its original language.
- additives: E-numbers in the ingredients, written as 'E' + number (e.g. E330). If an additive is
  named without its number (e.g. "citric acid"), add its E-number only if you are sure.
- category: the closest category to the product. Use "unknown" if you cannot tell."""


class LabelExtractionError(Exception):
    """Error técnico (sin API key, fichero no válido, fallo de la API).
    La app lo captura y muestra el mensaje al usuario."""


# ---------------------------------------------------------------- Funciones
def _read_image(image):
    """Acepta una ruta o bytes (p. ej. st.file_uploader(...).getvalue()).
    Devuelve (bytes, mime_type) y comprueba con Pillow que de verdad es una imagen."""
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
            fmt = img.format                     # "JPEG", "PNG", "WEBP"...
    except UnidentifiedImageError as e:
        raise LabelExtractionError("El fichero no es una imagen válida") from e

        
    # Las fotos de móvil a veces salen como MPO: un JPG normal con imágenes extra pegadas
    # al final (profundidad, modo retrato). Gemini lee la primera, así que vale como JPG
    mime = "image/jpeg" if fmt == "MPO" else Image.MIME.get(fmt)
    if mime not in {"image/jpeg", "image/png", "image/webp"}:
        raise LabelExtractionError(f"Formato no soportado: {fmt}. Usa JPG, PNG o WEBP")
    return data, mime


def _to_dict(result):
    """Pasa la respuesta de Gemini a un dict plano con los nombres de columna del dataset."""
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


def extract_label(image, client=None, model=None):
    """Foto de una etiqueta -> dict con nutrientes por 100 g, ingredientes, aditivos y categoría.

    Si la foto no es una etiqueta o no tiene tabla nutricional, lo indica con
    is_food_label / has_nutrition_table (A03 avisa al usuario). Los errores técnicos
    lanzan LabelExtractionError. `client` sirve para pasar un cliente falso en pruebas.
    """
    data, mime = _read_image(image)

    if client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "tu_api_key":
            raise LabelExtractionError("Falta GEMINI_API_KEY en el fichero .env")
        client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model=model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
            contents=[types.Part.from_bytes(data=data, mime_type=mime), PROMPT],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=LabelExtraction,   # obliga a Gemini a seguir el esquema
            ),
        )
    except errors.APIError as e:
        raise LabelExtractionError(f"Error de la API de Gemini ({e.code}): {e.message}") from e

    result = response.parsed                       # objeto LabelExtraction ya validado
    if result is None:                             # respuesta bloqueada o cortada
        raise LabelExtractionError("Gemini no devolvió una respuesta válida. Prueba otra foto")
    return _to_dict(result)


if __name__ == "__main__":
    # Prueba rápida: python src/label_extraction.py foto1.jpg foto2.png
    for path in sys.argv[1:]:
        print(f"--- {path}")
        try:
            print(json.dumps(extract_label(path), indent=2, ensure_ascii=False))
        except LabelExtractionError as e:
            print(f"ERROR: {e}")