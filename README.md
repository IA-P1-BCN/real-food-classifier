# Real Food Classifier 🥦

Proyecto grupal de clasificación multiclase del bootcamp IA School (Factoría F5).

A partir de la información de la etiqueta de un producto del supermercado (nutrientes por 100 g, ingredientes y aditivos), el modelo predice su **grado de procesamiento NOVA** (1: sin procesar · 2: ingredientes culinarios · 3: procesado · 4: ultraprocesado).

## Equipo

- Viviana Andrango
- Damaris Castro
- Charlotte Doulcet

## Dataset

[Open Food Facts](https://world.openfoodfacts.org/data), base de datos abierta de productos alimentarios (licencia ODbL). Trabajamos con los productos vendidos en España.

Los datos en bruto no se suben al repo. Para descargarlos:

```bash
mkdir -p data/raw data/interim
curl -L -o data/raw/off_products.csv.gz https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz
```

## Estructura

```
data/
  raw/          export original (no se sube)
  interim/      datos filtrados a España (no se sube)
  processed/    dataset limpio compartido
notebooks/      EDA y experimentos
src/            código reutilizable
app/            aplicación (Streamlit)
```

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate   # en fish: source .venv/bin/activate.fish
pip install -r requirements.txt
```

## Docker (entorno común)

Alternativa a la instalación local: todas usamos las mismas versiones de Python y de las librerías. Requisito: Docker Desktop abierto.

```bash
cp .env.example .env          # después añade tu propia GEMINI_API_KEY en .env
docker compose build          # construye la imagen (solo la primera vez o si cambia requirements.txt)
docker compose up jupyter     # solo Jupyter: http://127.0.0.1:8888 (el token sale en los logs)
docker compose up             # Jupyter + Streamlit (http://127.0.0.1:8501), cuando exista app/app.py
```

Para parar: `Ctrl+C`.

**Notebooks en VS Code:** abre el notebook, **Select Kernel → Existing Jupyter Server**, y pega la URL con el token que aparece en los logs. También puedes usar tu entorno local en lugar de Docker.

**Seguridad:** Los puertos están ligados a `127.0.0.1`, así que solo son accesibles desde tu propio ordenador.

## Estado

🚧 En curso: análisis exploratorio (EDA).
