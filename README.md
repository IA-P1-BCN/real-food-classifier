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

## Estado

🚧 En curso: análisis exploratorio (EDA).