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

## Reconstruir el dataset

```bash
python src/build_dataset.py
```

Lee `data/raw/off_products.csv.gz` y regenera `data/interim/` y `data/processed/off_nova_clean.csv.gz`, con los pasos 1-8 de `notebooks/01_eda.ipynb` y el mismo split fijo. Tarda unos minutos.

## Reentrenar con feedback

La app pregunta al usuario si el NOVA predicho es correcto y, si no, cuál es el correcto. Cada respuesta se guarda como una fila en `data/feedback/feedback.csv` (el fichero se crea solo la primera vez y no se sube a git; el formato está en `data/feedback/feedback_template.csv`).

```
app ─► guardar_feedback(fila) ─► data/feedback/feedback.csv
                                        │
          (una vez por semana)          ▼
                              python -m src.retrain
                                        │
        dataset limpio + feedback válido ─► XGBoost nuevo + métricas
```

**Guardar feedback (desde la app):**

```python
from src.feedback import guardar_feedback

guardar_feedback({
    "nova_predicho": 4, "es_correcto": "no", "nova_correcto": 3,
    "product_name": "...", "fat_100g": 10.0,  # + el resto de columnas del modelo
})
```

Valida la fila (NOVA entre 1 y 4, `es_correcto` "si"/"no" coherente con los NOVA, al menos un nutriente) y lanza `ValueError` si no sirve. `id_feedback` y `fecha` (UTC, ISO 8601) se rellenan solos. La etiqueta que se usa para entrenar es siempre `nova_correcto`.

**Reentrenar (manual):**

```bash
docker compose run --rm --no-deps jupyter python -m src.retrain
```

El script:
1. Lee el feedback y limpia **en memoria** (el CSV no se modifica): quita duplicados (`id_feedback`, y el mismo producto repetido: se queda el más reciente), filas inválidas y productos que están en el test (fuga de datos).
2. Entrena desde cero con el dataset limpio + todo el feedback acumulado. Cada fila de feedback pesa 0,5. El test no cambia, así que el F1 «antes» y «después» son comparables.
3. Guarda `models/m08_xgboost_retrained.joblib` (**no** sustituye al modelo anterior: se decide a mano cuál usa la app) y añade una entrada a `results/F03-metrics.json` con fecha, filas usadas, filas nuevas desde el último reentrenamiento y las métricas por clase.

Con 0 feedback reproduce el F1 macro de M08 (0,9611). Avisa si hay menos de 100 filas válidas.

**Por qué siempre desde cero:** con ~48.000 productos tarda segundos, permite retirar filas malas y evita que el modelo «olvide» el dataset. Si el volumen creciera mucho, el siguiente paso sería reentrenar solo cuando el acierto real baje de un umbral y usar una ventana reciente más una muestra del histórico.

## Estado

- [x] Fase 1 · EDA y limpieza del dataset
- [ ] Fase 2 · Modelos y comparación
- [ ] Fase 3 · App (Streamlit + Gemini)
- [ ] Fase 4 · Informe y demo