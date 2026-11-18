# Real Food Classifier 🥦

Clasificador del **grado de procesamiento NOVA** de un producto del supermercado (1: sin procesar · 2: ingredientes culinarios · 3: procesado · 4: ultraprocesado), a partir de su etiqueta: nutrientes por 100 g, ingredientes y aditivos.

Proyecto grupal del bootcamp IA School (Factoría F5, Barcelona).

**Equipo:** Viviana Andrango · Damaris Castro · Charlotte Doulcet

## Resultado: el modelo elegido es XGBoost

Comparamos 6 modelos con la misma métrica (validación cruzada de 5 folds, F1 macro). Elegimos **XGBoost** (notebook M08).

| Modelo | F1 macro (CV) | Sobreajuste (gap) |
|---|---|---|
| LightGBM | 0,965 | 0,035 |
| **XGBoost (elegido)** | **0,965** | **0,027** |
| Random Forest | 0,961 | 0,031 |
| Regresión logística | 0,951 | 0,001 |
| Árbol de decisión | 0,948 | 0,020 |
| KNN | 0,935 | 0,065 |

- **Por qué XGBoost:** empata con LightGBM en F1 (la diferencia está dentro del ruido), pero sobreajusta menos y acierta más en NOVA 3, la clase más difícil.
- **Nota final en test** (20 % de productos nunca vistos): **F1 macro 0,961**. Recall por clase: NOVA 1 0,978 · NOVA 2 0,990 · NOVA 3 0,944 · NOVA 4 0,960.
- **Dónde falla:** sobre todo confunde ultraprocesados (4) con procesados (3).

**Límite importante:** Open Food Facts calcula NOVA en parte con reglas basadas en los ingredientes y aditivos, que son también nuestras variables. El 0,96 mide en parte cuánto reproducimos esas reglas, no la precisión frente a un criterio experto. Sin las variables de ingredientes el F1 baja a 0,86. Detalle y propuestas de mejora en la conclusión de [`notebooks/04_m08_compare_models.ipynb`](notebooks/04_m08_compare_models.ipynb).

## De dónde salen los datos

- **Fuente:** [Open Food Facts](https://world.openfoodfacts.org/data), base de datos abierta de productos alimentarios (licencia ODbL). Trabajamos con los productos vendidos en España que tienen NOVA.
- **Datos limpios:** `data/processed/off_nova_clean.csv.gz` (48.119 productos, con el split train/test ya marcado). Está en el repositorio.
- **Datos en bruto:** no se suben. Solo hacen falta para reconstruir el dataset:

```bash
mkdir -p data/raw data/interim
curl -L -o data/raw/off_products.csv.gz https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz
```

## Cómo empezar

### Con Docker (recomendado: todas usamos las mismas versiones)

Requisito: Docker Desktop abierto.

```bash
cp .env.example .env          # después añade tu propia GEMINI_API_KEY en .env
docker compose build          # construye la imagen (solo la primera vez o si cambia requirements.txt)
docker compose up jupyter     # solo Jupyter: http://127.0.0.1:8888 (el token sale en los logs)
docker compose up             # Jupyter + Streamlit (http://127.0.0.1:8501), cuando exista app/app.py
```

Para parar: `Ctrl+C`.

**Notebooks en VS Code:** abre el notebook, **Select Kernel → Existing Jupyter Server**, y pega la URL con el token que aparece en los logs.

**Seguridad:** los puertos están ligados a `127.0.0.1`, así que solo son accesibles desde tu propio ordenador.

### Sin Docker

Necesitas Python 3.12 y Git.

```bash
python3 -m venv .venv
source .venv/bin/activate   # en fish: source .venv/bin/activate.fish
pip install -r requirements.txt
```

## Qué hay en los notebooks

| Notebook | Qué hace |
|---|---|
| `01_eda.ipynb` | Limpieza del dataset y creación de variables |
| `02_eda_analysis.ipynb` | Análisis exploratorio y conclusiones del EDA |
| `03_m01_logistic_regression.ipynb` | M01 · Regresión logística |
| `03_m02_knn.ipynb` | M02 · KNN |
| `03_m03_random_forest.ipynb` | M03 · Random Forest |
| `03_m05_xgboost.ipynb` | M05 · XGBoost |
| `03_m06_lightgbm.ipynb` | M06 · LightGBM |
| `03_m09_decision_tree.ipynb` | M09 · Árbol de decisión |
| `04_m08_compare_models.ipynb` | M08 · Compara los 6 modelos, elige el mejor, lo guarda y evalúa en test |

Cada modelo guarda sus métricas en `results/Mxx-metrics.json`.

## Estructura del repositorio

```
├── data/
│   ├── raw/          Export original de Open Food Facts (no se sube)
│   ├── interim/      Datos filtrados a España (no se sube)
│   └── processed/    Dataset limpio compartido
├── notebooks/        EDA (01-02), un notebook por modelo (03_mXX) y la comparación (04_m08)
├── src/              Código reutilizable (dataset, modelado, extracción con Gemini)
├── results/          Métricas de cada modelo (Mxx-metrics.json) y matriz de confusión final
├── reports/          Figuras sueltas de modelos
├── models/           Modelos .joblib (no se suben: se regeneran con el notebook M08)
├── app/              Aplicación Streamlit
├── docker-compose.yml, Dockerfile, requirements.txt
└── .env.example      Variables de entorno (copiar a .env, nunca subir .env)
```

## Regenerar el dataset y el modelo

**Dataset** (necesita los datos en bruto, tarda unos minutos):

```bash
python src/build_dataset.py
```

Regenera `data/interim/` y `data/processed/off_nova_clean.csv.gz` con los pasos 1-8 de `notebooks/01_eda.ipynb` y el mismo split fijo.

**Modelo final** (el que usará la app). `models/` no está en git, así que hay que crearlo:

1. Abre `notebooks/04_m08_compare_models.ipynb` y ejecútalo de principio a fin. Solo necesita ficheros que sí están en el repo (`data/processed/off_nova_clean.csv.gz` y `results/M05-metrics.json`).
2. Se crean `models/m08_xgboost.joblib`, `results/M08-metrics.json` y `results/M08-confusion_matrix.png`.

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
- [x] Fase 2 · Modelos y comparación (modelo elegido: XGBoost)
- [ ] Fase 3 · App (Streamlit + Gemini) · **en curso**: la extracción de la etiqueta con Gemini está hecha, falta la app
- [ ] Fase 4 · Informe y demo
