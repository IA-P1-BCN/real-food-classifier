# Real Food Classifier 🥦

¿Este producto del súper es ultraprocesado? Real Food Classifier predice su **grupo NOVA** (del 1 al 4) a partir de lo que pone en la etiqueta: nutrientes por 100 g, ingredientes y aditivos.

Proyecto grupal del bootcamp IA School (Factoría F5, Barcelona), noviembre 2026.

**Equipo:** [Viviana Andrango](https://github.com/alvi103-png) · [Damaris Castro](https://github.com/damcb1) · [Charlotte Doulcet](https://github.com/Charlottedoulcet)

> ⚠️ Proyecto educativo. No es consejo nutricional ni médico.

## ¿Qué es NOVA?

NOVA es una clasificación de alimentos creada por investigadores de la Universidad de São Paulo. Agrupa los alimentos según **cuánto y cómo se han procesado**, no según sus nutrientes. Por eso no es lo mismo que el Nutri-Score: un aceite de oliva puede tener mal Nutri-Score y ser NOVA 2, y un refresco *zero* puede tener buen Nutri-Score y ser NOVA 4.

| Grupo | Qué es | Ejemplos |
|---|---|---|
| **1 · Sin procesar** | Alimentos naturales o mínimamente procesados | Fruta, verdura, legumbres, huevos, leche |
| **2 · Ingrediente culinario** | Se extraen de alimentos del grupo 1 para cocinar | Aceites, sal, azúcar, mantequilla |
| **3 · Procesado** | Alimentos del grupo 1 con sal, azúcar o aceite añadidos | Conservas, quesos, pan artesano |
| **4 · Ultraprocesado** | Formulaciones industriales con aditivos que no usarías en casa | Refrescos, galletas, bollería, platos preparados |

## La app

Una app de Streamlit pensada para usar con el móvil en el súper:

1. Eliges la categoría del producto y rellenas la etiqueta nutricional y los ingredientes.
2. La app detecta sola los aditivos y los marcadores de ultraprocesado (aromas, edulcorantes, jarabes…) en la lista de ingredientes.
3. Comprueba que los valores tengan sentido (por ejemplo, que las grasas saturadas no superen las grasas totales).
4. Predice el grupo NOVA con el modelo XGBoost y muestra la probabilidad de cada grupo.
5. Muestra un aviso fijo abajo, y otro junto al resultado, de que es una estimación automática y no sustituye el consejo de un nutricionista.

Para arrancarla necesitas el modelo `models/m08_xgboost.joblib` (ver [Regenerar el dataset y el modelo](#regenerar-el-dataset-y-el-modelo)) y Docker:

```bash
docker compose up streamlit   # http://127.0.0.1:8501
```

**Próximamente:** leer la etiqueta a partir de una foto con Gemini (el código ya está en `src/label_extraction.py`) y preguntar al usuario si la predicción es correcta para reentrenar el modelo.

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

## Limitaciones

- **El modelo aprende en parte las reglas de Open Food Facts.** Open Food Facts calcula NOVA en parte con reglas basadas en los ingredientes y aditivos, que son también nuestras variables. El 0,96 mide en parte cuánto reproducimos esas reglas, no la precisión frente a un criterio experto. Sin las variables de ingredientes, el F1 baja a 0,86.
- **Quitamos los productos sin tabla nutricional.** El modelo los necesita, pero así la clase NOVA 2 pierde ~30 % de sus productos (aceites, sal, azúcar… muchas veces no traen tabla).
- **Solo productos vendidos en España.** Con otros países podría funcionar peor (otros ingredientes, otros idiomas en la etiqueta).
- **Las etiquetas son de voluntarios.** Open Food Facts es colaborativo, así que algunos datos tienen errores. En la limpieza corregimos los más evidentes.

Detalle y propuestas de mejora en las conclusiones de [`02_eda_analysis.ipynb`](notebooks/02_eda_analysis.ipynb) y [`04_m08_compare_models.ipynb`](notebooks/04_m08_compare_models.ipynb).

## Datos

- **Fuente:** [Open Food Facts](https://world.openfoodfacts.org/data), base de datos abierta y colaborativa de productos alimentarios (licencia ODbL). Trabajamos con los productos vendidos en España que tienen grupo NOVA.
- **Datos limpios:** `data/processed/off_nova_clean.csv.gz`, con 48.119 productos y el split train/test ya marcado en la columna `split` (80/20, agrupado por nombre + marca para que un mismo producto no esté en los dos lados). Está en el repositorio.
- **Datos en bruto:** no se suben. Solo hacen falta para reconstruir el dataset:

```bash
mkdir -p data/raw data/interim
curl -L -o data/raw/off_products.csv.gz https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz
```

### Variables del modelo

| Tipo | Variables |
|---|---|
| Categoría | `pnns_groups_2`: 42 subcategorías de alimento (pan, quesos, refrescos…) |
| Nutrientes por 100 g | energía (kcal), grasas, grasas saturadas, hidratos, azúcares, fibra, proteínas, sal, % de fruta/verdura/legumbres |
| Ingredientes | `n_ingredientes` (nº de ingredientes), `additives_n` (nº de aditivos), `sin_lista_ingredientes` |
| Marcadores de ultraprocesado | `tiene_*`: aromas, edulcorantes, jarabes, almidón modificado, proteína aislada, colorantes, emulsionantes, potenciadores del sabor, aceite de palma, conservantes, sal, azúcar, aceite |

La variable objetivo es `nova_group` (1-4).

## Cómo empezar

### Con Docker (recomendado: todas usamos las mismas versiones)

Requisito: Docker Desktop abierto.

```bash
cp .env.example .env          # después añade tu propia GEMINI_API_KEY en .env
docker compose build          # construye la imagen (solo la primera vez o si cambia requirements.txt)
docker compose up jupyter     # solo Jupyter: http://127.0.0.1:8888 (el token sale en los logs)
docker compose up             # Jupyter + la app de Streamlit (http://127.0.0.1:8501)
```

Para parar: `Ctrl+C`.

**Notebooks desde tu IDE** (VS Code, IntelliJ, PyCharm…): conecta el notebook a un servidor de Jupyter externo y pega la URL con el token que aparece en los logs. En VS Code es **Select Kernel → Existing Jupyter Server**.

**Seguridad:** los puertos están ligados a `127.0.0.1`, así que solo son accesibles desde tu propio ordenador. El fichero `.env` tiene tu clave de API: nunca lo subas (ya está en `.gitignore`).

### Sin Docker

Necesitas Python 3.12 y Git.

```bash
python3 -m venv .venv
source .venv/bin/activate   # en fish: source .venv/bin/activate.fish
pip install -r requirements.txt
jupyter lab                 # notebooks
streamlit run app/app.py    # la app (desde la raíz del repo)
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

Cada modelo guarda sus métricas en `results/Mxx-metrics.json` y algunas figuras en `reports/`.

## Estructura del repositorio

```
├── app/
│   ├── app.py               App de Streamlit
│   └── style.css            Estilos propios (el tema está en .streamlit/config.toml)
├── data/
│   ├── raw/                 Export original de Open Food Facts (no se sube)
│   ├── interim/             Datos filtrados a España (no se sube)
│   ├── processed/           Dataset limpio compartido
│   └── feedback/            Respuestas de los usuarios (solo se sube la plantilla)
├── notebooks/               EDA (01-02), un notebook por modelo (03_mXX) y la comparación (04_m08)
├── src/
│   ├── build_dataset.py     Reconstruye el dataset limpio desde los datos en bruto
│   ├── modeling.py          Carga de datos, variables y rutas comunes a todos los modelos
│   ├── ingredient_markers.py  Detecta aditivos y marcadores tiene_* en la lista de ingredientes
│   ├── label_checks.py      Comprueba que los valores de la etiqueta tengan sentido
│   ├── label_extraction.py  Lee la etiqueta desde fotos con Gemini
│   ├── predict.py           Carga el modelo final y predice el grupo NOVA
│   ├── feedback.py          Valida y guarda el feedback de los usuarios
│   └── retrain.py           Reentrena el modelo con el feedback
├── results/                 Métricas de cada modelo (Mxx-metrics.json) y matriz de confusión final
├── reports/                 Figuras de los modelos
├── models/                  Modelos .joblib (no se suben: se regeneran con el notebook M08)
├── Dockerfile, docker-compose.yml, requirements.txt
└── .env.example             Variables de entorno (copiar a .env, nunca subir .env)
```

## Regenerar el dataset y el modelo

**Dataset** (necesita los datos en bruto, tarda unos minutos):

```bash
python src/build_dataset.py
```

Con Docker: `docker compose run --rm --no-deps jupyter python src/build_dataset.py`

Regenera `data/interim/` y `data/processed/off_nova_clean.csv.gz` con los pasos 1-8 de `notebooks/01_eda.ipynb` y el mismo split fijo.

**Modelo final** (el que usará la app). `models/` no está en git, así que hay que crearlo:

1. Abre `notebooks/04_m08_compare_models.ipynb` y ejecútalo de principio a fin. Solo necesita ficheros que sí están en el repo (`data/processed/off_nova_clean.csv.gz` y `results/M05-metrics.json`).
2. Se crean `models/m08_xgboost.joblib`, `results/M08-metrics.json` y `results/M08-confusion_matrix.png`.

## Reentrenar con feedback

La idea es que la app pregunte al usuario si el NOVA predicho es correcto y, si no, cuál es el correcto (**todavía no está conectado a la app**: ver [Estado](#estado)). Cada respuesta se guarda como una fila en `data/feedback/feedback.csv`. El fichero se crea solo la primera vez y no se sube a git. El formato está en `data/feedback/feedback_template.csv`.

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

## Cómo trabajamos

- **Tablero de tareas:** [GitHub Projects](https://github.com/orgs/IA-P1-BCN/projects/57). Cada tarea tiene un ID (E01, M05, A04…).
- **Ramas:** `main` (versión estable), `dev` (integración) y una rama por tarea desde `dev`: `feat/<ID>-descripcion`, `fix/…`, `docs/…`, `refactor/…`.
- **Pull requests:** siempre a `dev`, con al menos 1 aprobación de otra compañera. Las ramas remotas no se borran después del merge, para que quede el historial de cada tarea.
- **Commits:** en inglés y con [Conventional Commits](https://www.conventionalcommits.org/) (`feat(app): …`, `fix(retrain): …`, `docs(readme): …`).

## Estado

- [x] **Fase 1 · EDA y limpieza** del dataset
- [x] **Fase 2 · Modelos** y comparación (modelo elegido: XGBoost)
- [ ] **Fase 3 · App y feedback** · en curso
    - [x] Formulario de la etiqueta y predicción con el modelo final (A01, A04)
    - [x] Extracción de la etiqueta desde fotos con Gemini (`src/label_extraction.py`)
    - [x] Pipeline de reentrenamiento con feedback (F03)
    - [ ] Foto con Gemini dentro de la app
    - [ ] Guardar el feedback del usuario desde la app (F01)
    - [ ] Métricas en vivo a partir del feedback (F02)
- [ ] **Fase 4 · Informe y demo**