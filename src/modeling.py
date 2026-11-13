"""Preprocesado y evaluación comunes a los modelos de la fase 2."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# Ruta relativa a este fichero. Para importar el módulo desde notebooks/ hace falta
# añadir la raíz al path: sys.path.insert(0, "..") en la primera celda
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "off_nova_clean.csv.gz"
MODELS_DIR = ROOT / "models"      # modelos .joblib (carpeta ignorada por git: se regeneran con el notebook)
REPORTS_DIR = ROOT / "reports"    # métricas en JSON (se suben al repo para comparar modelos)

TARGET = "nova_group"
GROUP = "grupo_producto"
LABELS = [1, 2, 3, 4]
CATEGORY = "pnns_groups_2"
SKEWED = ["n_ingredientes", "additives_n", "salt_100g"]   # log1p (acuerdo E02)
EXCLUDED = ["tiene_grasa_hidrogenada"]                    # 0,5 % de productos


def load_train(path=DATA_PATH):
    """Solo train. El test no se toca hasta M08."""
    df = pd.read_csv(path)
    return df[df["split"] == "train"].reset_index(drop=True)


def get_feature_columns(df, drop=()):
    """Devuelve las columnas del modelo agrupadas por tipo.
    `drop` sirve para las variantes (p. ej. sin energy-kcal_100g y saturated-fat_100g)."""
    drop = set(drop) | set(EXCLUDED)
    nutrients = [c for c in df.columns if c.endswith("_100g")]
    numeric = [c for c in nutrients + ["additives_n", "n_ingredientes"] if c not in drop]
    binary = [c for c in df.columns
              if (c.startswith("tiene_") or c == "sin_lista_ingredientes") and c not in drop]
    return {"numeric": numeric, "binary": binary, "category": [CATEGORY],
            "all": numeric + binary + [CATEGORY]}


def build_preprocessor(cols, log_skewed=True, scale=True):
    """ColumnTransformer: se ajusta solo con los datos que recibe en fit
    (dentro de la CV, solo con el train de cada fold).
    log_skewed=True y scale=True para regresión logística y KNN;
    log_skewed=False y scale=False para árboles (acuerdo E02)."""
    skewed = [c for c in SKEWED if c in cols["numeric"]] if log_skewed else []
    other_num = [c for c in cols["numeric"] if c not in skewed]

    # add_indicator solo crea columna para las variables con nulos en train (fibra, sal, etc.):
    # los nulos no son aleatorios (se concentran en NOVA 2), así que el aviso aporta información
    skewed_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        *([("scale", StandardScaler())] if scale else []),
    ])
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        *([("scale", StandardScaler())] if scale else []),
    ])
    cat_pipe = OneHotEncoder(min_frequency=50, handle_unknown="infrequent_if_exist")

    return ColumnTransformer([
        ("skewed", skewed_pipe, skewed),          # lista vacía en el baseline → se ignora
        ("num", num_pipe, other_num),
        ("cat", cat_pipe, cols["category"]),
        ("bin", "passthrough", cols["binary"]),
    ])


def evaluate_cv(model, X, y, groups, n_splits=5, random_state=42):
    """Convenio de métricas del equipo: F1 macro train/CV, gap, recall por clase, matriz."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_validate(model, X, y, groups=groups, cv=cv, scoring="f1_macro",
                            return_train_score=True, n_jobs=-1)
    # Predicciones out-of-fold (mismos folds: random_state fijo) para matriz y recall
    pred = cross_val_predict(model, X, y, groups=groups, cv=cv, n_jobs=-1)

    f1_train = scores["train_score"].mean()
    f1_cv = scores["test_score"].mean()
    recalls = recall_score(y, pred, labels=LABELS, average=None, zero_division=0)
    return {
        "f1_macro_train": round(float(f1_train), 4),
        "f1_macro_cv": round(float(f1_cv), 4),
        "f1_macro_cv_std": round(float(scores["test_score"].std()), 4),
        "gap": round(float(f1_train - f1_cv), 4),             # objetivo < 0.05
        "recall": {f"nova_{k}": round(float(r), 4) for k, r in zip(LABELS, recalls)},
        "confusion_matrix": confusion_matrix(y, pred, labels=LABELS).tolist(),  # serializable a JSON
    }
