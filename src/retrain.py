import json
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.metrics import f1_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.feedback import COLUMNS, FEEDBACK_PATH, NUTRIENTES, es_valida
from src.modeling import CATEGORY, DATA_PATH, LABELS, MODELS_DIR, RESULTS_DIR, TARGET

PESO_FEEDBACK = 0.5
MIN_FILAS_AVISO = 100


def cargar_datos():
    m05 = json.loads((RESULTS_DIR / "M05-metrics.json").read_text())
    df = pd.read_csv(DATA_PATH, dtype={"code": str})
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    return train, test, m05["features"], m05["params"]


def preparar_X(df, features, categorias):
    X = df[features].copy()
    X[CATEGORY] = pd.Categorical(X[CATEGORY], categories=categorias)
    numericas = [c for c in features if c != CATEGORY]
    X[numericas] = X[numericas].astype(float)
    return X


def entrenar(train, features, params):
    categorias = sorted(train[CATEGORY].dropna().unique())
    X = preparar_X(train, features, categorias)
    y = train[TARGET] - 1
    extra = train["peso"].fillna(1.0).to_numpy() if "peso" in train else 1.0
    pesos = compute_sample_weight("balanced", y) * extra
    modelo = XGBClassifier(**params)
    modelo.fit(X, y, sample_weight=pesos)
    return modelo, categorias


def evaluar(modelo, test, features, categorias):
    X = preparar_X(test, features, categorias)
    y_real = test[TARGET]
    y_pred = modelo.predict(X) + 1
    recalls = recall_score(y_real, y_pred, labels=LABELS, average=None)
    return {
        "macro_f1": round(float(f1_score(y_real, y_pred, average="macro")), 4),
        "recall": {f"nova_{k}": round(float(r), 4) for k, r in zip(LABELS, recalls)},
    }


def cargar_feedback(path=FEEDBACK_PATH):
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS + [TARGET, "peso"]), {"total": 0, "usadas": 0}
    df = pd.read_csv(path).drop_duplicates("id_feedback")
    total = len(df)
    df = df[df.apply(lambda fila: es_valida(fila.to_dict()), axis=1)]
    clave = df["product_name"].str.strip().str.lower().fillna(df["id_feedback"])
    df = df.assign(clave=clave).sort_values("fecha").drop_duplicates("clave", keep="last")
    df = df.drop(columns="clave")
    df[TARGET] = df["nova_correcto"].astype(int)
    df["peso"] = PESO_FEEDBACK
    return df.reset_index(drop=True), {"total": total, "usadas": len(df)}


def _claves_producto(df):
    nombre = df["product_name"].str.strip().str.lower()
    return pd.concat([nombre, df[NUTRIENTES].round(2)], axis=1)


def quitar_fuga_test(feedback, test):
    if feedback.empty:
        return feedback, 0
    claves_test = _claves_producto(test).dropna(subset=["product_name"]).drop_duplicates()
    cruce = _claves_producto(feedback).merge(claves_test, how="left", indicator=True)
    en_test = (cruce["_merge"] == "both").to_numpy()
    return feedback[~en_test].reset_index(drop=True), int(en_test.sum())


def main(feedback_path=FEEDBACK_PATH):
    train, test, features, params = cargar_datos()
    feedback, resumen = cargar_feedback(feedback_path)
    feedback, fuga = quitar_fuga_test(feedback, test)
    if len(feedback) < MIN_FILAS_AVISO:
        print(f"Aviso: solo {len(feedback)} filas de feedback válidas, el modelo apenas va a cambiar")

    base, categorias = entrenar(train, features, params)
    antes = evaluar(base, test, features, categorias)

    conjunto = train
    if not feedback.empty:
        conjunto = pd.concat([train, feedback[features + [TARGET, "peso"]]], ignore_index=True)
    modelo, categorias = entrenar(conjunto, features, params)
    despues = evaluar(modelo, test, features, categorias)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {"model": modelo, "features": features, "pnns_categories": categorias, "label_offset": 1},
        MODELS_DIR / "m08_xgboost_retrained.joblib",
    )

    ruta = RESULTS_DIR / "F03-metrics.json"
    historial = json.loads(ruta.read_text()) if ruta.exists() else []
    ultimo = historial[-1]["fecha"] if historial else ""
    historial.append({
        "fecha": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "feedback_total": resumen["total"],
        "feedback_validas": resumen["usadas"],
        "feedback_descartadas_por_fuga_test": fuga,
        "feedback_usadas": len(feedback),
        "feedback_nuevas_desde_el_ultimo_reentrenamiento": int((feedback["fecha"] > ultimo).sum()),
        "feedback_fecha_mas_reciente": feedback["fecha"].max() if not feedback.empty else None,
        "peso_feedback": PESO_FEEDBACK,
        "n_train": len(conjunto),
        "n_test": len(test),
        "test_antes": antes,
        "test_despues": despues,
    })
    ruta.write_text(json.dumps(historial, indent=2))

    print(f"Feedback: {resumen['total']} filas, {len(feedback)} usadas, {fuga} descartadas por fuga con el test")
    print("Test antes:  ", antes)
    print("Test después:", despues)


if __name__ == "__main__":
    main()
