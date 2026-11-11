"""
Reconstruye el dataset limpio de NOVA desde el export de Open Food Facts.

Reproduce los pasos 1-8 de notebooks/01_eda.ipynb (ahí están explicadas las decisiones).

Uso, desde la raíz del repo:
    python src/build_dataset.py

Entrada:  data/raw/off_products.csv.gz  (cómo descargarlo: ver README)
Salidas:  data/interim/off_spain.csv.gz
          data/processed/off_nova_clean.csv.gz
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Rutas relativas al propio script: funciona desde cualquier carpeta
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "off_products.csv.gz"
INTERIM = ROOT / "data" / "interim" / "off_spain.csv.gz"
PROCESSED = ROOT / "data" / "processed" / "off_nova_clean.csv.gz"

USECOLS = [
    "code", "product_name", "brands",
    "countries_tags", "categories_tags", "pnns_groups_1", "pnns_groups_2",
    "ingredients_text", "additives_n", "additives_tags",
    "nova_group",
    "energy-kcal_100g", "fat_100g", "saturated-fat_100g",
    "carbohydrates_100g", "sugars_100g", "fiber_100g",
    "proteins_100g", "salt_100g", "sodium_100g",
    "fruits-vegetables-legumes_100g", "ingredients_tags", "ingredients_analysis_tags",
]

# Nutrientes principales: sin ninguno de ellos, el producto no tiene tabla nutricional
NUTRIENTES = ["energy-kcal_100g", "fat_100g", "saturated-fat_100g",
              "carbohydrates_100g", "sugars_100g", "proteins_100g", "salt_100g"]

MACROS_G = ["fat_100g", "saturated-fat_100g", "carbohydrates_100g", "sugars_100g",
            "fiber_100g", "proteins_100g", "salt_100g"]


def energia_estimada(df):
    """Energía con los factores estándar: 9 kcal/g grasa, 4 hidratos, 4 proteína, 2 fibra."""
    return (9 * df["fat_100g"] + 4 * df["carbohydrates_100g"]
            + 4 * df["proteins_100g"] + 2 * df["fiber_100g"].fillna(0))


# ---------------------------------------------------------------- Paso 1
def cargar_espana():
    """Lee el export por trozos y se queda con los productos vendidos en España."""
    trozos = []
    for chunk in pd.read_csv(
            RAW, sep="\t", usecols=USECOLS,
            dtype={"code": str},          # CORRECCIÓN: sin esto se pierden los ceros iniciales del código
            chunksize=200_000, low_memory=False, on_bad_lines="skip",
    ):
        trozos.append(chunk[chunk["countries_tags"].str.contains("en:spain", na=False)])

    df = pd.concat(trozos, ignore_index=True)
    INTERIM.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(INTERIM, index=False)

    # Releemos el fichero intermedio igual que el notebook, para que los tipos de columna sean los mismos
    return pd.read_csv(INTERIM, low_memory=False, dtype={"code": str})


def filtrar_nova(df):
    """Solo productos con NOVA (sin clase no sirven para entrenar)."""
    df = df[df["nova_group"].notna()].copy()
    df["nova_group"] = df["nova_group"].astype(int)
    return df


# ---------------------------------------------------------------- Pasos 2 y 3
def quitar_columnas(df):
    """Países (ya filtrado), categorías largas (las resume pnns) y sodio (= sal / 2,5)."""
    return df.drop(columns=["countries_tags", "categories_tags", "sodium_100g"])


# ---------------------------------------------------------------- Paso 4
def corregir_atipicos(df):
    """Valores imposibles a nulo; productos con macros incoherentes, fuera."""
    for c in MACROS_G:
        df.loc[(df[c] < 0) | (df[c] > 100), c] = np.nan

    df["fruits-vegetables-legumes_100g"] = df["fruits-vegetables-legumes_100g"].clip(0, 100)
    df.loc[(df["energy-kcal_100g"] < 0) | (df["energy-kcal_100g"] > 900), "energy-kcal_100g"] = np.nan

    # Una parte mayor que su total: no sabemos cuál falla → las dos a nulo
    m = df["sugars_100g"] > df["carbohydrates_100g"] + 0.5
    df.loc[m, ["sugars_100g", "carbohydrates_100g"]] = np.nan
    m = df["saturated-fat_100g"] > df["fat_100g"] + 0.5
    df.loc[m, ["saturated-fat_100g", "fat_100g"]] = np.nan

    # Macros que suman más de 101 g → se elimina el producto
    suma = df[["fat_100g", "carbohydrates_100g", "proteins_100g", "salt_100g"]].sum(axis=1, min_count=1)
    df = df[~(suma > 101)].copy()

    # Energía que no cuadra con los macros (salvo alcohol) → nulo; se recalcula en el paso 5
    m = ((df["energy-kcal_100g"] - energia_estimada(df)).abs() > 100) \
        & (df["pnns_groups_1"] != "Alcoholic beverages")
    df.loc[m, "energy-kcal_100g"] = np.nan

    df.loc[df["additives_n"] > 30, "additives_n"] = np.nan
    return df


# ---------------------------------------------------------------- Paso 5
def tratar_nulos(df):
    """Solo arreglos deterministas: el resto se imputa en la fase 2, dentro del pipeline."""
    df = df[~df[NUTRIENTES].isna().all(axis=1)].copy()           # sin tabla nutricional → fuera

    df["sin_lista_ingredientes"] = df["ingredients_text"].isna().astype(int)
    df.loc[df["sin_lista_ingredientes"] == 1, "additives_n"] = 0
    df.loc[df["additives_n"] == 0, "additives_tags"] = ""

    falta_energia = (df["energy-kcal_100g"].isna()
                     & df[["fat_100g", "carbohydrates_100g", "proteins_100g"]].notna().all(axis=1)
                     & (df["pnns_groups_1"] != "Alcoholic beverages"))
    df.loc[falta_energia, "energy-kcal_100g"] = energia_estimada(df.loc[falta_energia])
    return df


# ---------------------------------------------------------------- Paso 6
def quitar_duplicados(df):
    """Códigos repetidos, copias del mismo producto y copias con NOVA contradictorio."""
    df = df.drop_duplicates(subset="code", keep="first").copy()

    nombre = df["product_name"].str.lower().str.strip()
    marca = df["brands"].str.lower().str.strip()
    df["grupo_producto"] = (nombre + " | " + marca).fillna(df["code"])

    claves_copia = ["grupo_producto"] + NUTRIENTES
    n_nova = df.groupby(claves_copia, dropna=False)["nova_group"].transform("nunique")
    df = df[~(n_nova > 1)].copy()

    return df.drop_duplicates(subset=claves_copia, keep="first").copy()


# ---------------------------------------------------------------- Paso 7
def _tiene(texto, patron):
    """1 si el texto contiene el patrón (expresión regular), 0 si no."""
    return texto.str.contains(patron, regex=True).astype(int)


def contar_ingredientes(texto):
    """Ingredientes principales: separa por comas, sin contar los de dentro de paréntesis."""
    if pd.isna(texto):
        return np.nan
    anterior = None
    while anterior != texto:                                     # quita paréntesis anidados
        anterior = texto
        texto = re.sub(r"\([^()]*\)|\[[^\[\]]*\]", "", texto)
    return len([p for p in re.split(r"[,;]", texto) if p.strip()])


def crear_variables(df):
    """Marcadores de ultraprocesado (tiene_*) y número de ingredientes."""
    ing = df["ingredients_tags"].fillna("")
    adi = df["additives_tags"].fillna("")
    ana = df["ingredients_analysis_tags"].fillna("")

    df["tiene_aromas"] = _tiene(ing, r"en:flavouring")
    df["tiene_edulcorantes"] = (_tiene(ing, r"en:sweetener") | _tiene(adi, r"en:e9[56]\d")).astype(int)
    df["tiene_jarabes"] = _tiene(ing, r"syrup|en:maltodextrin|en:dextrose")
    df["tiene_almidon_modificado"] = (_tiene(ing, r"modified-starch") | _tiene(adi, r"en:e14\d\d")).astype(int)
    df["tiene_grasa_hidrogenada"] = _tiene(ing, r"hydrogenated")
    df["tiene_proteina_aislada"] = _tiene(ing, r"protein-isolate|whey|caseinate|hydroly[sz]ed")
    df["tiene_colorantes"] = _tiene(adi, r"en:e1\d\d")
    df["tiene_emulsionantes"] = _tiene(adi, r"en:e4\d\d")
    df["tiene_potenciadores"] = _tiene(adi, r"en:e6[23]\d")
    df["tiene_aceite_palma"] = _tiene(ana, r"(?:^|,)en:palm-oil(?:,|$)")

    df["n_ingredientes"] = df["ingredients_text"].apply(contar_ingredientes)
    df.loc[df["sin_lista_ingredientes"] == 1, "n_ingredientes"] = 1   # sin lista = un ingrediente
    df.loc[df["n_ingredientes"] == 0, "n_ingredientes"] = np.nan      # texto ilegible
    return df


# ---------------------------------------------------------------- Paso 8
def guardar(df):
    """Quita las listas de tags, añade el split fijo 80/20 y guarda el CSV."""
    df = df.drop(columns=["ingredients_tags", "additives_tags", "ingredients_analysis_tags"]).reset_index(drop=True)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    _, test_idx = next(sgkf.split(df, df["nova_group"], groups=df["grupo_producto"]))
    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"

    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED, index=False)
    return df


def main():
    pasos = [filtrar_nova, quitar_columnas, corregir_atipicos, tratar_nulos,
             quitar_duplicados, crear_variables]

    df = cargar_espana()
    print(f"{'cargar_espana':<20} {len(df):>8,}")
    for paso in pasos:
        df = paso(df)
        print(f"{paso.__name__:<20} {len(df):>8,}")

    df = guardar(df)

    # Comprobaciones finales
    grupos_train = set(df.loc[df["split"] == "train", "grupo_producto"])
    grupos_test = set(df.loc[df["split"] == "test", "grupo_producto"])
    assert not grupos_train & grupos_test, "Hay grupos de producto en train y test a la vez"

    print(f"\nGuardado: {PROCESSED.relative_to(ROOT)} · {df.shape[0]:,} filas × {df.shape[1]} columnas")
    print(pd.crosstab(df["nova_group"], df["split"], normalize="columns").round(3))


if __name__ == "__main__":
    main()