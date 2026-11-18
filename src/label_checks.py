"""Comprobaciones lógicas de los valores de una etiqueta antes de predecir.

Son las mismas reglas que se usaron para limpiar el dataset (corregir_atipicos en
src/build_dataset.py, paso 3 de E01), más un aviso específico para la energía en kJ.
Las usa la app con lo que escribe el usuario; en A03 sirven también para lo que lee Gemini.

Uso:  from src.label_checks import check_label
"""

KJ_PER_KCAL = 4.184


def estimated_kcal(v):
    """Energía con los factores estándar: 9 kcal/g grasa, 4 hidratos, 4 proteína, 2 fibra."""
    return (9 * (v.get("fat_100g") or 0) + 4 * (v.get("carbohydrates_100g") or 0)
            + 4 * (v.get("proteins_100g") or 0) + 2 * (v.get("fiber_100g") or 0))


def check_label(v, category=None):
    """Recibe los valores por 100 g con los nombres de columna del dataset.
    Devuelve (errores, avisos):
    - errores: valores imposibles. La app no predice hasta que se corrijan.
    - avisos: valores raros pero posibles. La app predice igualmente y los muestra."""
    errors, warnings = [], []
    energy = v.get("energy-kcal_100g")
    fat, sat = v.get("fat_100g"), v.get("saturated-fat_100g")
    carbs, sugars = v.get("carbohydrates_100g"), v.get("sugars_100g")

    # Una parte no puede ser mayor que su total (con 0,5 g de margen por el redondeo)
    if fat is not None and sat is not None and sat > fat + 0.5:
        errors.append("Las grasas saturadas no pueden ser más que las grasas totales.")
    if carbs is not None and sugars is not None and sugars > carbs + 0.5:
        errors.append("Los azúcares no pueden ser más que los hidratos de carbono.")

    # En 100 g no caben más de 100 g de nutrientes (con 1 g de margen)
    total = sum(v.get(c) or 0 for c in ["fat_100g", "carbohydrates_100g", "proteins_100g", "salt_100g"])
    if total > 101:
        errors.append(f"Grasas, hidratos, proteínas y sal suman {total:.0f} g: no caben en 100 g.")

    if energy is not None:
        estimated = estimated_kcal(v)
        if energy > 900:
            # El máximo posible es ~900 kcal (grasa pura). Más que eso casi seguro son kJ
            errors.append(f"{energy:.0f} kcal es imposible en 100 g (el máximo es ~900). "
                          f"¿Son kJ? En kcal serían {energy / KJ_PER_KCAL:.0f}.")
        elif estimated >= 3 and 3 <= energy / estimated <= 5.5:
            # Si la energía es ~4,2 veces la que dan los nutrientes, probablemente son kJ
            warnings.append(f"La energía parece estar en kJ: con estos nutrientes saldrían unas "
                            f"{estimated:.0f} kcal. Si es así, escribe {energy / KJ_PER_KCAL:.0f} y vuelve a predecir.")
        elif abs(energy - estimated) > 100 and category != "Alcoholic beverages":
            # El alcohol aporta energía que no sale de estos nutrientes, por eso se excluye
            warnings.append(f"La energía ({energy:.0f} kcal) no cuadra con los nutrientes "
                            f"(saldrían unas {estimated:.0f} kcal). Revisa los valores.")
    return errors, warnings


def check_ranges(v, names=None):
    """Revisa cada valor por separado: negativo o más de 100 g en 100 g es imposible.
    (La energía va aparte: su límite es 900 kcal y ya lo revisa check_label.)
    Devuelve (valores_limpios, avisos). Los valores imposibles pasan a None, así
    el campo queda vacío en el formulario y el usuario tiene que escribirlo él.
    `names`: {columna: nombre legible} para los mensajes. Si falta, se usa la columna."""
    names = names or {}
    clean, warnings = dict(v), []
    for column, value in v.items():
        # Solo los nutrientes por 100 g (terminan en _100g), sin la energía
        if not column.endswith("_100g") or column == "energy-kcal_100g" or value is None:
            continue
        if value < 0 or value > 100:
            clean[column] = None   # vacío: mejor pedirlo que meter un dato falso
            warnings.append(f"{names.get(column, column)}: {value} no es posible en 100 g. Escríbelo a mano.")
    # Energía negativa: tampoco puede ir al formulario (min_value=0)
    energy = v.get("energy-kcal_100g")
    if energy is not None and energy < 0:
        clean["energy-kcal_100g"] = None
        warnings.append(f"Energía: {energy} kcal no es posible. Escríbela a mano.")
    return clean, warnings