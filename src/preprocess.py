# =====================================================
# Preprocesamiento del dataset Abalone
#
# Ejecutar desde la carpeta principal del proyecto (tp5-abalone):
#     uv run src/preprocess.py
# =====================================================

import json
from datetime import date

import pandas as pd


# -----------------------------------------------------
# 1. Leer el dataset original
# -----------------------------------------------------
df = pd.read_csv("data/raw/abalone_raw.csv")
filas_iniciales = len(df)

# Acá vamos anotando cada cosa que le hacemos al dataset.
# Al final, esta lista se guarda en data_log.json
transformaciones = []


# -----------------------------------------------------
# 2. Nueva columna: edad en años
# -----------------------------------------------------
df["age"] = df["rings"] + 1.5
transformaciones.append("added derived feature 'age = rings + 1.5'")


# -----------------------------------------------------
# 3. Codificar la variable sex
#    sex tiene 3 valores (M, F, I) y los modelos solo entienden números.
#    get_dummies crea una columna por valor, con 1 o 0:
#      sex = "M"  ->  sex_F=0, sex_I=0, sex_M=1
# -----------------------------------------------------
df = pd.get_dummies(df, columns=["sex"], dtype=int)
transformaciones.append("encoded sex with one-hot encoding (sex_F, sex_I, sex_M)")


# -----------------------------------------------------
# 4. Eliminar valores atípicos de height
#    Criterio: height <= 0 es imposible (un animal sin altura)
#              height > 0.4 son valores aislados (el resto no pasa de ~0.25)
# -----------------------------------------------------
altura_min = 0
altura_max = 0.4

# Columna de True/False: True si la fila es atípica
es_atipico = (df["height"] <= altura_min) | (df["height"] > altura_max)

cantidad_eliminada = es_atipico.sum()   # cuenta cuántos True hay
df = df[~es_atipico]                    # me quedo con las filas que NO son atípicas

transformaciones.append(
    f"removed {cantidad_eliminada} outliers with height <= {altura_min} or height > {altura_max}"
)


# -----------------------------------------------------
# 5. Guardar el dataset limpio
# -----------------------------------------------------
df.to_csv("data/processed/abalone_clean.csv", index=False)


# -----------------------------------------------------
# 6. Guardar el registro de lo que hicimos (data_log.json)
# -----------------------------------------------------
registro = {
    "version": "v1",
    "created_at": str(date.today()),
    "source": "data/raw/abalone_raw.csv",
    "output": "data/processed/abalone_clean.csv",
    "rows_before": filas_iniciales,
    "rows_after": len(df),
    "transformations": transformaciones,
    "parameters": {"age_offset": 1.5, "height_min": altura_min, "height_max": altura_max},
}


with open("data/data_log.json", "w") as archivo:
    json.dump(registro, archivo, indent=2)


print("Listo!")
print("Filas antes:", filas_iniciales)
print("Filas después:", len(df))