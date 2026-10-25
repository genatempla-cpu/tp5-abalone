# =====================================================
# Entrenamiento y evaluación del modelo (Abalone)
#
# Ejecutar desde la carpeta principal del proyecto (tp5-abalone):
#     uv run src/train_model.py
# =====================================================

import json
import os
from datetime import datetime

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

VERSION = "v1"
SEMILLA = 42          # la misma semilla en el split y en el modelo
TEST_SIZE = 0.2       # 20% de los datos para test
HIPERPARAMETROS = {"n_estimators": 100, "max_depth": 10}


# -----------------------------------------------------
# 1. Cargar el dataset limpio (salida del preprocesamiento)
# -----------------------------------------------------
df = pd.read_csv("data/processed/abalone_clean.csv")

# -----------------------------------------------------
# 2. Separar variables de entrada (X) y objetivo (y)
#    Sacamos "rings" de X: age = rings + 1.5, así que si rings
#    quedara como entrada, el modelo tendría la respuesta en la mano.
# -----------------------------------------------------
X = df.drop(columns=["age", "rings"])
y = df["age"]

print("X:", X.shape, "| y:", y.shape)

# -----------------------------------------------------
# 3. Dividir en entrenamiento y test (con semilla fija)
# -----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEMILLA
)

print("Train:", X_train.shape, "| Test:", X_test.shape)

# -----------------------------------------------------
# 4. Pipeline: escalador + modelo
#    El Pipeline hace que el escalador se ajuste SOLO con X_train
#    al llamar a .fit(). Al predecir sobre X_test, usa lo aprendido
#    en train, sin "ver" los datos de test.
#    (RandomForest en realidad no necesita escalado, pero así el
#    Pipeline queda listo si después se cambia el modelo.)
# -----------------------------------------------------
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestRegressor(**HIPERPARAMETROS, random_state=SEMILLA)),
])

pipe.fit(X_train, y_train)

# -----------------------------------------------------
# 5. Evaluar sobre test
# -----------------------------------------------------
y_pred = pipe.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")

# -----------------------------------------------------
# 6. Guardar el modelo entrenado (serializado y comprimido)
# -----------------------------------------------------
joblib.dump(pipe, f"models/model_{VERSION}.pkl", compress=3)

# -----------------------------------------------------
# 7. Guardar métricas, hiperparámetros y semilla (metrics_v1.json)
# -----------------------------------------------------
fecha_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
metricas = {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}

resultado = {
    "version": VERSION,
    "created_at": fecha_hora,
    "data": "data/processed/abalone_clean.csv",
    "model": "RandomForestRegressor",
    "target": "age",
    "features": list(X.columns),
    "random_state": SEMILLA,
    "test_size": TEST_SIZE,
    "n_train": len(X_train),
    "n_test": len(X_test),
    "hyperparameters": HIPERPARAMETROS,
    "metrics": metricas,
}

with open(f"results/metrics_{VERSION}.json", "w") as archivo:
    json.dump(resultado, archivo, indent=2)

# -----------------------------------------------------
# 8. Registrar la ejecución en experiments_log.csv
#    Cada vez que se corre el script se AGREGA una fila
#    (no se borra lo anterior): es el historial de experimentos.
# -----------------------------------------------------
ruta_log = "results/experiments_log.csv"
ya_existe = os.path.exists(ruta_log)

# El id es correlativo: exp_001, exp_002, ...
if ya_existe:
    cantidad_previa = len(pd.read_csv(ruta_log))
else:
    cantidad_previa = 0

experimento = {
    "experiment_id": f"exp_{cantidad_previa + 1:03d}",
    "datetime": fecha_hora,
    "model_version": VERSION,
    "model": "RandomForestRegressor",
    "hyperparameters": json.dumps(HIPERPARAMETROS),
    "random_state": SEMILLA,
    "MAE": metricas["MAE"],
    "RMSE": metricas["RMSE"],
    "R2": metricas["R2"],
}

# mode="a" = append (agregar al final). El encabezado se escribe solo la primera vez.
pd.DataFrame([experimento]).to_csv(ruta_log, mode="a", header=not ya_existe, index=False)

print("Modelo, métricas y registro guardados.")